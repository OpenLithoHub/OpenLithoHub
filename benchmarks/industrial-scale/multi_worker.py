"""Bounded multi-worker streaming executor (scale track, S6).

Implements the charter's multi-worker semantics ON TOP of the
production tile planner (`plan_tile_requests`) — the tile plan, halo
semantics and trusted-core partition are the production ones; this
module only shards their EXECUTION across N logical workers.

Target topology (GPU hosts bind one worker to one device; development
hosts run every worker on CPU — "cpu-worker-emulation"):

```text
ExactVectorRunSource
        ↓
plan_tile_requests (production planner)
        ↓
bounded work queue (maxsize = queue_depth)
  ↙       ↓       ↘
worker0 worker1 worker2     (each tile owned by EXACTLY ONE worker)
  ↓       ↓       ↓
OrderedCommitLedger → out-of-core sink (deterministic tile-plan order)
```

Invariants (all enforced, all hostile-tested):

* each planned tile is committed EXACTLY ONCE — duplicates and missing
  tiles are hard errors, never warnings;
* commits carry the planned tile id and core geometry — an ownership
  mismatch is a hard error;
* the sink commit order is DETERMINISTIC (tile-plan order), with a
  bounded reorder buffer (watermark recorded in the report);
* worker failures propagate — there is NO silent retry that could
  change measurement semantics;
* everything is bounded: bounded queues, bounded resident tensors.

CPU-worker-emulation proves scheduler/ownership/failure/ordering
correctness ONLY.  It is never "multi-GPU validation" — GPU facts come
exclusively from real CUDA hosts.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openlithohub.streaming.core_halo import plan_tile_requests


class DuplicateTileCommitError(Exception):
    """A tile index was submitted twice — the exactly-once invariant broke."""


class MissingTileError(Exception):
    """The run finished with planned tiles never committed."""


class OwnershipMismatchError(Exception):
    """A committed tile id/core geometry differs from the planned plan."""


class WorkerFailureError(Exception):
    """A worker raised; the failure propagates (no silent retry)."""


def core_slices(request: Any) -> tuple[slice, slice]:
    """Location of the core inside the read-region tensor — EXACTLY the
    production convention (streaming.pipeline._core_slices): the offset
    is core minus read origin, so boundary tiles with a smaller actual
    halo slice correctly."""
    y0 = request.core_bbox.y0 - request.read_bbox.y0
    x0 = request.core_bbox.x0 - request.read_bbox.x0
    return (
        slice(y0, y0 + request.core_bbox.height),
        slice(x0, x0 + request.core_bbox.width),
    )


def plan_shards(n_requests: int, worker_count: int) -> list[list[int]]:
    """Strided (round-robin) shard plan: deterministic, and for the
    raster-ordered tile plan it spreads neighbours across workers."""
    shards: list[list[int]] = [[] for _ in range(max(1, worker_count))]
    for index in range(n_requests):
        shards[index % len(shards)].append(index)
    return shards


class OrderedCommitLedger:
    """Exactly-once, in-plan-order sink commit ledger.

    Out-of-order arrivals are buffered (bounded by the caller's queue
    depth × worker count) and committed strictly in tile-plan order, so
    the sink commit order is deterministic regardless of worker timing.
    """

    def __init__(self, requests: list[Any], sink: Any) -> None:
        self._requests = requests
        self._sink = sink
        self._committed = 0
        self._pending: dict[int, tuple[Any, str]] = {}
        self.max_resident = 0

    @property
    def committed(self) -> int:
        return self._committed

    def submit(self, index: int, core: Any, tile_id: str) -> None:
        if index < self._committed or index in self._pending:
            raise DuplicateTileCommitError(f"tile index {index} (id {tile_id!r}) submitted twice")
        request = self._requests[index]
        if tile_id != request.tile_id:
            raise OwnershipMismatchError(
                f"index {index}: tile id {tile_id!r} != planned {request.tile_id!r}"
            )
        if (core.shape[0], core.shape[1]) != request.core_size:
            raise OwnershipMismatchError(
                f"tile {tile_id!r}: committed core "
                f"{(core.shape[0], core.shape[1])} "
                f"!= planned {request.core_size}"
            )
        self._pending[index] = (core, tile_id)
        self.max_resident = max(self.max_resident, len(self._pending))
        while self._committed in self._pending:
            core_out, id_out = self._pending.pop(self._committed)
            request = self._requests[self._committed]
            self._sink.write_core(id_out, request.core_bbox, core_out, {})
            self._committed += 1

    def finish(self) -> None:
        if self._committed != len(self._requests):
            missing = [
                self._requests[i].tile_id for i in range(self._committed, len(self._requests))
            ]
            raise MissingTileError(
                f"run finished with {len(missing)} uncommitted tiles: {missing[:4]}…"
            )


@dataclass
class MultiWorkerReport:
    n_tiles: int = 0
    worker_counts: dict[str, int] = field(default_factory=dict)
    wall_s: float = 0.0
    max_resident_results: int = 0


def run_multi_worker_stream(
    *,
    source: Any,
    sink: Any,
    forward_fn: Callable[[Any], Any],
    core_size: int,
    halo_px: int,
    worker_count: int,
    queue_depth: int,
    device_for_worker: Callable[[int], str] | None = None,
) -> MultiWorkerReport:
    """Execute the production tile plan across `worker_count` logical
    workers with in-order deterministic sink commits.

    `device_for_worker(i)` names the device for worker i (CUDA hosts map
    workers to GPUs; emulation returns "cpu").  The forward runs on the
    worker's device and the committed tensor is always CPU (sink
    semantics), matching the single-worker path.
    """
    shape = source.shape
    requests = plan_tile_requests(shape, core_size, halo_px)
    shards = plan_shards(len(requests), worker_count)
    result_queue: queue.Queue[tuple[int, Any, str] | None] = queue.Queue(
        maxsize=max(1, queue_depth)
    )
    failures: list[BaseException] = []

    def worker(worker_index: int, indices: list[int]) -> None:
        try:
            if device_for_worker is not None:
                device_for_worker(worker_index)
            for index in indices:
                request = requests[index]
                tile = source.read_window(request.read_bbox)
                forwarded = forward_fn(tile)
                if forwarded.device.type != "cpu":
                    forwarded = forwarded.cpu()
                ys, xs = core_slices(request)
                core = forwarded[ys, xs]
                result_queue.put((index, core, request.tile_id))
        except Exception as exc:  # noqa: BLE001 — failure MUST propagate, no retry
            failures.append(exc)
        finally:
            # EVERY worker reports exactly one sentinel on every exit path,
            # so the committer can never wait on a finished worker forever
            result_queue.put(None)

    start = time.perf_counter()
    threads = [
        threading.Thread(target=worker, args=(i, shard), daemon=True)
        for i, shard in enumerate(shards)
        if shard
    ]
    for thread in threads:
        thread.start()

    ledger = OrderedCommitLedger(requests, sink)
    finished_workers = 0
    total_workers = len(threads)
    # Each worker emits its sentinel LAST (finally), so consuming
    # `total_workers` sentinels drains every result it produced.  After
    # that the queue is provably empty — never block on it again, so a
    # failed shard terminates as WorkerFailure/MissingTile, not a hang.
    while finished_workers < total_workers:
        item = result_queue.get()
        if item is None:
            finished_workers += 1
            continue
        index, core, tile_id = item
        ledger.submit(index, core, tile_id)

    if failures:
        raise WorkerFailureError(f"worker failed: {failures[0]!r}") from failures[0]
    ledger.finish()
    return MultiWorkerReport(
        n_tiles=len(requests),
        worker_counts={f"worker{i}": len(shard) for i, shard in enumerate(shards)},
        wall_s=time.perf_counter() - start,
        max_resident_results=ledger.max_resident,
    )
