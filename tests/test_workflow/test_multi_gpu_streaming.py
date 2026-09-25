"""Multi-worker streaming scheduler hostile tests (scale track, S6/S7).

Runs entirely WITHOUT klayout (numpy fake source — the production
planner and executor never touch klayout), so the whole hostile matrix
executes in every CI shard.  Proves the charter invariants on
CPU-emulated workers: exact 1/2/3-worker output parity, exactly-once
ownership, duplicate/missing/mismatch refusals, worker-failure
propagation, deterministic commit order and bounded queues.

CPU emulation validates SCHEDULER SEMANTICS ONLY — it is never
multi-GPU validation.  Real-GDS integration lives in the benchmark
shard (tests/test_benchmark/test_industrial_scale_harness.py).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from openlithohub.streaming import run_streaming
from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.sinks import TensorTileSink

REPO = Path(__file__).resolve().parents[2]
MULTI_WORKER = REPO / "benchmarks" / "industrial-scale" / "multi_worker.py"
WINDOW = 128
CORE = 32
HALO = 64

_MULTI_WORKER = None


def _load_multi_worker():
    global _MULTI_WORKER
    if _MULTI_WORKER is None:
        spec = importlib.util.spec_from_file_location("scale_multi_worker", MULTI_WORKER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["scale_multi_worker"] = module  # dataclasses needs the registration
        spec.loader.exec_module(module)
        _MULTI_WORKER = module
    return _MULTI_WORKER


class NumpyFakeSource:
    """Minimal TileSource over a numpy raster (klayout-free)."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array
        self.shape = array.shape

    def read_window(self, bbox) -> torch.Tensor:
        window = self._array[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
        return torch.from_numpy(window.copy())


@pytest.fixture(scope="module")
def routed_source() -> NumpyFakeSource:
    rng = np.random.default_rng(42)
    raster = rng.random((WINDOW, WINDOW)).astype(np.float32)
    return NumpyFakeSource(raster)


def _blur_forward():
    from openlithohub.benchmark.industrial_v2 import BatchedFiniteSupportBlur

    return BatchedFiniteSupportBlur(radius=4, sigma=1.6).window_forward


def _run_workers(source, forward_fn, sink, worker_count: int, queue_depth: int = 8):
    multi_worker = _load_multi_worker()
    return multi_worker.run_multi_worker_stream(
        source=source,
        sink=sink,
        forward_fn=forward_fn,
        core_size=CORE,
        halo_px=HALO,
        worker_count=worker_count,
        queue_depth=queue_depth,
    )


# ---- §10.4: 1/2/3-worker exact parity ------------------------------------------


def test_worker_outputs_bit_identical_across_worker_counts(routed_source) -> None:
    """§10.4: 1/2/3 workers must produce EXACTLY the same output bytes —
    the tile plan and per-tile arithmetic are worker-count independent."""
    forward = _blur_forward()
    outputs = []
    for worker_count in (1, 2, 3):
        sink = TensorTileSink(routed_source.shape)
        report = _run_workers(routed_source, lambda tile: forward(tile).cpu(), sink, worker_count)
        outputs.append(sink.finalize())
        assert report.n_tiles == (WINDOW // CORE) ** 2
        assert sum(report.worker_counts.values()) == report.n_tiles
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


def test_output_matches_single_worker_production_spine_within_frozen_tolerance(
    routed_source,
) -> None:
    """The executor output matches the single-worker production spine
    within the tolerance frozen BEFORE measurement (1e-5 fp32).  (The
    spine clamps requested halo to core-1 while the executor uses the
    full planned halo — a sub-tolerance boundary effect, frozen here.)"""
    reference_sink = TensorTileSink(routed_source.shape)
    forward = _blur_forward()
    run_streaming(
        routed_source,
        reference_sink,
        lambda tile: forward(tile).cpu(),
        core_size=CORE,
        halo_policy=LegacyFixedHaloPolicy(halo_px=HALO),
    )
    reference = reference_sink.finalize()

    sink = TensorTileSink(routed_source.shape)
    _run_workers(routed_source, lambda tile: forward(tile).cpu(), sink, 3)
    output = sink.finalize()
    assert torch.allclose(output, reference, atol=1e-5, rtol=0.0)


def test_three_workers_exactly_once_and_deterministic_order(routed_source) -> None:
    forward = _blur_forward()

    class RecordingSink:
        def __init__(self) -> None:
            self.shape = routed_source.shape
            self.order: list[str] = []

        def write_core(self, tile_id, bbox, tensor, metadata) -> None:
            self.order.append(tile_id)

        def finalize(self):
            return None

    orders = []
    report = None
    for _ in range(3):
        sink = RecordingSink()
        report = _run_workers(routed_source, lambda tile: forward(tile).cpu(), sink, 3)
        orders.append(sink.order)
    assert orders[0] == orders[1] == orders[2], "commit order is deterministic"
    assert orders[0][0] == "tile_0" and orders[0][-1] == f"tile_{report.n_tiles - 1}"
    assert len(orders[0]) == report.n_tiles


# ---- §10.3: duplicate / missing / ownership mismatch ----------------------------


class LedgerSink:
    def write_core(self, tile_id, bbox, tensor, metadata) -> None:
        pass


def test_duplicate_tile_commit_is_fatal() -> None:
    multi_worker = _load_multi_worker()
    from openlithohub.streaming.core_halo import plan_tile_requests

    requests = plan_tile_requests((WINDOW, WINDOW), CORE, HALO)
    ledger = multi_worker.OrderedCommitLedger(requests, LedgerSink())
    request = requests[0]
    ledger.submit(0, torch.zeros(request.core_size), request.tile_id)
    with pytest.raises(multi_worker.DuplicateTileCommitError):
        ledger.submit(0, torch.zeros(request.core_size), request.tile_id)


def test_missing_tile_is_fatal() -> None:
    multi_worker = _load_multi_worker()
    from openlithohub.streaming.core_halo import plan_tile_requests

    requests = plan_tile_requests((WINDOW, WINDOW), CORE, HALO)
    ledger = multi_worker.OrderedCommitLedger(requests, LedgerSink())
    for index in range(len(requests) - 1):  # never submit the last tile
        request = requests[index]
        ledger.submit(index, torch.zeros(request.core_size), request.tile_id)
    with pytest.raises(multi_worker.MissingTileError):
        ledger.finish()


def test_ownership_mismatch_is_fatal() -> None:
    multi_worker = _load_multi_worker()
    from openlithohub.streaming.core_halo import plan_tile_requests

    requests = plan_tile_requests((WINDOW, WINDOW), CORE, HALO)

    # wrong tile id for the index
    ledger = multi_worker.OrderedCommitLedger(requests, LedgerSink())
    request = requests[0]
    with pytest.raises(multi_worker.OwnershipMismatchError):
        ledger.submit(0, torch.zeros(request.core_size), "tile_999")

    # wrong core geometry for the planned tile
    ledger2 = multi_worker.OrderedCommitLedger(requests, LedgerSink())
    with pytest.raises(multi_worker.OwnershipMismatchError):
        ledger2.submit(0, torch.zeros(1, 1), requests[0].tile_id)


# ---- worker failure propagation (no silent retry) --------------------------------


def test_worker_failure_propagates(routed_source) -> None:
    multi_worker = _load_multi_worker()
    forward = _blur_forward()
    sink = TensorTileSink(routed_source.shape)
    first_read = True

    def failing_forward(tile):
        nonlocal first_read
        if first_read and tile.shape[0] > 1:
            first_read = False
            raise RuntimeError("worker exploded mid-shard")
        return forward(tile)

    with pytest.raises(multi_worker.WorkerFailureError, match="worker exploded"):
        _run_workers(routed_source, failing_forward, sink, 3)


# ---- boundedness ------------------------------------------------------------------


def test_bounded_queue_and_resident_watermark(routed_source) -> None:
    forward = _blur_forward()
    sink = TensorTileSink(routed_source.shape)
    report = _run_workers(routed_source, lambda tile: forward(tile).cpu(), sink, 3, queue_depth=1)
    assert report.n_tiles == (WINDOW // CORE) ** 2
    # resident reorder buffer never exceeds workers × queue depth (+ commit edge)
    assert report.max_resident_results <= 3 * 1 + 3


def test_shard_plan_is_an_exact_partition() -> None:
    multi_worker = _load_multi_worker()
    shards = multi_worker.plan_shards(17, 3)
    flat = sorted(i for shard in shards for i in shard)
    assert flat == list(range(17)), "every tile planned exactly once"
    assert all(len(shard) >= 5 for shard in shards)
