"""Multi-process tile inference for `openlithohub optimize run`.

RFC 0004 picks `torch.multiprocessing.spawn` over tile shards as the v0.3
direction. The model layer stays untouched so ONNX/TorchScript export
keeps returning a bare ``nn.Module``; parallelism wraps the *tile loop*,
not the model.

Each worker re-instantiates the model from the registry — workers receive
a ``(model_name, model_kwargs)`` factory rather than a live model, which
keeps pickling sane and avoids CUDA-context-fork hazards. Tiles are
sharded round-robin; the parent process keeps the canonical ``Tile``
geometry and stitches results returned over an ``mp.Queue``.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import queue as queue_mod
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from openlithohub.workflow.tiling import Tile

_DEFAULT_TIMEOUT_SECONDS = 0.5


def parallel_tile_inference(
    model_name: str,
    model_kwargs: dict[str, Any],
    tiles: list[Tile],
    *,
    num_gpus: int,
    base_perf_kwargs: dict[str, Any],
    progress_cb: Callable[[], None] | None = None,
) -> list[tuple[Tile, torch.Tensor]]:
    """Shard tiles round-robin across ``num_gpus`` worker processes.

    Each worker resolves the model from the registry, pins itself to its
    assigned device (``cuda:rank`` if enough GPUs are available, else
    ``cpu``), runs ``model.predict`` on its shard, and returns the masks
    over a queue. Results are returned in original tile order. The caller
    stitches.

    Falls back to CPU dispatch when fewer than ``num_gpus`` CUDA devices
    are visible — this is what makes CPU-only CI exercise the dispatch
    logic.
    """
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if not tiles:
        return []

    effective = min(num_gpus, len(tiles))
    shards = _round_robin_shards(len(tiles), effective)

    ctx = mp.get_context("spawn")
    queue: mp.Queue[Any] = ctx.Queue()
    processes: list[Any] = []

    for rank, indices in enumerate(shards):
        # Convert tensors to numpy arrays so pickle never needs to share
        # storage file descriptors across the spawn boundary — FD sharing
        # breaks on some Linux configs (EOFError in rebuild_storage_fd).
        payload = [(idx, tiles[idx].tensor.detach().cpu().numpy()) for idx in indices]
        p = ctx.Process(
            target=_worker,
            args=(rank, effective, model_name, model_kwargs, base_perf_kwargs, payload, queue),
            daemon=False,
        )
        p.start()
        processes.append(p)

    results: dict[int, torch.Tensor] = {}
    expected = len(tiles)
    try:
        while len(results) < expected:
            try:
                item = queue.get(timeout=_DEFAULT_TIMEOUT_SECONDS)
            except queue_mod.Empty:
                if any(not p.is_alive() for p in processes) and queue.empty():
                    dead = [p for p in processes if not p.is_alive()]
                    if dead:
                        codes = ", ".join(
                            f"rank={i} exit={p.exitcode}" for i, p in enumerate(processes) if not p.is_alive()
                        )
                        raise RuntimeError(
                            f"parallel_tile_inference: worker(s) exited ({codes}), "
                            f"received {len(results)}/{expected} results"
                        ) from None
                continue

            if isinstance(item, tuple) and item and item[0] == "error":
                _, rank, exc_repr, tb = item
                _terminate(processes)
                raise RuntimeError(
                    f"parallel_tile_inference: worker rank={rank} failed: {exc_repr}\n{tb}"
                )

            idx, mask_arr = item
            results[idx] = torch.from_numpy(mask_arr)
            if progress_cb is not None:
                progress_cb()
    except KeyboardInterrupt:
        _terminate(processes)
        raise
    finally:
        for p in processes:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)

    return [(tiles[idx], results[idx]) for idx in range(expected)]


def _round_robin_shards(num_tiles: int, num_workers: int) -> list[list[int]]:
    """Split ``range(num_tiles)`` into ``num_workers`` round-robin shards.

    Round-robin keeps shard sizes within 1 of each other regardless of how
    many tiles there are, which is the simplest balanced strategy when
    per-tile cost is roughly equal.
    """
    shards: list[list[int]] = [[] for _ in range(num_workers)]
    for i in range(num_tiles):
        shards[i % num_workers].append(i)
    return shards


def _resolve_worker_device(rank: int, num_workers: int, base_device: str) -> str:
    """Pick the device a worker should target.

    If CUDA is visible and we have at least ``num_workers`` devices, pin
    each rank to its own GPU. Otherwise dispatch on CPU regardless of what
    the user passed — workers can't share a single GPU usefully here, and
    falling back to CPU is what makes the CPU-only CI smoke valid.
    """
    if (
        base_device.startswith("cuda")
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= num_workers
    ):
        return f"cuda:{rank}"
    return "cpu"


def _worker(
    rank: int,
    num_workers: int,
    model_name: str,
    model_kwargs: dict[str, Any],
    base_perf_kwargs: dict[str, Any],
    payload: list[tuple[int, np.ndarray[Any, Any]]],
    result_queue: mp.Queue[Any],
) -> None:
    try:
        from openlithohub.models.registry import register_builtin_models, registry

        register_builtin_models()
        device_str = _resolve_worker_device(
            rank, num_workers, base_perf_kwargs.get("device", "cpu")
        )
        worker_perf_kwargs = {**base_perf_kwargs, "device": device_str}

        model = registry.get(model_name, **model_kwargs)
        model.setup()
        try:
            for idx, tile_arr in payload:
                tile_tensor = torch.from_numpy(tile_arr)
                result = model.predict(tile_tensor, **worker_perf_kwargs)
                mask = result.mask.detach().cpu().numpy()
                result_queue.put((idx, mask))
        finally:
            model.teardown()
    except BaseException as exc:  # noqa: BLE001 — propagate everything to the parent
        with contextlib.suppress(Exception):
            result_queue.put(("error", rank, repr(exc), traceback.format_exc()))
        raise


def _terminate(processes: list[Any]) -> None:
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join(timeout=5.0)
