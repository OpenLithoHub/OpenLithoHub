"""Multi-process tile inference for `openlithohub optimize run`.

RFC 0004 picks `torch.multiprocessing.spawn` over tile shards as the v0.3
direction. The model layer stays untouched so ONNX/TorchScript export
keeps returning a bare ``nn.Module``; parallelism wraps the *tile loop*,
not the model.

v0.7 (WS-E): shared-weight dispatch. The parent process loads model weights
into CPU shared memory *before* spawning workers. Workers memory-map the
weights instead of independently re-instantiating and reloading, reducing
peak memory from O(N × model_size) to O(model_size + N × activation_size).

Compile-cache: when ``--compile`` is active, the Inductor cache directory
is set in the parent so workers reuse compiled kernels without per-worker
recompilation.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue as queue_mod
import tempfile
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from openlithohub.workflow.tiling import Tile

_DEFAULT_TIMEOUT_SECONDS = 0.5


def _share_weights(
    model_name: str,
    model_kwargs: dict[str, Any],
) -> dict[str, torch.Tensor] | None:
    """Instantiate the model, move state_dict to CPU shared memory, return it.

    Returns ``None`` if the model cannot be instantiated (e.g. missing weights).
    The caller is responsible for keeping the shared tensors alive until all
    workers have finished loading.
    """
    try:
        from openlithohub.models.registry import register_builtin_models, registry

        register_builtin_models()
        model = registry.get(model_name, **model_kwargs)
        model.setup()

        state_dict = model.state_dict()
        shared: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            t = tensor.detach().cpu()
            t.share_memory_()
            shared[key] = t

        model.teardown()
        return shared
    except Exception:
        return None


def _setup_compile_cache() -> str | None:
    """Set up a persistent Inductor cache directory for ``torch.compile``.

    Returns the cache directory path, or None if torch.compile is not active.
    """
    if os.environ.get("TORCH_COMPILE", "").lower() not in ("1", "true", "yes"):
        return None

    cache_dir = os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "openlithohub_inductor_cache"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    return cache_dir


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

    The parent loads model weights into CPU shared memory before spawning.
    Workers memory-map these weights, reducing peak memory from
    O(N x model_size) to approximately O(model_size + N x activation_size).

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

    # Set up compile cache if torch.compile is active
    compile_cache_dir = _setup_compile_cache()

    # Load weights into CPU shared memory
    weight_state = _share_weights(model_name, model_kwargs)

    ctx = mp.get_context("spawn")
    queue: mp.Queue[Any] = ctx.Queue()
    processes: list[Any] = []

    for rank, indices in enumerate(shards):
        payload = [(idx, tiles[idx].tensor.detach().cpu().numpy()) for idx in indices]
        p = ctx.Process(
            target=_worker,
            args=(
                rank,
                effective,
                model_name,
                model_kwargs,
                base_perf_kwargs,
                payload,
                queue,
                weight_state,
                compile_cache_dir,
            ),
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
                            f"rank={i} exit={p.exitcode}"
                            for i, p in enumerate(processes)
                            if not p.is_alive()
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
    shared_state: dict[str, torch.Tensor] | None = None,
    compile_cache_dir: str | None = None,
) -> None:
    try:
        from openlithohub.models.registry import register_builtin_models, registry

        register_builtin_models()
        device_str = _resolve_worker_device(
            rank, num_workers, base_perf_kwargs.get("device", "cpu")
        )
        worker_perf_kwargs = {**base_perf_kwargs, "device": device_str}

        # Set compile cache in worker if configured
        if compile_cache_dir is not None:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = compile_cache_dir

        model = registry.get(model_name, **model_kwargs)

        # Load shared weights if available
        if shared_state is not None:
            model.load_state_dict(shared_state, strict=False)

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
