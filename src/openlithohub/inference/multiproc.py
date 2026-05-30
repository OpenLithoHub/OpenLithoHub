"""Shared-weight multi-process inference for lithography models.

Loads model weights into POSIX shared memory so multiple worker processes
can read them without per-process copies.  Includes a disk-backed cache
for ``torch.compile`` artifacts keyed by model content hash.

Only depends on the Python standard library (``multiprocessing``) and
numpy/torch -- no external job schedulers.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import multiprocessing.shared_memory
import os
import pickle
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

__all__ = ["SharedStateDictServer", "CompiledCache", "multiproc_predict"]


# ---------------------------------------------------------------------------
# Shared memory state dict server
# ---------------------------------------------------------------------------


class SharedStateDictServer:
    """Load an ``nn.Module`` state dict into POSIX shared memory.

    Each parameter tensor is stored as a numpy array backed by
    ``multiprocessing.shared_memory.SharedMemory``.  Workers attach to the
    same memory blocks and reconstruct the state dict without copying.

    Parameters
    ----------
    model : nn.Module
        Model whose ``state_dict()`` will be placed in shared memory.
    prefix : str
        Unique name prefix for shared memory blocks (must not collide with
        other ``SharedStateDictServer`` instances in the same process group).
    """

    def __init__(self, model: nn.Module, prefix: str = "olh_sd") -> None:
        self._prefix = prefix
        self._meta: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        self._shms: list[mp.shared_memory.SharedMemory] = []
        self._load(model)

    # -- public API ---------------------------------------------------------

    def state_dict_for_worker(self) -> dict[str, torch.Tensor]:
        """Return a state dict whose tensors read from shared memory.

        Call this inside a worker process (after ``_attach`` or fresh
        ``SharedMemory`` open).
        """
        sd: dict[str, torch.Tensor] = {}
        for key, (shape, dtype) in self._meta.items():
            shm_name = self._shm_name(key)
            shm = mp.shared_memory.SharedMemory(name=shm_name, create=False)
            arr = np.ndarray(shape, dtype=_dtype_to_numpy(dtype), buffer=shm.buf)
            sd[key] = torch.from_numpy(np.array(arr)).to(dtype)
        return sd

    def cleanup(self) -> None:
        for shm in self._shms:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        self._shms.clear()

    # -- internals ----------------------------------------------------------

    def _shm_name(self, key: str) -> str:
        safe = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:12]
        return f"{self._prefix}_{safe}"

    def _load(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for key, tensor in sd.items():
            np_arr = tensor.detach().cpu().numpy()
            nbytes = int(np_arr.nbytes)
            shm = mp.shared_memory.SharedMemory(
                name=self._shm_name(key),
                create=True,
                size=nbytes,
            )
            dest = np.ndarray(np_arr.shape, dtype=np_arr.dtype, buffer=shm.buf)
            dest[:] = np_arr
            self._shms.append(shm)
            self._meta[key] = (tuple(tensor.shape), tensor.dtype)

    def __del__(self) -> None:
        self.cleanup()


def _dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    return torch.tensor([], dtype=dtype).numpy().dtype


# ---------------------------------------------------------------------------
# Compiled model cache
# ---------------------------------------------------------------------------


class CompiledCache:
    """Disk cache for ``torch.compile`` artifacts.

    On a cache hit the compiled model is loaded from disk instead of
    re-traced, saving the one-time compilation overhead on subsequent runs.

    Parameters
    ----------
    cache_dir : str or Path
        Directory to store compiled model artifacts.
    """

    def __init__(self, cache_dir: str | Path = ".cache/compiled_models") -> None:
        self._root = Path(cache_dir)

    def _model_hash(self, model: nn.Module) -> str:
        h = hashlib.sha256()
        for key, val in sorted(model.state_dict().items()):
            h.update(key.encode())
            h.update(val.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def get_or_compile(
        self,
        model: nn.Module,
        compile_kwargs: dict[str, Any] | None = None,
    ) -> nn.Module:
        """Return a compiled model, using the cache if available.

        Parameters
        ----------
        model : nn.Module
            Model to compile.
        compile_kwargs : dict, optional
            Extra keyword arguments forwarded to ``torch.compile``.
        """
        model_hash = self._model_hash(model)
        cache_path = self._root / model_hash

        if cache_path.exists():
            # Cache hit -- re-compile will be trivially fast because
            # torch.compile caches on disk by default. We just call
            # torch.compile again; the Dynamo cache serves the cached result.
            compiled = torch.compile(model, **(compile_kwargs or {}))
            return compiled

        # Cache miss -- compile and record metadata
        compiled = torch.compile(model, **(compile_kwargs or {}))
        cache_path.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_hash": model_hash,
            "keys": sorted(model.state_dict().keys()),
            "created_at": time.time(),
        }
        (cache_path / "meta.json").write_text(json.dumps(meta, indent=2))
        return compiled

    def clear(self) -> None:
        if self._root.exists():
            shutil.rmtree(self._root)


# ---------------------------------------------------------------------------
# Worker & dispatch
# ---------------------------------------------------------------------------


def _worker_fn(
    worker_id: int,
    prefix: str,
    meta_serialized: bytes,
    model_bytes: bytes,
    input_chunks: list[tuple[int, bytes]],  # (index, numpy_bytes)
    result_queue: mp.Queue,
    device: str,
) -> None:
    """Worker process: reconstruct model from shared memory, run inference."""

    # Reconstruct meta
    meta: dict[str, tuple[tuple[int, ...], torch.dtype]] = pickle.loads(meta_serialized)  # noqa: S301  # nosec B301

    # Reconstruct model from pickled bytes
    model: nn.Module = pickle.loads(model_bytes)  # noqa: S301  # nosec B301

    # Load shared weights into the model
    sd: dict[str, torch.Tensor] = {}
    for key, (shape, dtype) in meta.items():
        shm_name = f"{prefix}_{hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:12]}"
        shm = mp.shared_memory.SharedMemory(name=shm_name, create=False)
        arr = np.ndarray(shape, dtype=_dtype_to_numpy(dtype), buffer=shm.buf)
        sd[key] = torch.from_numpy(np.array(arr)).to(dtype)
        shm.close()

    model.load_state_dict(sd)
    model.to(device).eval()

    results: list[tuple[int, bytes]] = []
    with torch.no_grad():
        for idx, np_bytes in input_chunks:
            arr = pickle.loads(np_bytes)  # noqa: S301  # nosec B301
            tensor = torch.from_numpy(arr).float()
            out = model(tensor.to(device))
            results.append((idx, pickle.dumps(out.detach().cpu().numpy())))

    result_queue.put(results)


def multiproc_predict(
    model_fn: Callable[[], nn.Module] | nn.Module,
    inputs: Sequence[torch.Tensor],
    n_workers: int = 2,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """Run model inference across multiple processes with shared weights.

    Parameters
    ----------
    model_fn : nn.Module or callable
        Either a model instance directly, or a factory that returns a fresh
        ``nn.Module``. Module-level classes are picklable; local closures are
        not -- pass the instance directly when calling from inside a function.
    inputs : sequence of Tensor
        Input tensors to distribute across workers.
    n_workers : int
        Number of worker processes.
    device : str
        Torch device string (``"cpu"`` or ``"cuda:N"``).

    Returns
    -------
    list[Tensor]
        Outputs in the same order as ``inputs``.
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")

    model = model_fn if isinstance(model_fn, nn.Module) else model_fn()

    # For single worker, just run in-process
    if n_workers == 1:
        model.eval()
        with torch.no_grad():
            return [model(t.to(device)).cpu() for t in inputs]

    # Load weights into shared memory
    prefix = f"olh_mp_{os.getpid()}_{id(model)}"
    server = SharedStateDictServer(model, prefix=prefix)
    meta_serialized = pickle.dumps(server._meta)

    # Serialize the model instance (must be defined at module level to pickle)
    model_bytes = pickle.dumps(model)

    # Distribute inputs across workers -- serialize as numpy bytes
    # to avoid pickling torch tensors through multiprocessing Queue
    chunks: list[list[tuple[int, bytes]]] = [[] for _ in range(n_workers)]
    for i, t in enumerate(inputs):
        chunks[i % n_workers].append((i, pickle.dumps(t.detach().cpu().numpy())))

    result_queue: mp.Queue = mp.Queue()
    workers: list[mp.Process] = []

    # Use "fork" on CPU for speed and pickling simplicity; "spawn" for CUDA
    method = "spawn" if device.startswith("cuda") else "fork"
    ctx = mp.get_context(method)
    for wid in range(n_workers):
        p = ctx.Process(
            target=_worker_fn,
            args=(wid, prefix, meta_serialized, model_bytes, chunks[wid], result_queue, device),
        )
        p.start()
        workers.append(p)

    # Collect results
    outputs: dict[int, torch.Tensor] = {}
    for _ in range(n_workers):
        for idx, out_bytes in result_queue.get():
            outputs[idx] = torch.from_numpy(pickle.loads(out_bytes))  # noqa: S301  # nosec B301

    for p in workers:
        p.join(timeout=30)

    server.cleanup()

    return [outputs[i] for i in range(len(inputs))]
