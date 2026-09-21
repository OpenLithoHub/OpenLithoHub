"""FastAPI app exposing the OpenLithoHub optimization engine over HTTP.

Endpoints:
  - GET  /v1/health   — liveness probe.
  - GET  /v1/models   — list registered model names.
  - POST /v1/optimize — multipart upload of a layout file + model name,
    returns the optimized layout binary.

Models are loaded lazily on first request and cached in-process; repeat
requests against the same model skip weight loading entirely. The cache
is keyed by ``(name, frozenset(kwargs.items()))`` so a `pretrained=True`
variant does not collide with the bare model.

Concurrency
-----------
The endpoint is ``async def`` but the underlying optimization is pure
CPU/GPU work, so we dispatch it to a worker thread via
``asyncio.to_thread`` to keep the event loop responsive (issue #36).
The model cache is guarded by ``_CACHE_LOCK`` so two concurrent
load-on-miss requests for the same key cannot both build the model and
race to evict each other (issue #37). Per-model ``predict()`` is
serialised behind a per-instance ``threading.Lock`` so two requests
hitting the same cached model cannot stomp on its mutable state.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import os
import queue
import shutil
import tempfile as _tempfile
import threading
import time as _time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

# LRU-bounded model cache. Keyed on ``(name, frozenset(kwargs.items()))`` so
# a ``pretrained=True`` variant does not collide with the bare model.
# Bounded so a long-running worker that sees many distinct kwarg
# combinations does not leak GPU memory; the least-recently-used entry is
# torn down when the cap is hit.
_MODEL_CACHE_CAP = 8
_MODEL_CACHE: OrderedDict[tuple[str, frozenset[tuple[str, Any]]], Any] = OrderedDict()
# Guards _MODEL_CACHE itself (lookup / insert / eviction). Held only across
# pure dict ops; the actual model load happens *outside* the lock so a slow
# weight download does not stall unrelated requests.
_CACHE_LOCK = threading.Lock()
# Per-model serialisation lock. Models cache mutable state (kernel tensors,
# optimizer momentum, RNG cursors); concurrent predict() on the same
# instance would interleave reads and writes. Stored in a sidecar dict
# keyed identically to _MODEL_CACHE.
_MODEL_LOCKS: dict[tuple[str, frozenset[tuple[str, Any]]], threading.Lock] = {}
# In-flight request count per key, and models evicted from the LRU while
# still in use. Calling teardown() on a model another request is
# mid-predict() on frees its weights underneath the running forward pass,
# so an evicted-but-busy model is parked in _PENDING_TEARDOWN until the
# last holder releases it.
_MODEL_REFCOUNTS: dict[tuple[str, frozenset[tuple[str, Any]]], int] = {}
_PENDING_TEARDOWN: OrderedDict[tuple[str, frozenset[tuple[str, Any]]], Any] = OrderedDict()

# Resource admission control (P1.14): concurrent optimize jobs can each
# allocate significant CPU/RAM; without a global bound, different model
# keys running in parallel exhaust the host.  One bounded semaphore
# admits at most MAX_CONCURRENT_OPTIMIZE jobs; excess requests get
# 429 + Retry-After instead of degrading the host.
MAX_CONCURRENT_OPTIMIZE = max(1, int(os.environ.get("OPENLITHOHUB_MAX_CONCURRENT_OPTIMIZE", "2")))
_ADMIT = threading.BoundedSemaphore(MAX_CONCURRENT_OPTIMIZE)

# Optional API key boundary (P1.17): when set, every /v1 request must
# present it in the X-API-Key header.  The server is an on-prem/private
# service; full IAM belongs to a reverse proxy / service mesh.
API_KEY = os.environ.get("OPENLITHOHUB_API_KEY") or ""

# In-memory job store for long-running optimizations (P1.15).  Jobs are
# process-local (restart clears them) and bounded: the oldest terminal
# jobs are evicted beyond _JOB_HISTORY_CAP.
_JOBS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_JOB_LOCK = threading.Lock()
_JOB_HISTORY_CAP = 100
_JOB_QUEUE_DEPTH = int(os.environ.get("OPENLITHOHUB_JOB_QUEUE_DEPTH", "8"))
if _JOB_QUEUE_DEPTH < 1:
    raise RuntimeError("OPENLITHOHUB_JOB_QUEUE_DEPTH must be >= 1")
_JOB_QUEUE: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(maxsize=_JOB_QUEUE_DEPTH)
_JOB_TTL_SECONDS = float(os.environ.get("OPENLITHOHUB_JOB_TTL_SECONDS", "3600"))
_JOB_COUNTER = itertools.count(1)


def _evict_terminal_locked(now: float) -> list[str]:
    """Evict terminal jobs beyond the history cap or TTL. CALLER HOLDS
    _JOB_LOCK — only bookkeeping happens here.  Returns the scratch dirs
    to remove; the CALLER runs ``shutil.rmtree`` AFTER releasing the lock
    (audit P1.3: slow directory deletion must not block job state)."""
    doomed: list[str] = []
    for job_id in list(_JOBS.keys()):
        record = _JOBS[job_id]
        if record["status"] not in ("succeeded", "failed", "cancelled"):
            continue
        age = now - record.get("_created_monotonic", now)
        if len(_JOBS) > _JOB_HISTORY_CAP or age > _JOB_TTL_SECONDS:
            _JOBS.pop(job_id, None)
            doomed.append(str(record.get("scratch_dir") or ""))
    return doomed


def _cleanup_scratch_dirs(doomed: list[str]) -> None:
    for d in doomed:
        if d:
            shutil.rmtree(d, ignore_errors=True)


_JOB_WORKER_STARTED = threading.Event()
_JOB_WORKER_SHUTDOWN = threading.Event()
_JOB_WORKER_THREAD: threading.Thread | None = None


def _admit_blocking_with_shutdown(timeout: float = 1.0) -> bool:
    """Wait (in bounded sleeps) for admission capacity. Queued jobs are
    patient: transient contention with synchronous /v1/optimize must not
    permanently fail a legitimately queued job (audit P1.3)."""
    while True:
        if _JOB_WORKER_SHUTDOWN.is_set():
            return False
        if _ADMIT.acquire(blocking=False):
            return True
        _time.sleep(timeout)


def _start_job_worker(app: Any) -> None:
    """Single fixed worker draining the bounded job queue (audit C3):
    a real queue with locked state transitions and scratch cleanup, not
    one daemon thread per request.  Idempotent (audit P1.1): repeated
    create_app() calls reuse the one process-global worker."""
    if _JOB_WORKER_STARTED.is_set():
        return
    _JOB_WORKER_STARTED.set()

    def _worker() -> None:
        while True:
            job_id, params = _JOB_QUEUE.get()
            with _JOB_LOCK:
                record = _JOBS.get(job_id)
                if record is None or record["status"] in ("cancelled", "uploading"):
                    shutil.rmtree(
                        record.get("scratch_dir", "") if record else "",
                        ignore_errors=True,
                    )
                    _JOB_QUEUE.task_done()
                    continue
                record["status"] = "running"
            admitted = _admit_blocking_with_shutdown()
            if not admitted:
                _JOB_QUEUE.task_done()
                continue
            try:
                # The worker HOLDS the admission slot for the whole run:
                # queued jobs wait patiently for capacity (P1.3) instead of
                # racing the synchronous endpoint for a slot.
                summary = _run_optimize(**params)
                with _JOB_LOCK:
                    record = _JOBS.get(job_id)
                    if record is not None:
                        record["status"] = "succeeded"
                        record["summary"] = summary
                        record["output_path"] = str(summary["output_path"])
                        doomed = _evict_terminal_locked(_time.monotonic())
                _cleanup_scratch_dirs(doomed)
            except Exception as e:  # noqa: BLE001 - surfaced via job status
                with _JOB_LOCK:
                    record = _JOBS.get(job_id)
                    if record is not None:
                        record["status"] = "failed"
                        record["error"] = str(e)
                    # failed jobs have no artifact: their scratch goes now
                    if record is not None:
                        shutil.rmtree(record.get("scratch_dir", ""), ignore_errors=True)
            finally:
                _release_admit()
                _JOB_QUEUE.task_done()

    global _JOB_WORKER_THREAD
    _JOB_WORKER_THREAD = threading.Thread(target=_worker, name="olh-job-worker", daemon=True)
    _JOB_WORKER_THREAD.start()


# Hard cap on multipart upload size for /v1/optimize. Mirrors the 2 GB ceiling
# enforced by ``ModelHub._download_url`` for incoming weights — uniform
# attacker-controlled-bytes contract across the surface. The body is streamed
# to disk in 1 MB chunks below; the cap aborts the stream once cumulative
# bytes-read crosses the threshold so a multi-GB POST cannot fill the worker's
# tmpfs (or, on systems where /tmp is a memory-backed mount, OOM the worker).
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


def _teardown_model(key: tuple[str, frozenset[tuple[str, Any]]], model: Any) -> None:
    """Tear down an evicted model and release its CUDA memory.

    Must be called OUTSIDE ``_CACHE_LOCK`` — teardown can be slow (CUDA
    cache flushes) and must not block unrelated cache operations.
    """
    try:
        model.teardown()
    except Exception:  # noqa: BLE001 — teardown failure shouldn't block eviction
        logger.exception("teardown failed while evicting %r", key)
    finally:
        # teardown() drops Python refs, but CUDA caching allocator holds
        # the freed VRAM until empty_cache(). Without this, an
        # LRU-bounded cache still leaks GPU memory on a long-running
        # worker.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("evicted model %r from cache (capacity=%d)", key, _MODEL_CACHE_CAP)


def _acquire_locked(
    key: tuple[str, frozenset[tuple[str, Any]]], model: Any, lock: threading.Lock
) -> tuple[Any, threading.Lock, tuple[str, frozenset[tuple[str, Any]]]]:
    """Refcount an acquisition. MUST be called while holding _CACHE_LOCK:
    bumping the refcount after releasing the lock opens a window in which
    a concurrent eviction sees refcount == 0 and tears down a model this
    caller is about to use.
    """
    _MODEL_REFCOUNTS[key] = _MODEL_REFCOUNTS.get(key, 0) + 1
    return model, lock, key


def _get_or_load_model(
    name: str, kwargs: dict[str, Any]
) -> tuple[Any, threading.Lock, tuple[str, frozenset[tuple[str, Any]]]]:
    """Return a cached LithographyModel, its predict-serialisation lock, and
    the cache key to pass to :func:`_release_model` when done.

    Two concurrent requests for the same (name, kwargs) pair must not both
    build the model — the second would either double-load weights or race
    to evict the first. We resolve that by holding ``_CACHE_LOCK`` across
    the lookup *and* the insertion of a placeholder lock; the heavy
    ``model.setup()`` happens outside the cache lock under that per-key
    lock so unrelated requests stay unblocked. Acquired models are
    refcounted *under the cache lock* (see :func:`_acquire_locked`);
    callers MUST call ``_release_model(key)`` when finished.
    """
    from openlithohub.models.registry import register_builtin_models, registry

    register_builtin_models()
    key = (name, frozenset(kwargs.items()))

    with _CACHE_LOCK:
        if key in _MODEL_CACHE:
            _MODEL_CACHE.move_to_end(key)
            return _acquire_locked(key, _MODEL_CACHE[key], _MODEL_LOCKS[key])
        # Reserve a per-key lock so a second concurrent caller for the same
        # key blocks on it instead of double-loading.
        per_key_lock = _MODEL_LOCKS.setdefault(key, threading.Lock())

    with per_key_lock:
        # Re-check under the per-key lock — another caller may have loaded
        # while we were waiting.
        with _CACHE_LOCK:
            if key in _MODEL_CACHE:
                _MODEL_CACHE.move_to_end(key)
                return _acquire_locked(key, _MODEL_CACHE[key], per_key_lock)
            # A previous holder may have gotten this key evicted while it
            # was still in use; reuse the parked model instead of
            # reloading weights from disk.
            parked = _PENDING_TEARDOWN.pop(key, None)
            if parked is not None:
                _MODEL_CACHE[key] = parked
                return _acquire_locked(key, parked, per_key_lock)

        model = registry.get(name, ignore_unsupported=False, **kwargs)
        model.setup()

        to_teardown: list[tuple[tuple[str, frozenset[tuple[str, Any]]], Any]] = []
        with _CACHE_LOCK:
            _MODEL_CACHE[key] = model
            while len(_MODEL_CACHE) > _MODEL_CACHE_CAP:
                evicted_key, evicted = _MODEL_CACHE.popitem(last=False)
                if _MODEL_REFCOUNTS.get(evicted_key, 0) > 0:
                    # Still in flight — defer teardown to _release_model.
                    _PENDING_TEARDOWN[evicted_key] = evicted
                    logger.info(
                        "parked in-flight model %r pending teardown (capacity=%d)",
                        evicted_key,
                        _MODEL_CACHE_CAP,
                    )
                    continue
                to_teardown.append((evicted_key, evicted))
            # Refcount OUR acquisition in the same critical section as the
            # insertion: only after this increment can a concurrent insert
            # legitimately evict (and then park, not teardown) this model.
            acquired = _acquire_locked(key, model, per_key_lock)
            # Restore the sidecar lock (an earlier eviction of the same key
            # may have popped it while a waiter was blocked on the old lock).
            _MODEL_LOCKS.setdefault(key, per_key_lock)
        # Slow teardown happens outside the cache lock (P1.2). The sidecar
        # entries are safe to drop: the evicted model has no in-flight
        # holders and is not cached or parked.
        for evicted_key, evicted in to_teardown:
            _MODEL_LOCKS.pop(evicted_key, None)
            _MODEL_REFCOUNTS.pop(evicted_key, None)
            _teardown_model(evicted_key, evicted)
        logger.info("loaded model %r (kwargs=%s) into resident cache", name, kwargs)
        return acquired


def _release_model(key: tuple[str, frozenset[tuple[str, Any]]]) -> None:
    """Drop the caller's reference acquired from ``_get_or_load_model``.

    The last release for an evicted-but-parked model performs the deferred
    teardown.
    """
    with _CACHE_LOCK:
        count = _MODEL_REFCOUNTS.get(key, 0) - 1
        if count > 0:
            _MODEL_REFCOUNTS[key] = count
            return
        _MODEL_REFCOUNTS.pop(key, None)
        model = _PENDING_TEARDOWN.pop(key, None)
        if model is not None:
            # Final teardown of a parked model: the key is neither cached
            # nor parked anymore, so its sidecar entries must go too —
            # otherwise a long-running worker leaks locks/refcounts for
            # every distinct key ever evicted mid-flight (P1.3).
            _MODEL_LOCKS.pop(key, None)
    if model is not None:
        _teardown_model(key, model)


class AdmissionDeniedError(RuntimeError):
    """Raised when the global optimize admission semaphore is exhausted."""


def _run_optimize_admitted(**kwargs: Any) -> dict[str, Any]:
    """Run one optimize under the global admission semaphore (P1.14)."""
    if not _admit_sync():
        raise AdmissionDeniedError()
    try:
        return _run_optimize(**kwargs)
    finally:
        _release_admit()


def _run_optimize(
    *,
    input_path: Path,
    output_path: Path,
    model_name: str,
    node: str,
    pixel_nm: float | None,
    tile_size: int,
    writer: str,
    layer: str | None,
    pretrained: bool,
    min_area_nm2: float = 0.0,
) -> dict[str, Any]:
    """Synchronous optimization core. Mirrors the CLI optimize flow but
    with no Rich I/O — returns a small JSON-friendly summary."""
    from openlithohub.data.io import load_layout
    from openlithohub.workflow.export import export_oasis
    from openlithohub.workflow.halo import compute_halo_px
    from openlithohub.workflow.process_node import get_node
    from openlithohub.workflow.tiling import stitch_tiles, tile_layout

    node_config = get_node(node)
    if pixel_nm is None:
        pixel_nm = node_config.pixel_size_nm
    if writer not in ("mbmw", "vsb"):
        raise ValueError(f"unknown writer {writer!r}; expected 'mbmw' or 'vsb'")

    model_kwargs: dict[str, Any] = {"pretrained": True} if pretrained else {}
    model, model_lock, model_key = _get_or_load_model(model_name, model_kwargs)

    try:
        layout_tensor = load_layout(input_path, pixel_nm, layer=layer)
        halo_px = compute_halo_px(
            node=node_config,
            model=model,
            pixel_nm=pixel_nm,
            tile_size=tile_size,
        )

        tiles = tile_layout(layout_tensor, tile_size=tile_size, overlap=halo_px)
        tile_results = []
        # Hold the per-model lock across all tiles for one request so a
        # concurrent request cannot interleave its predict() calls with ours
        # and corrupt the model's per-tile state (caches, RNG cursors, etc.).
        with model_lock:
            for tile in tiles:
                result = model.predict(tile.tensor)
                tile_results.append((tile, result.mask))

        h, w = layout_tensor.shape
        optimized = stitch_tiles(tile_results, (h, w))
        optimized = (optimized > 0.5).float()

        export_mode = "curvilinear" if writer == "mbmw" else "manhattan"
        try:
            export_oasis(
                optimized,
                output_path,
                mode=export_mode,
                pixel_size_nm=pixel_nm,
                min_area_nm2=min_area_nm2,
            )
            export_format = "oasis"
        except ImportError:
            fallback = output_path.with_suffix(".pt")
            torch.save(optimized, str(fallback))
            output_path = fallback
            export_format = "torch"
    finally:
        # Release the refcount before dropping request-local tensors so a
        # concurrent eviction is free to teardown this model once we are
        # no longer using it.
        _release_model(model_key)

    n_tiles = len(tiles)
    # Drop request-local tensors and flush the CUDA caching allocator so
    # per-request activations from this optimize() don't accumulate as
    # reserved (but unused) VRAM across many requests on the same worker.
    del layout_tensor, tiles, tile_results, optimized
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "shape": [int(h), int(w)],
        "tiles": n_tiles,
        "halo_px": int(halo_px),
        "writer": writer,
        "export_format": export_format,
        "output_path": str(output_path),
    }


def scratch_root() -> Path:
    """Configured scratch root (audit C2): Docker sets OLH_SCRATCH_DIR;
    bare environments fall back to the system temp dir.  Request/job
    temp files and the readiness probe all use THIS path."""
    base = os.environ.get("OLH_SCRATCH_DIR") or _tempfile.gettempdir()
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_scratch_dir(prefix: str) -> Path:
    return Path(_tempfile.mkdtemp(prefix=prefix, dir=str(scratch_root())))


def _admit_sync() -> bool:
    """Non-blocking admission probe for worker threads (P1.14)."""
    return _ADMIT.acquire(blocking=False)


def _release_admit() -> None:
    _ADMIT.release()


_JOB_QUEUE_RESERVED = 0


def _try_reserve_slot() -> bool:
    """Reserve a queue slot BEFORE ingesting a large upload (audit P1.4).

    The reservation counter is checked+incremented with the queue size
    under one lock, so capacity can never be oversubscribed between the
    reservation and the actual enqueue.  Prefer :func:`reserve_job_slot`.
    """
    global _JOB_QUEUE_RESERVED
    with _JOB_LOCK:
        if _JOB_QUEUE.qsize() + _JOB_QUEUE_RESERVED >= _JOB_QUEUE_DEPTH:
            return False
        _JOB_QUEUE_RESERVED += 1
        return True


def _release_slot_reservation() -> None:
    global _JOB_QUEUE_RESERVED
    with _JOB_LOCK:
        _JOB_QUEUE_RESERVED = max(0, _JOB_QUEUE_RESERVED - 1)


@contextlib.contextmanager
def reserve_job_slot() -> Any:
    """Exception-safe slot reservation (audit P1.1): the reservation is
    released on EVERY exit path unless explicitly committed by the
    enqueue (which releases it itself)."""
    if not _try_reserve_slot():
        raise JobQueueFullError()
    committed = False
    try:
        yield
        committed = True
    finally:
        if not committed:
            _release_slot_reservation()


class JobQueueFullError(RuntimeError):
    """Raised when the bounded job queue has no free slot."""


def _put_job(record: dict[str, Any], params: dict[str, Any]) -> tuple[str, bool]:
    """Register a queued job and enqueue it WITHOUT blocking (audit P1.2):
    put_nowait + rollback of the registration on Full. Returns
    (job_id, enqueued)."""
    with _JOB_LOCK:
        job_id = f"job-{next(_JOB_COUNTER)}-{uuid.uuid4().hex[:8]}"
        try:
            _JOB_QUEUE.put_nowait((job_id, params))
        except queue.Full:
            return "", False
        record["job_id"] = job_id
        record["_created_monotonic"] = _time.monotonic()
        _JOBS[job_id] = record
        doomed = _evict_terminal_locked(_time.monotonic())
    _cleanup_scratch_dirs(doomed)
    return job_id, True


def create_app() -> FastAPI:
    """Build the FastAPI app. Factored so tests can spin up a fresh
    instance with TestClient without depending on import-time globals.

    P1.4: the job worker is owned by the app LIFESPAN — started once
    (idempotently across repeated create_app calls) and shut down with a
    bounded drain when the app exits.
    """
    app = FastAPI(
        title="OpenLithoHub Engine",
        description=(
            "HTTP micro-service for computational lithography mask "
            "optimization. The Python engine stays resident; fab-side "
            "C++/Perl pipelines drive it via multipart POST."
        ),
        version="1",
    )

    @app.on_event("startup")
    def _start_worker_lifespan() -> None:
        _JOB_WORKER_SHUTDOWN.clear()
        _start_job_worker(app)

    @app.on_event("shutdown")
    def _stop_worker_lifespan() -> None:
        _JOB_WORKER_SHUTDOWN.set()
        thread = _JOB_WORKER_THREAD
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    @app.middleware("http")
    async def _request_observability(request: Any, call_next: Any) -> Any:
        # P1.18: request id + timing on every response; structured enough
        # for log-based queue/latency analysis without a metrics stack.
        import logging as _logging

        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        if (
            API_KEY
            and request.url.path != "/v1/health"
            and request.headers.get("X-API-Key") != API_KEY
        ):
            from fastapi.responses import JSONResponse as _JSONResponse

            return _JSONResponse(
                status_code=401, content={"detail": "invalid or missing X-API-Key"}
            )
        start = _time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((_time.perf_counter() - start) * 1000, 3)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-OLH-Duration-Ms"] = str(elapsed_ms)
        _logging.getLogger("openlithohub.server.access").info(
            json.dumps(
                {
                    "event": "request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": elapsed_ms,
                }
            )
        )
        return response

    def _require_api_key(request: Any) -> None:
        if API_KEY and request.headers.get("X-API-Key") != API_KEY:
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    _start_job_worker(app)

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/ready")
    def ready() -> dict[str, Any]:
        # P1.16: readiness — can this instance accept work RIGHT NOW?

        from openlithohub.models.registry import register_builtin_models, registry

        registry_ok = True
        try:
            register_builtin_models()
            registry_ok = len(registry.list_models()) > 0
        except Exception:  # noqa: BLE001
            registry_ok = False
        writable = True
        try:
            probe = make_scratch_dir("olh_ready_") / "probe"
            probe.write_text("ok")
            probe.unlink()
            probe.parent.rmdir()
        except OSError:
            writable = False
        ready_now = registry_ok and writable
        body: dict[str, Any] = {
            "ready": ready_now,
            "checks": {
                "models_registered": registry_ok,
                "scratch_writable": writable,
                "admission_slots_free": _ADMIT._value,  # noqa: SLF001 - introspection
            },
            "metrics": {
                "job_queue_depth": _JOB_QUEUE_DEPTH,
                "job_queue_size": _JOB_QUEUE.qsize(),
                "job_queue_reserved": _JOB_QUEUE_RESERVED,
                "jobs_tracked": len(_JOBS),
            },
        }
        if not ready_now:
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail=body)
        return body

    @app.get("/v1/version")
    def version() -> dict[str, Any]:
        from openlithohub._version import __version__
        from openlithohub.benchmark.industrial import git_commit

        commit = git_commit()
        # P2.2: release wheels/containers carry a baked build identity
        # (no .git checkout); a source checkout falls back to git.
        build_info: dict[str, str] = {}
        try:
            # P0.10: explicitly IMPORT the module (the package __init__ does
            # not reference it) — release wheels/containers bake _build.py
            # (see publish.yml); source checkouts do not have one.
            from importlib import import_module

            build_mod = import_module("openlithohub._build")
            build_info = {
                "build_commit": getattr(build_mod, "BUILD_COMMIT", ""),
                "build_timestamp": getattr(build_mod, "BUILD_TIMESTAMP", ""),
                "build_run_id": getattr(build_mod, "BUILD_RUN_ID", ""),
                "build_version": getattr(build_mod, "BUILD_VERSION", ""),
            }
        except ModuleNotFoundError:
            build_info = {}
        except Exception:  # noqa: BLE001 - build metadata is best-effort
            build_info = {}
        build_commit = build_info.get("build_commit") or ""
        return {
            "api": "v1",
            "package": "openlithohub",
            "version": __version__,
            "git_commit": commit or build_commit or "unknown",
            "build": build_info,
            "torch": torch.__version__,
        }

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        from openlithohub.models.registry import register_builtin_models, registry
        from openlithohub.simulators.registry import list_simulators

        register_builtin_models()
        gpu: dict[str, Any] = {"available": bool(torch.cuda.is_available())}
        if torch.cuda.is_available():
            gpu["device_count"] = torch.cuda.device_count()
            gpu["device_name"] = torch.cuda.get_device_name(0)
        return {
            "api_schema_version": "1",
            "capability_schema_version": "1",
            "models": sorted(registry.list_models()),
            "simulator_backends": sorted(list_simulators()),
            "gpu": gpu,
            "streaming_pipeline": True,
            "input_formats": ["npy", "pt", "oas", "gds"],
            "export_formats": ["oasis", "gds", "pt"],
            "proof_verification": {
                # P4.3: report what THIS installed artifact can actually do,
                # separate from repository governance provenance.
                "runtime_available": (
                    Path(__file__).resolve().parents[2] / "proof_artifacts/p054/replay-receipt.json"
                ).exists(),
                "level": "p054-governed",
            },
            "build_provenance": {
                "p054_governance": "see proof_artifacts/p054/README.md in the source repository",
            },
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, list[str]]:
        from openlithohub.models.registry import register_builtin_models, registry

        register_builtin_models()
        return {"models": sorted(registry.list_models())}

    @app.post("/v1/optimize", response_model=None)
    async def optimize(
        layout: UploadFile = File(..., description="Layout file (.oas, .gds, .pt, .npy)"),
        model: str = Form(..., description="Registered model name."),
        node: str = Form("3nm-euv", description="Process node."),
        pixel_nm: float | None = Form(
            None,
            description=(
                "Pixel size in nanometers. If unset, falls back to the node's "
                "native pitch (1.0 nm/px is treated as a real value, not a sentinel)."
            ),
        ),
        tile_size: int = Form(2048, description="Tile size in pixels."),
        writer: str = Form("mbmw", description="Target writer: mbmw or vsb."),
        layer: str | None = Form(
            None, description="OASIS/GDSII layer 'LAYER:DTYPE'; required for multi-layer files."
        ),
        pretrained: bool = Form(False, description="Load pretrained weights when supported."),
        min_area_nm2: float = Form(
            0.0,
            description=(
                "Drop curvilinear shapes below this polygon area (nm^2) at export. "
                "Default 0.0 keeps every shape (Hackathon-safe); set >0 for "
                "fab-oriented export quality. This export does NOT establish foundry "
                "sign-off or complete manufacturing-rule compliance."
            ),
        ),
    ) -> Response | JSONResponse:
        if not layout.filename:
            raise HTTPException(status_code=400, detail="layout upload missing filename")

        suffix = Path(layout.filename).suffix or ".bin"
        # Per-request scratch dir WITHOUT a context manager: the optimized
        # output must outlive this handler so the response can stream it
        # from disk; a BackgroundTask removes the dir after the response
        # completes. On any failure path we clean up before re-raising.
        tmp_path = make_scratch_dir("olh_req_")
        try:
            input_path = tmp_path / f"input{suffix}"
            output_path = tmp_path / "optimized.oas"

            # Stream the upload to disk in chunks rather than slurping the
            # whole body into memory; abort once we cross the cap so a
            # malicious multi-GB POST cannot OOM the worker before the
            # request handler ever runs.
            bytes_read = 0
            chunk_size = 1024 * 1024
            with input_path.open("wb") as out_f:
                while True:
                    chunk = await layout.read(chunk_size)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    if bytes_read > _MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(f"layout upload exceeds {_MAX_UPLOAD_BYTES} bytes"),
                        )
                    out_f.write(chunk)

            try:
                # CPU/GPU-bound work — run in a thread so the event loop
                # stays responsive to other requests and to the health
                # endpoint (issue #36). FastAPI's default "async def"
                # endpoint runs the body on the event-loop thread, which
                # would otherwise stall every other in-flight request for
                # seconds-to-minutes per optimization.
                summary = await asyncio.to_thread(
                    _run_optimize_admitted,
                    input_path=input_path,
                    output_path=output_path,
                    model_name=model,
                    node=node,
                    pixel_nm=pixel_nm,
                    tile_size=tile_size,
                    writer=writer,
                    layer=layer,
                    pretrained=pretrained,
                    min_area_nm2=min_area_nm2,
                )
            except AdmissionDeniedError:
                raise HTTPException(
                    status_code=429,
                    detail="server at maximum concurrent optimizations",
                    headers={"Retry-After": "5"},
                ) from None
            except KeyError as e:
                # Both unknown model names and unknown node names raise KeyError;
                # disambiguate by message so the client gets the right status.
                msg = str(e)
                if "process node" in msg.lower():
                    raise HTTPException(status_code=400, detail=msg.strip("'\"")) from None
                raise HTTPException(status_code=404, detail=f"unknown model: {e}") from None
            except (FileNotFoundError, ValueError) as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except ImportError as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"missing optional dependency: {e}",
                ) from None

            served_path = Path(summary["output_path"])
            if not served_path.exists():
                raise HTTPException(status_code=500, detail="optimization produced no output file")

            # Stream the file from disk instead of read_bytes(): a multi-GB
            # output no longer transits through RAM a second time.
            return FileResponse(
                served_path,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f'attachment; filename="{served_path.name}"',
                    "X-OLH-Tiles": str(summary["tiles"]),
                    "X-OLH-Halo-Px": str(summary["halo_px"]),
                    "X-OLH-Export-Format": summary["export_format"],
                    "X-OLH-Shape": "x".join(str(d) for d in summary["shape"]),
                },
                background=BackgroundTask(shutil.rmtree, tmp_path, True),
            )
        except BaseException:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise

    # ---- Long-task job API (P1.15): minute-scale OPC/ILT should not hold
    # an HTTP connection open. Same core as /v1/optimize, executed in a
    # worker thread under the same admission semaphore; jobs live in the
    # process (restart clears them) and are bounded by _JOB_HISTORY_CAP.

    @app.post("/v1/jobs/optimize", response_model=None)
    async def create_optimize_job(
        layout: UploadFile = File(..., description="Layout file (.oas, .gds, .pt, .npy)"),
        model: str = Form(..., description="Registered model name."),
        node: str = Form("3nm-euv", description="Process node."),
        pixel_nm: float | None = Form(None, description="Pixel size in nanometers."),
        tile_size: int = Form(2048, description="Tile size in pixels."),
        writer: str = Form("mbmw", description="Target writer: mbmw or vsb."),
        layer: str | None = Form(None, description="OASIS/GDSII layer 'LAYER:DTYPE'."),
        pretrained: bool = Form(False, description="Load pretrained weights."),
        min_area_nm2: float = Form(0.0, description="Drop shapes below this area (nm^2)."),
    ) -> JSONResponse:
        if not layout.filename:
            raise HTTPException(status_code=400, detail="layout upload missing filename")
        # P1.4/P1.1: reserve the queue slot BEFORE copying the (possibly
        # large) upload, via an exception-safe context manager — every
        # failure path releases the reservation automatically.
        try:
            reserve_cm = reserve_job_slot()
            reserve_cm.__enter__()
        except JobQueueFullError:
            raise HTTPException(
                status_code=429,
                detail="job queue is full",
                headers={"Retry-After": "10"},
            ) from None
        suffix = Path(layout.filename).suffix or ".bin"
        tmp_path = make_scratch_dir("olh_job_")
        input_path = tmp_path / f"input{suffix}"
        output_path = tmp_path / "optimized.oas"
        bytes_read = 0
        with input_path.open("wb") as out_f:
            while True:
                chunk = await layout.read(1024 * 1024)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > _MAX_UPLOAD_BYTES:
                    reserve_cm.__exit__(None, None, None)
                    shutil.rmtree(tmp_path, ignore_errors=True)
                    raise HTTPException(status_code=413, detail="layout upload too large")
                out_f.write(chunk)
        params: dict[str, Any] = dict(
            input_path=input_path,
            output_path=output_path,
            model_name=model,
            node=node,
            pixel_nm=pixel_nm,
            tile_size=tile_size,
            writer=writer,
            layer=layer,
            pretrained=pretrained,
            min_area_nm2=min_area_nm2,
        )
        job_id, queued = _put_job(
            {
                "status": "queued",
                "created_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                "scratch_dir": str(tmp_path),
                "summary": None,
                "error": None,
                "output_path": None,
            },
            params,
        )
        _release_slot_reservation()  # the enqueued item now owns the slot
        if not queued:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise HTTPException(
                status_code=429,
                detail="job queue is full",
                headers={"Retry-After": "10"},
            )
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "queued", "poll": f"/v1/jobs/{job_id}"},
        )

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        # P1.5: reads also drive time-based eviction of terminal jobs.
        doomed: list[str] = []
        with _JOB_LOCK:
            doomed = _evict_terminal_locked(_time.monotonic())
            job = _JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
            # P1.6: snapshot copy under the lock — worker threads mutate
            # the live record concurrently.
            snapshot = {
                "api_schema_version": "1",
                "job_id": job_id,
                "status": job["status"],
                "created_utc": job["created_utc"],
                "summary": job["summary"],
                "error": job["error"],
            }
        _cleanup_scratch_dirs(doomed)
        return snapshot

    @app.get("/v1/jobs/{job_id}/artifact", response_model=None)
    def get_job_artifact(job_id: str) -> FileResponse:
        # P1.2: snapshot status/output path UNDER the lock, same contract
        # as get_job; a successful GET also refreshes the job TTL so the
        # janitor cannot delete the artifact mid-download.
        doomed: list[str] = []
        with _JOB_LOCK:
            doomed = _evict_terminal_locked(_time.monotonic())
            job = _JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
            if job["status"] != "succeeded" or not job.get("output_path"):
                raise HTTPException(status_code=409, detail=f"job {job_id} has no artifact yet")
            output_path = str(job["output_path"])
            job["_last_access_monotonic"] = _time.monotonic()
        _cleanup_scratch_dirs(doomed)
        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{Path(output_path).name}"',
            },
        )

    @app.delete("/v1/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, str]:
        with _JOB_LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
            if job["status"] == "running":
                raise HTTPException(status_code=409, detail="running job cannot be deleted")
            if job["status"] == "queued":
                # The worker drops cancelled jobs (and their scratch) when
                # dequeued; mark cancelled so it is never executed.
                job["status"] = "cancelled"
            _JOBS.pop(job_id, None)
            scratch = job.get("scratch_dir")
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        return {"deleted": job_id}

    return app
