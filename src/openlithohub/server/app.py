"""FastAPI app exposing the OpenLithoHub optimization engine over HTTP.

Endpoints:
  - GET  /v1/health   — liveness probe.
  - GET  /v1/ready    — readiness probe (registry + scratch + runtime).
  - GET  /v1/models   — list registered model names.
  - POST /v1/optimize — multipart upload of a layout file + model name,
    returns the optimized layout binary.
  - POST /v1/jobs/optimize + GET /v1/jobs/{id}[/artifact], DELETE —
    long-task job API for minute-scale optimizations.

Lifecycle authority (repair-plan P0.1): app construction starts NO
threads and owns NO mutable lifecycle state. The FastAPI lifespan is the
single owner — it creates one :class:`~openlithohub.server.runtime.ServerRuntime`
per app instance on entry (``app.state.runtime``), starts its single job
worker, and terminates it on exit. Constructing an app with
``create_app()`` is side-effect free; entering the lifespan is what makes
the service able to accept work.

Models are loaded lazily on first request and cached in-process; repeat
requests against the same model skip weight loading entirely. The cache
is keyed by ``(name, frozenset(kwargs.items()))`` so a `pretrained=True`
variant does not collide with the bare model.

Concurrency
-----------
The optimize endpoint is ``async def`` but the underlying optimization
is pure CPU/GPU work, so we dispatch it to a worker thread via
``asyncio.to_thread`` to keep the event loop responsive (issue #36).
Admission to that work is bounded by the runtime's semaphore
(``ServerConfig.max_concurrent_optimize``); excess requests get
429 + Retry-After instead of degrading the host. Per-model ``predict()``
is serialised behind a per-instance ``threading.Lock`` so two requests
hitting the same cached model cannot stomp on its mutable state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import tempfile as _tempfile
import threading
import time as _time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask  # type: ignore[import-not-found]
from starlette.exceptions import HTTPException as StarletteHTTPException

from openlithohub.server.config import ServerConfig
from openlithohub.server.errors import ApiHTTPException, error_envelope, runtime_error_code
from openlithohub.server.job_store import JobRecord, serialize_params
from openlithohub.server.observability import emit_event, registry
from openlithohub.server.runtime import (
    AdmissionDeniedError,
    JobArtifactUnavailableError,
    JobQueueFullError,
    JobStillRunningError,
    RuntimeNotAcceptingWorkError,
    ServerRuntime,
    UnknownJobError,
)
from openlithohub.server.schemas import (
    API_SCHEMA_VERSION,
    CapabilitiesResponse,
    ErrorCode,
    HealthResponse,
    JobCreateResponse,
    JobDeleteResponse,
    JobStatus,
    JobStatusResponse,
    MetricsResponse,
    ReadyResponse,
    VersionResponse,
)

logger = logging.getLogger(__name__)

# LRU-bounded model cache. Keyed on ``(name, frozenset(kwargs.items()))`` so
# a ``pretrained=True`` variant does not collide with the bare model.
# Bounded so a long-running worker that sees many distinct kwarg
# combinations does not leak GPU memory; the least-recently-used entry is
# torn down when the cap is hit.
#
# This cache is request-scoped state shared across app instances in a
# process, not lifecycle-owned state; it stays module-level until the
# model-cache ownership refactor (repair-plan P1.1 follow-up).
_MODEL_CACHE_CAP = 8
_MODEL_CACHE: OrderedDict[tuple[str, frozenset[tuple[str, Any]], str], Any] = OrderedDict()
# Guards _MODEL_CACHE itself (lookup / insert / eviction). Held only across
# pure dict ops; the actual model load happens *outside* the lock so a slow
# weight download does not stall unrelated requests.
_CACHE_LOCK = threading.Lock()
# Per-model serialisation lock. Models cache mutable state (kernel tensors,
# optimizer momentum, RNG cursors); concurrent predict() on the same
# instance would interleave reads and writes. Stored in a sidecar dict
# keyed identically to _MODEL_CACHE.
_MODEL_LOCKS: dict[tuple[str, frozenset[tuple[str, Any]], str], threading.Lock] = {}
# In-flight request count per key, and models evicted from the LRU while
# still in use. Calling teardown() on a model another request is
# mid-predict() on frees its weights underneath the running forward pass,
# so an evicted-but-busy model is parked in _PENDING_TEARDOWN until the
# last holder releases it.
_MODEL_REFCOUNTS: dict[tuple[str, frozenset[tuple[str, Any]], str], int] = {}
_PENDING_TEARDOWN: OrderedDict[tuple[str, frozenset[tuple[str, Any]], str], Any] = OrderedDict()


def _teardown_model(key: tuple[str, frozenset[tuple[str, Any]], str], model: Any) -> None:
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
    key: tuple[str, frozenset[tuple[str, Any]], str], model: Any, lock: threading.Lock
) -> tuple[Any, threading.Lock, tuple[str, frozenset[tuple[str, Any]], str]]:
    """Refcount an acquisition. MUST be called while holding _CACHE_LOCK:
    bumping the refcount after releasing the lock opens a window in which
    a concurrent eviction sees refcount == 0 and tears down a model this
    caller is about to use.
    """
    _MODEL_REFCOUNTS[key] = _MODEL_REFCOUNTS.get(key, 0) + 1
    return model, lock, key


def _model_cache_key(
    name: str, kwargs: dict[str, Any], device: str
) -> tuple[str, frozenset[tuple[str, Any]], str]:
    return (name, frozenset(kwargs.items()), device)


def _get_or_load_model(
    name: str, kwargs: dict[str, Any], device: str = "cpu"
) -> tuple[Any, threading.Lock, tuple[str, frozenset[tuple[str, Any]], str]]:
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
    key = _model_cache_key(name, kwargs, device)

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

        to_teardown: list[tuple[tuple[str, frozenset[tuple[str, Any]], str], Any]] = []
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


def _release_model(key: tuple[str, frozenset[tuple[str, Any]], str]) -> None:
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
    execution_mode: str = "auto",
    request_id: str | None = None,
    job_id: str | None = None,
    device_policy: str = "auto",
    gpu_batch_tiles: int = 1,
) -> dict[str, Any]:
    """Synchronous optimization core. Both the sync endpoint and the job
    worker execute through this one function, which dispatches into the
    shared product execution spine (``workflow.execution.optimize_layout``)
    — dense under the memory policy, streaming for supported large jobs,
    fail-closed otherwise. Returns a small JSON-friendly summary."""
    import time as _clock

    from openlithohub.server import observability as _obs
    from openlithohub.server.config import resolve_execution_device
    from openlithohub.workflow.execution import (
        OptimizeRequest,
        coerce_execution_mode,
        optimize_layout,
    )
    from openlithohub.workflow.process_node import get_node

    mode = coerce_execution_mode(execution_mode)
    node_config = get_node(node)
    if pixel_nm is None:
        pixel_nm = node_config.pixel_size_nm

    # PR-G §8: one server-owned execution device; explicit CUDA fails
    # closed when unavailable — never a silent CPU fallback.
    selected_device = resolve_execution_device(
        device_policy,
        cuda_available=torch.cuda.is_available(),
        device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
    )

    model_kwargs: dict[str, Any] = {"pretrained": True} if pretrained else {}
    model, model_lock, model_key = _get_or_load_model(model_name, model_kwargs, selected_device)

    _obs.emit_event(
        "optimize_started",
        request_id=request_id,
        job_id=job_id,
        model_name=model_name,
        selected_device=selected_device,
        execution_mode=mode,
    )
    _obs.registry.optimize_started()
    started = _clock.perf_counter()
    try:
        try:
            supported = getattr(model, "supported_devices", ("cpu",))
            if selected_device not in supported:
                raise ValueError(
                    f"model {model_name!r} does not declare support for device "
                    f"{selected_device!r} (supports: {', '.join(supported)}); the "
                    "server device policy must match model capability"
                )
            forward_kwargs: dict[str, Any] = {"device": selected_device}
            request = OptimizeRequest(
                model=model,
                pixel_size_nm=pixel_nm,
                input_path=input_path,
                output_path=output_path,
                output_kind="oasis",
                writer=writer,
                layer=layer,
                node=node_config,
                tile_size=tile_size,
                threshold=0.5,
                min_area_nm2=min_area_nm2,
                execution_mode=mode,
                gpu_batch_tiles=max(1, int(gpu_batch_tiles)),
                forward_kwargs=forward_kwargs,
            )
            # Hold the per-model lock across the whole run (dense tile loop
            # or streaming tile schedule) so a concurrent request cannot
            # interleave its predict() calls with ours and corrupt the
            # model's per-tile state. PR-A ownership preserved: the runtime
            # owns admission, the lock owns predict serialization, the
            # streaming executor owns tile scheduling.
            with model_lock:
                result = optimize_layout(request)
            summary = result_to_summary(result, writer=writer, device=selected_device)
        except Exception as exc:
            _obs.registry.optimize_completed(
                execution_mode=mode,
                execution_reason="UNPLANNED_FAILURE",
                input_backend="unknown",
                output_backend="unknown",
                tiles=0,
                outcome="failed",
            )
            _obs.emit_event(
                "optimize_failed",
                request_id=request_id,
                job_id=job_id,
                model_name=model_name,
                execution_mode=mode,
                error_type=type(exc).__name__,
                duration_ms=round((_clock.perf_counter() - started) * 1000, 3),
            )
            raise
        _obs.registry.optimize_completed(
            execution_mode=result.plan.mode,
            execution_reason=result.plan.reason,
            input_backend=result.plan.input_backend,
            output_backend=result.plan.output_backend,
            tiles=result.n_tiles,
            forward_pixels=int(
                (result.work_accounting or {}).get("forward_simulator_input_pixels", 0)
            ),
            screened_pixels=int((result.work_accounting or {}).get("screened_out_pixels", 0)),
        )
        _obs.emit_event(
            "optimize_completed",
            request_id=request_id,
            job_id=job_id,
            model_name=model_name,
            execution_mode=result.plan.mode,
            execution_reason=result.plan.reason,
            input_backend=result.plan.input_backend,
            output_backend=result.plan.output_backend,
            tile_count=result.n_tiles,
            halo_px=result.halo_px,
            duration_ms=round((_clock.perf_counter() - started) * 1000, 3),
        )
        return summary
    finally:
        # Release the refcount before dropping request-local tensors so a
        # concurrent eviction is free to teardown this model once we are no
        # longer using it. Flush the CUDA caching allocator so per-request
        # activations don't accumulate as reserved-but-unused VRAM.
        _release_model(model_key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def result_to_summary(result: Any, *, writer: str, device: str = "cpu") -> dict[str, Any]:
    """JSON-friendly execution summary shared by the sync endpoint, the
    job worker and (indirectly) the CLI. Private fields (``output_path``)
    ride along internally but are dropped by the public projection in
    :func:`openlithohub.server.schemas.project_optimize_metadata`."""
    plan = result.plan
    accounting = result.work_accounting or {}
    summary: dict[str, Any] = {
        "shape": [int(result.shape[0]), int(result.shape[1])],
        "tiles": int(result.n_tiles),
        "halo_px": int(result.halo_px),
        "writer": writer,
        "export_format": result.output_format,
        "output_path": str(result.output_path),
        "execution_mode": plan.mode,
        "execution_reason": plan.reason,
        "input_backend": plan.input_backend,
        "output_backend": plan.output_backend,
        "threshold": float(result.threshold),
        "estimated_dense_bytes": int(plan.estimated_dense_bytes),
        "max_dense_bytes": int(plan.max_dense_bytes),
        "selected_device": device,
        "batch_size": int(result.batch_size),
        "forward_batches": int(result.forward_batches),
    }
    if result.work_accounting is not None:
        # Stable, bounded counter projection (PR-D §7) — the full internal
        # ledger is never dumped into the contract.
        summary["work"] = {
            "forward_pixels": int(accounting.get("forward_simulator_input_pixels", 0)),
            "read_pixels": int(accounting.get("read_window_pixels", 0)),
            "screened_pixels": int(accounting.get("screened_out_pixels", 0)),
        }
    return summary


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


def _require_runtime(request: Request) -> ServerRuntime:
    """Fetch the lifespan-owned runtime or 503. Requests that need the
    runtime (anything admitting work or touching jobs) cannot be served
    before the lifespan starts it — construction deliberately does not."""
    runtime: ServerRuntime | None = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise ApiHTTPException(
            503,
            ErrorCode.SERVER_NOT_ACCEPTING,
            "server runtime not started (app lifespan has not been entered)",
        )
    return runtime


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build the FastAPI app. Factored so tests can spin up a fresh
    instance with TestClient without depending on import-time globals.

    P0.1 (server lifecycle authority): this factory is side-effect free —
    no worker thread, no runtime. The lifespan below owns exactly one
    :class:`ServerRuntime`; entering it starts the worker, exiting it
    drains and stops the runtime.
    """
    config = config if config is not None else ServerConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        # Ownership exclusion (PR-A red-team blocker A): a forced stop
        # leaves the previous runtime's worker (or an admitted run) alive
        # in this process. Starting a second runtime then would create a
        # second execution authority — e.g. two concurrent optimizes
        # under max_concurrent_optimize=1. Fail closed until the previous
        # runtime truly owns nothing.
        previous = getattr(app.state, "runtime", None)
        if previous is not None and (previous.worker_alive or previous.executing):
            raise RuntimeError(
                "refusing to start a second execution authority: the previous "
                "runtime still owns a live worker/in-flight run after a forced "
                "stop; wait for it to exit before restarting"
            )
        from openlithohub.server.observability import emit_event as _emit

        runtime = ServerRuntime(config, optimize_runner=_run_optimize)
        app.state.runtime = runtime
        runtime.start()
        _emit("runtime_started", job_backend=config.job_backend)
        try:
            yield
        finally:
            _emit("runtime_draining", stop_forced=False)
            runtime.stop()
            _emit("runtime_stopped", stop_forced=runtime.stop_forced)
            if runtime.worker_alive or runtime.executing:
                # Forced stop: RETAIN the reference so a future lifespan
                # cannot overlap the detached execution (exclusion above).
                app.state.runtime = runtime
            else:
                # Clean stop: detach so post-shutdown requests observe a
                # clean 503 instead of a stopped runtime.
                app.state.runtime = None

    app = FastAPI(
        title="OpenLithoHub Engine",
        description=(
            "HTTP micro-service for computational lithography mask "
            "optimization. The Python engine stays resident; fab-side "
            "C++/Perl pipelines drive it via multipart POST."
        ),
        version="1",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _request_observability(request: Any, call_next: Any) -> Any:
        # PR-D §4/§11: one request_id correlates the response header, the
        # error envelope and the structured access log. Client-supplied IDs
        # are preserved only when bounded-safe; anything else is replaced by
        # a generated ID (log-injection / cardinality firewall).
        import logging as _logging

        from fastapi.responses import JSONResponse as _JSONResponse

        from openlithohub.server.errors import error_envelope
        from openlithohub.server.observability import (
            emit_event,
            registry,
            sanitize_request_id,
        )

        request_id = sanitize_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        if (
            config.api_key
            and request.url.path != "/v1/health"
            and request.headers.get("X-API-Key") != config.api_key
        ):
            registry.request_completed(401, 0.0)
            return _JSONResponse(
                status_code=401,
                content=error_envelope(
                    code="INVALID_REQUEST",
                    message="invalid or missing X-API-Key",
                    request_id=request_id,
                ),
                headers={"X-Request-ID": request_id},
            )
        start = _time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Unhandled exception: log the traceback SERVER-side, answer the
            # client with the fixed INTERNAL_ERROR envelope — never a stack
            # trace or internal path. The exception-handler below covers
            # HTTPException (which resolves inside the router); this covers
            # everything that escapes it.
            logger.exception("unhandled error on %s %s", request.method, request.url.path)
            elapsed_ms = round((_time.perf_counter() - start) * 1000, 3)
            registry.request_completed(500, elapsed_ms)
            emit_event(
                "http_request_completed",
                request_id=request_id,
                method=request.method,
                status=500,
                status_class="5xx",
                duration_ms=elapsed_ms,
            )
            return _JSONResponse(
                status_code=500,
                content=error_envelope(
                    code="INTERNAL_ERROR",
                    message="internal server error",
                    request_id=request_id,
                ),
                headers={"X-Request-ID": request_id},
            )
        elapsed_ms = round((_time.perf_counter() - start) * 1000, 3)
        registry.request_completed(response.status_code, elapsed_ms)
        emit_event(
            "http_request_completed",
            request_id=request_id,
            method=request.method,
            status=response.status_code,
            status_class=f"{response.status_code // 100}xx",
            duration_ms=elapsed_ms,
        )
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

    @app.exception_handler(RequestValidationError)
    async def _validation_error_envelope(request: Request, exc: Exception) -> JSONResponse:
        """Malformed user input is 400 INVALID_REQUEST (PR-D §3), not the
        FastAPI default 422; body shape matches the standard envelope."""
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=400,
            content=error_envelope(
                code=ErrorCode.INVALID_REQUEST,
                message="invalid request payload",
                detail=json.loads(exc.json()) if hasattr(exc, "json") else str(exc),
                request_id=request_id,
            ),
            headers={"X-Request-ID": request_id} if request_id else {},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_envelope(request: Request, exc: Exception) -> JSONResponse:
        """Render any HTTPException as the PR-D error envelope. The legacy
        top-level ``detail`` is preserved verbatim for compatibility;
        clients key on ``error.code``."""
        request_id = getattr(request.state, "request_id", None)
        if isinstance(exc, ApiHTTPException):
            code: ErrorCode = exc.code
            message = exc.error_message
        else:
            status = getattr(exc, "status_code", 500)
            message = str(getattr(exc, "detail", "") or "")
            if status in (400, 405, 422):
                code = ErrorCode.INVALID_REQUEST
            elif status == 404:
                code = (
                    ErrorCode.UNKNOWN_JOB
                    if "/v1/jobs/" in request.url.path
                    else ErrorCode.INVALID_REQUEST
                )
            else:
                code = runtime_error_code(exc)[1]
        exc_detail = getattr(exc, "detail", message)
        headers = dict(getattr(exc, "headers", None) or {})
        if request_id:
            headers.setdefault("X-Request-ID", request_id)
        return JSONResponse(
            status_code=getattr(exc, "status_code", 500),
            content=error_envelope(
                code=code,
                message=message,
                detail=exc_detail,
                request_id=request_id,
            ),
            headers=headers,
        )

    @app.get("/v1/health", response_model=HealthResponse, status_code=200)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/ready", response_model=ReadyResponse, status_code=200)
    def ready(request: Request) -> dict[str, Any]:
        # P1.16: readiness — can this instance accept work RIGHT NOW?
        # Runtime counters come from the public snapshot; no private
        # semaphore internals (repair-plan P1.4).
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
        runtime = getattr(request.app.state, "runtime", None)
        snap = runtime.snapshot() if runtime is not None else None
        accepting = bool(snap and snap["accepting_requests"])
        ready_now = registry_ok and writable and accepting
        body: dict[str, Any] = {
            "ready": ready_now,
            "checks": {
                "models_registered": registry_ok,
                "scratch_writable": writable,
                "accepting_requests": accepting,
            },
            "metrics": (
                {
                    "job_queue_depth": snap["job_queue_capacity"],
                    "job_queue_size": snap["job_queue_size"],
                    "job_queue_reserved": snap["job_queue_reserved"],
                    "jobs_tracked": snap["jobs_tracked"],
                }
                if snap
                else {}
            ),
            "runtime": snap,
        }
        if not ready_now:
            raise ApiHTTPException(503, ErrorCode.SERVER_NOT_ACCEPTING, "server not ready")
        return body

    @app.get("/v1/version", response_model=VersionResponse, status_code=200)
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
            # Repair-plan §10 (PR-B): only the DETERMINISTIC fields the
            # build module actually carries are reported. BUILD_TIMESTAMP /
            # BUILD_RUN_ID were read here but never written by
            # scripts/write_build_info.py — nondeterministic provenance
            # belongs in OCI labels / attestations, not this response.
            build_info = {
                "build_commit": getattr(build_mod, "BUILD_COMMIT", ""),
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

    @app.get("/v1/capabilities", response_model=CapabilitiesResponse, status_code=200)
    def capabilities() -> dict[str, Any]:
        from openlithohub.models.registry import register_builtin_models, registry
        from openlithohub.simulators.registry import list_simulators
        from openlithohub.workflow.execution import streaming_capability_matrix

        register_builtin_models()
        # PR-G §9: CUDA availability is NOT product GPU execution. The
        # resolved device and whether the policy actually produced a usable
        # device are reported separately and truthfully.
        from openlithohub.server.config import resolve_execution_device

        cuda_available = bool(torch.cuda.is_available())
        device_count = torch.cuda.device_count() if cuda_available else 0
        try:
            selected_device = resolve_execution_device(
                config.device, cuda_available=cuda_available, device_count=device_count
            )
            product_execution_enabled = True
        except ValueError:
            selected_device = "unavailable"
            product_execution_enabled = False
        gpu: dict[str, Any] = {
            "available": cuda_available,
            "device_count": device_count,
            "selected_device": selected_device,
            "product_execution_enabled": product_execution_enabled,
        }
        if cuda_available and device_count > 0:
            gpu["device_name"] = torch.cuda.get_device_name(0)
        return {
            "api_schema_version": API_SCHEMA_VERSION,
            "capability_schema_version": "1",
            "models": sorted(registry.list_models()),
            "simulator_backends": sorted(list_simulators()),
            "gpu": gpu,
            # Legacy coarse flag, kept for v1 clients; superseded by the
            # structured "streaming" matrix below (PR-C §11).
            "streaming_pipeline": True,
            "streaming": streaming_capability_matrix(),
            "input_formats": ["npy", "pt", "oas", "gds"],
            "export_formats": ["oasis", "gds", "pt"],
            # Truthful job contract (PR-E §17): in-memory is non-durable
            # and loses jobs on restart; sqlite persists committed jobs and
            # artifacts under OPENLITHOHUB_STATE_DIR. BOTH stay
            # single-process — durability does not license more workers.
            "jobs": {
                "backend": config.job_backend,
                "durable": config.job_backend == "sqlite",
                "restart_loses_jobs": config.job_backend != "sqlite",
                "single_process_only": True,
            },
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
        request: Request,
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
        execution_mode: str = Form(
            "auto",
            description=(
                "Execution topology: 'auto' picks dense under the dense memory "
                "policy (OPENLITHOHUB_MAX_DENSE_BYTES) and streaming for supported "
                "large jobs; unsupported large jobs fail closed (400) instead of "
                "silently materializing a full-chip raster."
            ),
        ),
    ) -> Response | JSONResponse:
        runtime = _require_runtime(request)
        if not runtime.accepting_jobs:
            raise ApiHTTPException(
                503, ErrorCode.SERVER_NOT_ACCEPTING, "server is draining; not accepting work"
            )
        if not layout.filename:
            raise ApiHTTPException(400, ErrorCode.INVALID_REQUEST, "layout upload missing filename")

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
                    if bytes_read > config.max_upload_bytes:
                        raise ApiHTTPException(
                            413,
                            ErrorCode.UPLOAD_TOO_LARGE,
                            f"layout upload exceeds {config.max_upload_bytes} bytes",
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
                    runtime.run_admitted,
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
                    execution_mode=execution_mode,
                    request_id=getattr(request.state, "request_id", None),
                )
            except AdmissionDeniedError as exc:
                raise ApiHTTPException(
                    429,
                    ErrorCode.ADMISSION_FULL,
                    "server at maximum concurrent optimizations",
                    headers={"Retry-After": "5"},
                ) from exc
            except RuntimeNotAcceptingWorkError as exc:
                # Lifecycle authority rejection (draining/stopped) — 503,
                # never 400/429 (PR-A red-team blocker B).
                raise ApiHTTPException(
                    503,
                    ErrorCode.SERVER_NOT_ACCEPTING,
                    "server is draining; not accepting new work",
                ) from exc
            except Exception as exc:
                # PR-D §3: one central mapping owns exception -> (status,
                # code). Messages stay user-facing; tracebacks stay local.
                status, code = runtime_error_code(exc)
                if code is ErrorCode.INTERNAL_ERROR:
                    logger.exception("optimize failed")
                    raise ApiHTTPException(
                        500, ErrorCode.INTERNAL_ERROR, "internal server error"
                    ) from exc
                message = str(exc)
                if isinstance(exc, KeyError):
                    message = message.strip("'\"")
                raise ApiHTTPException(status, code, message) from exc

            served_path = Path(summary["output_path"])
            if not served_path.exists():
                raise ApiHTTPException(
                    500, ErrorCode.INTERNAL_ERROR, "optimization produced no output file"
                )

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
                    "X-OLH-Execution-Mode": summary["execution_mode"],
                    "X-OLH-Execution-Reason": summary["execution_reason"],
                    "X-OLH-Input-Backend": summary["input_backend"],
                    "X-OLH-Output-Backend": summary["output_backend"],
                },
                background=BackgroundTask(shutil.rmtree, tmp_path, True),
            )
        except BaseException:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise

    # ---- Long-task job API (P1.15): minute-scale OPC/ILT should not hold
    # an HTTP connection open. Same core as /v1/optimize, executed in the
    # runtime's single worker thread under the same admission semaphore;
    # jobs live in the runtime (process-local, restart clears them) and
    # are bounded by ServerConfig.job_history_cap.

    @app.post(
        "/v1/jobs/optimize",
        response_model=JobCreateResponse,
        status_code=202,
    )
    async def create_optimize_job(
        request: Request,
        layout: UploadFile = File(..., description="Layout file (.oas, .gds, .pt, .npy)"),
        model: str = Form(..., description="Registered model name."),
        node: str = Form("3nm-euv", description="Process node."),
        pixel_nm: float | None = Form(None, description="Pixel size in nanometers."),
        tile_size: int = Form(2048, description="Tile size in pixels."),
        writer: str = Form("mbmw", description="Target writer: mbmw or vsb."),
        layer: str | None = Form(None, description="OASIS/GDSII layer 'LAYER:DTYPE'."),
        pretrained: bool = Form(False, description="Load pretrained weights."),
        min_area_nm2: float = Form(0.0, description="Drop shapes below this area (nm^2)."),
        execution_mode: str = Form(
            "auto",
            description=(
                "Execution topology: 'auto' (planner), 'dense' or 'streaming'. "
                "Unsupported large jobs fail closed instead of falling back to dense."
            ),
        ),
    ) -> JSONResponse:
        runtime = _require_runtime(request)
        if not runtime.accepting_jobs:
            raise ApiHTTPException(
                503, ErrorCode.SERVER_NOT_ACCEPTING, "server is not accepting new jobs"
            )
        if not layout.filename:
            raise ApiHTTPException(400, ErrorCode.INVALID_REQUEST, "layout upload missing filename")

        # P0.2: ONE transactional reservation protocol. The slot is
        # reserved BEFORE the upload body is ingested; the reservation is
        # consumed exactly once by commit() and released on EVERY other
        # exit path by the context manager.
        #
        # PR-E commit order (durable mode): validated upload -> atomic
        # input publication -> QUEUED store commit -> reservation consumed
        # -> HTTP 202. There is no state where the client saw a 202 and no
        # durable job exists (PR-E §8/E5).
        tmp_path: Path | None = None
        durable = config.job_backend == "sqlite"
        staging: Path | None = None
        job_dir: Path | None = None
        try:
            with runtime.reserve_job_slot() as reservation:
                tmp_path = make_scratch_dir("olh_job_")
                suffix = Path(layout.filename).suffix or ".bin"
                job_id = runtime.new_job_id()
                if durable:
                    staging = runtime.staging_path(job_id, suffix)
                    upload_path = staging
                else:
                    upload_path = tmp_path / f"input{suffix}"
                bytes_read = 0
                with upload_path.open("wb") as out_f:
                    while True:
                        chunk = await layout.read(1024 * 1024)
                        if not chunk:
                            break
                        bytes_read += len(chunk)
                        if bytes_read > config.max_upload_bytes:
                            raise ApiHTTPException(
                                413, ErrorCode.UPLOAD_TOO_LARGE, "layout upload too large"
                            )
                        out_f.write(chunk)

                if durable and staging is not None:
                    input_path = Path(runtime.finalize_durable_input(job_id, staging, suffix))
                    job_dir = runtime.durable_job_dir(job_id)
                    output_path = job_dir / "artifact.oas"
                else:
                    input_path = upload_path
                    output_path = tmp_path / "optimized.oas"

                params: dict[str, Any] = dict(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    model_name=model,
                    node=node,
                    pixel_nm=pixel_nm,
                    tile_size=tile_size,
                    writer=writer,
                    layer=layer,
                    pretrained=pretrained,
                    min_area_nm2=min_area_nm2,
                    execution_mode=execution_mode,
                )
                record = JobRecord(
                    job_id=job_id,
                    status=JobStatus.QUEUED,
                    created_utc=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                    input_path=str(input_path),
                    input_format=suffix.lstrip("."),
                    params_json=serialize_params(params),
                )
                job_id = reservation.commit(record)
                emit_event(
                    "job_created",
                    job_id=job_id,
                    request_id=getattr(request.state, "request_id", None),
                )
                registry.job_created()
        except JobQueueFullError as exc:
            raise ApiHTTPException(
                429,
                ErrorCode.QUEUE_FULL,
                "job queue is full",
                headers={"Retry-After": "10"},
            ) from exc
        except RuntimeNotAcceptingWorkError as exc:
            # Commit-time lifecycle gate: shutdown won the race against
            # this reservation's upload/commit — 503, no job enqueued
            # (PR-A red-team blocker B). The context manager already
            # released the reservation.
            raise ApiHTTPException(
                503,
                ErrorCode.SERVER_NOT_ACCEPTING,
                "server is not accepting new jobs",
            ) from exc
        except BaseException:
            # The scratch dir is request-local garbage on every failure
            # path (on success the durable job owns its state instead).
            # A crashed upload must leave no ghost job, no consumed
            # capacity and no un-tracked staging bytes (PR-E E5).
            if staging is not None:
                with contextlib.suppress(OSError):
                    staging.unlink()
            if job_dir is not None:
                shutil.rmtree(job_dir, ignore_errors=True)
            if tmp_path is not None:
                shutil.rmtree(tmp_path, ignore_errors=True)
            raise
        return JSONResponse(
            status_code=202,
            content={
                "api_schema_version": API_SCHEMA_VERSION,
                "job_id": job_id,
                "status": JobStatus.QUEUED.value,
                "poll": f"/v1/jobs/{job_id}",
            },
        )

    @app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, status_code=200)
    def get_job(request: Request, job_id: str) -> dict[str, Any]:
        runtime = _require_runtime(request)
        try:
            return runtime.get_job_snapshot(job_id)
        except UnknownJobError as exc:
            raise ApiHTTPException(404, ErrorCode.UNKNOWN_JOB, f"unknown job: {job_id}") from exc

    @app.get("/v1/jobs/{job_id}/artifact", response_model=None)
    def get_job_artifact(request: Request, job_id: str) -> FileResponse:
        runtime = _require_runtime(request)
        try:
            output_path = runtime.get_job_artifact_path(job_id)
        except UnknownJobError as exc:
            raise ApiHTTPException(404, ErrorCode.UNKNOWN_JOB, f"unknown job: {job_id}") from exc
        except JobArtifactUnavailableError as exc:
            raise ApiHTTPException(
                409, ErrorCode.JOB_ARTIFACT_UNAVAILABLE, f"job {job_id} has no artifact yet"
            ) from exc
        # Download lease (PR-E §14): the artifact cannot be evicted while
        # this response streams; the background task releases after the
        # last byte is sent.
        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{Path(output_path).name}"',
            },
            background=BackgroundTask(runtime.release_artifact_lease, job_id),
        )

    @app.delete("/v1/jobs/{job_id}", response_model=JobDeleteResponse, status_code=200)
    def delete_job(request: Request, job_id: str) -> dict[str, str]:
        runtime = _require_runtime(request)
        try:
            scratch = runtime.delete_job(job_id)
        except UnknownJobError as exc:
            raise ApiHTTPException(404, ErrorCode.UNKNOWN_JOB, f"unknown job: {job_id}") from exc
        except JobStillRunningError as exc:
            raise ApiHTTPException(
                409, ErrorCode.JOB_RUNNING, "running job cannot be deleted"
            ) from exc
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        return {"deleted": job_id}

    # ---- process-local metrics (PR-D §12/§14) ----------------------------
    # Rendered from the observability registry plus the runtime's public
    # snapshot — never private semaphore/queue introspection. Documented as
    # process-local, non-durable, single-worker authority.

    @app.get("/v1/metrics", response_model=MetricsResponse, status_code=200)
    def metrics(request: Request) -> dict[str, Any]:
        runtime = _require_runtime(request)
        body = registry.render()
        body["api_schema_version"] = API_SCHEMA_VERSION
        body["scope"] = "process"
        body["runtime"] = runtime.snapshot()
        return body

    return app
