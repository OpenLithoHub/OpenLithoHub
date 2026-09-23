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
import json
import logging
import os
import shutil
import tempfile as _tempfile
import threading
import time as _time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask  # type: ignore[import-not-found]

from openlithohub.server.config import ServerConfig
from openlithohub.server.runtime import (
    AdmissionDeniedError,
    JobArtifactUnavailableError,
    JobQueueFullError,
    JobStillRunningError,
    RuntimeNotAcceptingWorkError,
    ServerRuntime,
    UnknownJobError,
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


def _require_runtime(request: Request) -> ServerRuntime:
    """Fetch the lifespan-owned runtime or 503. Requests that need the
    runtime (anything admitting work or touching jobs) cannot be served
    before the lifespan starts it — construction deliberately does not."""
    runtime: ServerRuntime | None = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail="server runtime not started (app lifespan has not been entered)",
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
        runtime = ServerRuntime(config, optimize_runner=_run_optimize)
        app.state.runtime = runtime
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()
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
        # P1.18: request id + timing on every response; structured enough
        # for log-based queue/latency analysis without a metrics stack.
        import logging as _logging

        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        if (
            config.api_key
            and request.url.path != "/v1/health"
            and request.headers.get("X-API-Key") != config.api_key
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

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/ready")
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
            # Truthful single-process job contract (repair-plan §4 Phase 1):
            # in-memory store, no persistence, single worker process only.
            "jobs": {
                "backend": config.job_backend,
                "durable": False,
                "restart_loses_jobs": True,
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
    ) -> Response | JSONResponse:
        runtime = _require_runtime(request)
        if not runtime.accepting_jobs:
            raise HTTPException(status_code=503, detail="server is draining; not accepting work")
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
                    if bytes_read > config.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(f"layout upload exceeds {config.max_upload_bytes} bytes"),
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
                )
            except AdmissionDeniedError:
                raise HTTPException(
                    status_code=429,
                    detail="server at maximum concurrent optimizations",
                    headers={"Retry-After": "5"},
                ) from None
            except RuntimeNotAcceptingWorkError:
                # Lifecycle authority rejection (draining/stopped) — 503,
                # never 400/429 (PR-A red-team blocker B).
                raise HTTPException(
                    status_code=503,
                    detail="server is draining; not accepting new work",
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
    # an HTTP connection open. Same core as /v1/optimize, executed in the
    # runtime's single worker thread under the same admission semaphore;
    # jobs live in the runtime (process-local, restart clears them) and
    # are bounded by ServerConfig.job_history_cap.

    @app.post("/v1/jobs/optimize", response_model=None)
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
    ) -> JSONResponse:
        runtime = _require_runtime(request)
        if not runtime.accepting_jobs:
            raise HTTPException(status_code=503, detail="server is not accepting new jobs")
        if not layout.filename:
            raise HTTPException(status_code=400, detail="layout upload missing filename")

        # P0.2: ONE transactional reservation protocol. The slot is
        # reserved BEFORE the upload body is ingested; the reservation is
        # consumed exactly once by commit() and released on EVERY other
        # exit path by the context manager.
        tmp_path: Path | None = None
        try:
            with runtime.reserve_job_slot() as reservation:
                tmp_path = make_scratch_dir("olh_job_")
                suffix = Path(layout.filename).suffix or ".bin"
                input_path = tmp_path / f"input{suffix}"
                output_path = tmp_path / "optimized.oas"
                bytes_read = 0
                with input_path.open("wb") as out_f:
                    while True:
                        chunk = await layout.read(1024 * 1024)
                        if not chunk:
                            break
                        bytes_read += len(chunk)
                        if bytes_read > config.max_upload_bytes:
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
                job_id = reservation.commit(
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
        except JobQueueFullError:
            raise HTTPException(
                status_code=429,
                detail="job queue is full",
                headers={"Retry-After": "10"},
            ) from None
        except RuntimeNotAcceptingWorkError:
            # Commit-time lifecycle gate: shutdown won the race against
            # this reservation's upload/commit — 503, no job enqueued
            # (PR-A red-team blocker B). The context manager already
            # released the reservation.
            raise HTTPException(
                status_code=503,
                detail="server is not accepting new jobs",
            ) from None
        except BaseException:
            # The scratch dir is request-local garbage on every failure
            # path (on success it belongs to the committed job record).
            if tmp_path is not None:
                shutil.rmtree(tmp_path, ignore_errors=True)
            raise
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "queued",
                "poll": f"/v1/jobs/{job_id}",
            },
        )

    @app.get("/v1/jobs/{job_id}")
    def get_job(request: Request, job_id: str) -> dict[str, Any]:
        runtime = _require_runtime(request)
        try:
            return runtime.get_job_snapshot(job_id)
        except UnknownJobError:
            raise HTTPException(status_code=404, detail=f"unknown job: {job_id}") from None

    @app.get("/v1/jobs/{job_id}/artifact", response_model=None)
    def get_job_artifact(request: Request, job_id: str) -> FileResponse:
        runtime = _require_runtime(request)
        try:
            output_path = runtime.get_job_artifact_path(job_id)
        except UnknownJobError:
            raise HTTPException(status_code=404, detail=f"unknown job: {job_id}") from None
        except JobArtifactUnavailableError:
            raise HTTPException(
                status_code=409, detail=f"job {job_id} has no artifact yet"
            ) from None
        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{Path(output_path).name}"',
            },
        )

    @app.delete("/v1/jobs/{job_id}")
    def delete_job(request: Request, job_id: str) -> dict[str, str]:
        runtime = _require_runtime(request)
        try:
            scratch = runtime.delete_job(job_id)
        except UnknownJobError:
            raise HTTPException(status_code=404, detail=f"unknown job: {job_id}") from None
        except JobStillRunningError:
            raise HTTPException(status_code=409, detail="running job cannot be deleted") from None
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        return {"deleted": job_id}

    return app
