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
import logging
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

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

# Hard cap on multipart upload size for /v1/optimize. Mirrors the 2 GB ceiling
# enforced by ``ModelHub._download_url`` for incoming weights — uniform
# attacker-controlled-bytes contract across the surface. The body is streamed
# to disk in 1 MB chunks below; the cap aborts the stream once cumulative
# bytes-read crosses the threshold so a multi-GB POST cannot fill the worker's
# tmpfs (or, on systems where /tmp is a memory-backed mount, OOM the worker).
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


def _teardown_model(key: tuple[str, frozenset[tuple[str, Any]]], model: Any) -> None:
    """Tear down an evicted model and release its CUDA memory."""
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


def _acquire(
    key: tuple[str, frozenset[tuple[str, Any]]], model: Any, lock: threading.Lock
) -> tuple[Any, threading.Lock, tuple[str, frozenset[tuple[str, Any]]]]:
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
    refcounted; callers MUST call ``_release_model(key)`` when finished.
    """
    from openlithohub.models.registry import register_builtin_models, registry

    register_builtin_models()
    key = (name, frozenset(kwargs.items()))

    with _CACHE_LOCK:
        if key in _MODEL_CACHE:
            _MODEL_CACHE.move_to_end(key)
            return _acquire(key, _MODEL_CACHE[key], _MODEL_LOCKS[key])
        # Reserve a per-key lock so a second concurrent caller for the same
        # key blocks on it instead of double-loading.
        per_key_lock = _MODEL_LOCKS.setdefault(key, threading.Lock())

    with per_key_lock:
        # Re-check under the per-key lock — another caller may have loaded
        # while we were waiting.
        with _CACHE_LOCK:
            if key in _MODEL_CACHE:
                _MODEL_CACHE.move_to_end(key)
                return _acquire(key, _MODEL_CACHE[key], per_key_lock)
            # A previous holder may have gotten this key evicted while it
            # was still in use; reuse the parked model instead of
            # reloading weights from disk.
            parked = _PENDING_TEARDOWN.pop(key, None)
            if parked is not None:
                _MODEL_CACHE[key] = parked
                return _acquire(key, parked, per_key_lock)

        model = registry.get(name, **kwargs)
        model.setup()

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
                _teardown_model(evicted_key, evicted)
        logger.info("loaded model %r (kwargs=%s) into resident cache", name, kwargs)
        return _acquire(key, model, per_key_lock)


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


def create_app() -> FastAPI:
    """Build the FastAPI app. Factored so tests can spin up a fresh
    instance with TestClient without depending on import-time globals."""
    app = FastAPI(
        title="OpenLithoHub Engine",
        description=(
            "HTTP micro-service for computational lithography mask "
            "optimization. The Python engine stays resident; fab-side "
            "C++/Perl pipelines drive it via multipart POST."
        ),
        version="1",
    )

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

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
                "fab-ready MRC-compliant output."
            ),
        ),
    ) -> Response | JSONResponse:
        if not layout.filename:
            raise HTTPException(status_code=400, detail="layout upload missing filename")

        suffix = Path(layout.filename).suffix or ".bin"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
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
                    _run_optimize,
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

            payload = served_path.read_bytes()

        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{served_path.name}"',
                "X-OLH-Tiles": str(summary["tiles"]),
                "X-OLH-Halo-Px": str(summary["halo_px"]),
                "X-OLH-Export-Format": summary["export_format"],
                "X-OLH-Shape": "x".join(str(d) for d in summary["shape"]),
            },
        )

    return app
