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
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, frozenset[tuple[str, Any]]], Any] = {}


def _get_or_load_model(name: str, kwargs: dict[str, Any]) -> Any:
    """Return a cached LithographyModel or instantiate + setup a new one."""
    from openlithohub.models.registry import register_builtin_models, registry

    register_builtin_models()
    key = (name, frozenset(kwargs.items()))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model = registry.get(name, **kwargs)
    model.setup()
    _MODEL_CACHE[key] = model
    logger.info("loaded model %r (kwargs=%s) into resident cache", name, kwargs)
    return model


def _run_optimize(
    *,
    input_path: Path,
    output_path: Path,
    model_name: str,
    node: str,
    pixel_nm: float,
    tile_size: int,
    writer: str,
    layer: str | None,
    pretrained: bool,
) -> dict[str, Any]:
    """Synchronous optimization core. Mirrors the CLI optimize flow but
    with no Rich I/O — returns a small JSON-friendly summary."""
    from openlithohub.data.io import load_layout
    from openlithohub.workflow.export import export_oasis
    from openlithohub.workflow.halo import compute_halo_px
    from openlithohub.workflow.process_node import get_node
    from openlithohub.workflow.tiling import stitch_tiles, tile_layout

    node_config = get_node(node)
    if pixel_nm == 1.0:
        pixel_nm = node_config.pixel_size_nm

    model_kwargs: dict[str, Any] = {"pretrained": True} if pretrained else {}
    model = _get_or_load_model(model_name, model_kwargs)

    layout_tensor = load_layout(input_path, pixel_nm, layer=layer)
    halo_px = compute_halo_px(
        node=node_config,
        model=model,
        pixel_nm=pixel_nm,
        tile_size=tile_size,
    )

    tiles = tile_layout(layout_tensor, tile_size=tile_size, overlap=halo_px)
    tile_results = []
    for tile in tiles:
        result = model.predict(tile.tensor)
        tile_results.append((tile, result.mask))

    h, w = layout_tensor.shape
    optimized = stitch_tiles(tile_results, (h, w))
    optimized = (optimized > 0.5).float()

    export_mode = "curvilinear" if writer == "mbmw" else "manhattan"
    try:
        export_oasis(optimized, output_path, mode=export_mode, pixel_size_nm=pixel_nm)
        export_format = "oasis"
    except ImportError:
        fallback = output_path.with_suffix(".pt")
        torch.save(optimized, str(fallback))
        output_path = fallback
        export_format = "torch"

    return {
        "shape": [int(h), int(w)],
        "tiles": len(tiles),
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
        pixel_nm: float = Form(1.0, description="Pixel size in nanometers."),
        tile_size: int = Form(2048, description="Tile size in pixels."),
        writer: str = Form("mbmw", description="Target writer: mbmw or vsb."),
        layer: str | None = Form(
            None, description="OASIS/GDSII layer 'LAYER:DTYPE'; required for multi-layer files."
        ),
        pretrained: bool = Form(False, description="Load pretrained weights when supported."),
    ) -> Response | JSONResponse:
        if not layout.filename:
            raise HTTPException(status_code=400, detail="layout upload missing filename")

        suffix = Path(layout.filename).suffix or ".bin"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / f"input{suffix}"
            output_path = tmp_path / "optimized.oas"

            content = await layout.read()
            input_path.write_bytes(content)

            try:
                summary = _run_optimize(
                    input_path=input_path,
                    output_path=output_path,
                    model_name=model,
                    node=node,
                    pixel_nm=pixel_nm,
                    tile_size=tile_size,
                    writer=writer,
                    layer=layer,
                    pretrained=pretrained,
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
