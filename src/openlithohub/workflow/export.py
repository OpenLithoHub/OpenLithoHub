"""OASIS/GDSII export coordination."""

from __future__ import annotations

from pathlib import Path

import torch

from openlithohub._utils.tensor_ops import ensure_2d


def export_oasis(
    mask: torch.Tensor,
    output_path: str | Path,
    *,
    mode: str = "curvilinear",
    pixel_size_nm: float = 1.0,
) -> None:
    """Export an optimized mask tensor to OASIS format.

    For manhattan mode, extracts rectilinear contours and writes via KLayout.
    For curvilinear mode, fits B-splines and writes OASIS.MBW format.
    """
    if mode not in ("manhattan", "curvilinear"):
        raise ValueError(f"mode must be 'manhattan' or 'curvilinear', got '{mode}'")

    m = ensure_2d(mask)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "manhattan":
        _export_manhattan(m, output_path, pixel_size_nm)
    else:
        _export_curvilinear(m, output_path, pixel_size_nm)


def _export_manhattan(
    mask: torch.Tensor, output_path: Path, pixel_size_nm: float
) -> None:
    from openlithohub.workflow.contour.manhattan import extract_manhattan_contour

    polygons = extract_manhattan_contour(mask, pixel_size_nm=pixel_size_nm)

    try:
        import klayout.db as db
    except ImportError:
        raise ImportError(
            "klayout is required for Manhattan OASIS export. "
            "Install with: pip install openlithohub[workflow]"
        ) from None

    layout = db.Layout()
    layout.dbu = pixel_size_nm / 1000.0
    top = layout.create_cell("TOP")
    layer_idx = layout.layer(1, 0)

    for poly_vertices in polygons:
        if len(poly_vertices) < 3:
            continue
        points = [
            db.Point(int(x / layout.dbu), int(y / layout.dbu))
            for x, y in poly_vertices
        ]
        top.shapes(layer_idx).insert(db.Polygon(points))

    layout.write(str(output_path))


def _export_curvilinear(
    mask: torch.Tensor, output_path: Path, pixel_size_nm: float
) -> None:
    from openlithohub.workflow.contour.curvilinear import export_oasis_mbw, fit_bspline

    curves = fit_bspline(mask, tolerance_nm=pixel_size_nm * 0.5, pixel_size_nm=pixel_size_nm)
    if not curves:
        output_path.write_bytes(b"")
        return
    export_oasis_mbw(curves, str(output_path))
