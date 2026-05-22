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
    min_area_nm2: float = 0.0,
) -> None:
    """Export an optimized mask tensor to OASIS format.

    For manhattan mode, extracts rectilinear contours and writes via KLayout.
    For curvilinear mode, fits B-splines and writes a curvilinear OASIS file
    (sampled polygons on a designated layer; see ``contour.curvilinear``).
    Native SEMI P39 (OASIS.MASK) curve primitives and SEMI P44 multi-beam
    mask-writer input are tracked separately and not yet emitted here.

    ``min_area_nm2`` (curvilinear only) drops sub-resolution islands below
    the given polygon area before writing. Default ``0.0`` keeps every
    shape so academic / Hackathon evaluation stays bit-exact.
    """
    if mode not in ("manhattan", "curvilinear"):
        raise ValueError(f"mode must be 'manhattan' or 'curvilinear', got '{mode}'")

    m = ensure_2d(mask)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "manhattan":
        _export_manhattan(m, output_path, pixel_size_nm)
    else:
        _export_curvilinear(m, output_path, pixel_size_nm, min_area_nm2=min_area_nm2)


def export_gds(
    mask: torch.Tensor,
    output_path: str | Path,
    *,
    mode: str = "curvilinear",
    pixel_size_nm: float = 1.0,
    samples_per_curve: int = 64,
    min_area_nm2: float = 0.0,
) -> None:
    """Export an optimized mask tensor to GDSII format.

    GDSII is the academic / contest lingua franca (ICCAD, SPIE benchmarks
    and the cuLitho whitepaper all use ``.gds``); OASIS is dominant in
    mask-shop flows. This function covers the academic path so users do
    not have to convert ``.oas`` → ``.gds`` themselves before running our
    benchmark on contest-style inputs.

    Same routing as :func:`export_oasis`: ``mode="manhattan"`` extracts
    rectilinear polygons; ``mode="curvilinear"`` fits B-splines, samples
    them to polygons (GDSII has no native curve primitive) and writes
    a polygon-only ``.gds`` via KLayout. The polygon density is controlled
    by ``samples_per_curve`` and matches the OASIS curvilinear writer's
    default — a curvilinear OASIS and a curvilinear GDS exported from the
    same mask are visually identical, but the GDS file is larger because
    every curve becomes an explicit vertex list.
    """
    if mode not in ("manhattan", "curvilinear"):
        raise ValueError(f"mode must be 'manhattan' or 'curvilinear', got '{mode}'")

    m = ensure_2d(mask)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "manhattan":
        _export_manhattan(m, output_path, pixel_size_nm)
    else:
        _export_curvilinear(
            m,
            output_path,
            pixel_size_nm,
            samples_per_curve=samples_per_curve,
            min_area_nm2=min_area_nm2,
        )


def _export_manhattan(mask: torch.Tensor, output_path: Path, pixel_size_nm: float) -> None:
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

    # KLayout DB units: layout.dbu is in microns. A DB integer coord i represents
    # i * dbu microns = i * dbu * 1000 nm. Coordinates here are already in nm,
    # so divide by (dbu * 1000) — equivalently by pixel_size_nm.
    nm_per_dbu = layout.dbu * 1000.0
    for poly_vertices in polygons:
        if len(poly_vertices) < 3:
            continue
        points = [
            db.Point(int(round(x / nm_per_dbu)), int(round(y / nm_per_dbu)))
            for x, y in poly_vertices
        ]
        top.shapes(layer_idx).insert(db.Polygon(points))

    layout.write(str(output_path))


def _export_curvilinear(
    mask: torch.Tensor,
    output_path: Path,
    pixel_size_nm: float,
    *,
    samples_per_curve: int = 64,
    min_area_nm2: float = 0.0,
) -> None:
    from openlithohub.workflow.contour.curvilinear import export_oasis_mbw, fit_bspline

    curves = fit_bspline(mask, tolerance_nm=pixel_size_nm * 0.5, pixel_size_nm=pixel_size_nm)
    if not curves:
        raise ValueError(
            "No curvilinear contours could be extracted from the mask — refusing "
            "to write an empty file. Check that the mask contains foreground "
            "pixels and that the tolerance is appropriate for the pixel pitch."
        )
    # klayout auto-detects OASIS vs GDSII from the output extension; the
    # writer name is historical — ``.oas`` and ``.gds`` are both supported.
    export_oasis_mbw(
        curves,
        str(output_path),
        pixel_size_nm=pixel_size_nm,
        samples_per_curve=samples_per_curve,
        min_area_nm2=min_area_nm2,
    )
