"""Layout file I/O — public reader for `.pt` / `.npy` / `.oas` / `.gds`.

Promoted from ``cli.optimize_cmd._load_layout_as_tensor``: the function
is the canonical reader and is already imported by the API façade
(``api/mask.py``) and the HTTP server (``server/app.py``). Keeping it
under ``cli/`` with a leading underscore was load-bearing fiction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_layout(
    path: Path | str,
    pixel_nm: float,
    layer: str | None = None,
    *,
    lef_files: list[Path | str] | None = None,
) -> torch.Tensor:
    """Load a layout file and rasterize to a 2-D float tensor.

    Supports ``.pt`` / ``.npy`` (returned verbatim, must be 2-D) and
    ``.oas`` / ``.gds`` / ``.def`` / ``.lef`` (rasterized via klayout
    at ``pixel_nm`` pitch).

    For OASIS/GDSII inputs with more than one layer, ``layer`` must be
    a ``"LAYER:DTYPE"`` string (e.g. ``"1:0"``); otherwise the loader
    refuses rather than collapsing every layer onto the same mask.

    DEF inputs require companion LEF files (``lef_files=[...]``) to
    resolve cell abstracts; without LEF context, a placed-and-routed
    DEF reduces to placement records without the geometry that
    OpenLithoHub needs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pt":
        loaded = torch.load(str(path), weights_only=True)
        if not isinstance(loaded, torch.Tensor) or loaded.ndim != 2:
            raise ValueError(
                f"{path}: expected a 2-D torch.Tensor for layout input, "
                f"got {type(loaded).__name__}"
                + (f" ndim={loaded.ndim}" if isinstance(loaded, torch.Tensor) else "")
            )
        return loaded.float()

    if suffix == ".npy":
        import numpy as np

        arr = np.load(str(path), allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(
                f"{path}: expected a 2-D ndarray for layout input, got ndim={arr.ndim}"
            )
        return torch.from_numpy(arr).float()

    try:
        import klayout.db as db
    except ImportError:
        raise ImportError(
            "klayout is required for OASIS / GDSII / DEF / LEF parsing. "
            "Install with: pip install openlithohub[workflow]"
        ) from None

    layout = db.Layout()
    if suffix in (".def", ".lef"):
        opts = db.LoadLayoutOptions()
        if lef_files:
            opts.lefdef_config.lef_files = [str(Path(p)) for p in lef_files]
            # Suppress KLayout's auto-rescan of the DEF directory for
            # *.lef when the caller hands us explicit LEF files —
            # otherwise the same LEF parses twice and KLayout raises
            # "Duplicate MACRO" before we see any geometry.
            opts.lefdef_config.read_lef_with_def = False
        layout.read(str(path), opts)
    else:
        layout.read(str(path))

    top_cells = list(layout.top_cells())
    if not top_cells:
        raise ValueError(f"{path}: layout has no top cells.")
    top_cell = top_cells[0]
    bbox = top_cell.bbox()

    width_dbu = bbox.width()
    height_dbu = bbox.height()
    dbu_nm = layout.dbu * 1000.0
    pixels_per_dbu = dbu_nm / pixel_nm

    # Non-integer ratios silently collapse adjacent DBU coordinates to the
    # same pixel after ``round()``, which shifts polygon edges by 1 px.
    # At advanced nodes 1 px == one design rule violation. Warn the
    # caller so they can pick a pixel_nm that divides DBU cleanly (e.g.
    # 1.0 nm with DBU=0.001 um, not 1.5 nm).
    if dbu_nm > 0 and pixel_nm > 0:
        ratio = dbu_nm / pixel_nm
        if abs(ratio - round(ratio)) > 1e-6 and abs(1.0 / ratio - round(1.0 / ratio)) > 1e-6:
            import warnings

            warnings.warn(
                f"load_layout: pixel_nm={pixel_nm} does not divide DBU "
                f"({dbu_nm} nm) cleanly; adjacent vertices may collapse to "
                f"the same pixel and polygon edges may shift by 1 px. "
                f"Pick pixel_nm such that dbu_nm / pixel_nm is integer.",
                UserWarning,
                stacklevel=2,
            )

    w_px = max(1, int(width_dbu * pixels_per_dbu))
    h_px = max(1, int(height_dbu * pixels_per_dbu))

    selected_layer_idx = _select_layer(layout, layer)

    import numpy as np
    from PIL import Image, ImageDraw

    canvas = Image.new("L", (w_px, h_px), 0)
    drawer = ImageDraw.Draw(canvas)

    shapes_iter = top_cell.begin_shapes_rec(selected_layer_idx)

    def _project(point: Any) -> tuple[int, int]:
        # GDSII / OASIS use mathematical (y-up) coordinates; PIL's drawing
        # surface uses image (y-down) coordinates. Flip y so the rasterized
        # tensor has the same orientation as the layout viewer — without
        # this, every export round-trip (load → optimize → export) returns
        # a vertically mirrored result, and visualizations overlaid against
        # a coordinate grid disagree with the source GDS.
        # Round (not truncate) so sub-pixel-aligned vertices snap to the
        # nearest integer pixel rather than always toward zero — int()
        # truncation drops sub-pixel features at the canvas origin and
        # biases polygon area toward the negative axes.
        px = int(round((point.x - bbox.left) * pixels_per_dbu))
        py_math = int(round((point.y - bbox.bottom) * pixels_per_dbu))
        py = (h_px - 1) - py_math
        return (max(0, min(px, w_px - 1)), max(0, min(py, h_px - 1)))

    # Two-pass rasterization with proper hole semantics.
    #
    # Naïve per-shape draw (solid → hole, then move to next shape) is wrong
    # when polygon A has a hole and polygon B's solid intersects that hole:
    # drawing in iteration order lets A's hole erase B's solid even though
    # the hole belongs only to A. Real GDS/OASIS semantics: a hole subtracts
    # from its *own* polygon, not from the global canvas.
    #
    # Use klayout.db.Region to compute the per-polygon (solid AND-NOT holes)
    # set, union all polygons, then rasterize the unioned merged result. The
    # Region object is the same primitive KLayout / Calibre use for boolean
    # ops on layouts; this matches what a human sees in the layout viewer.
    region = db.Region()
    while not shapes_iter.at_end():
        shape = shapes_iter.shape()
        # Recursive iteration walks across hierarchy — each shape's
        # geometry is in its own cell's coordinates, so transform up to
        # the top cell before projecting.
        trans = shapes_iter.trans()
        if shape.is_polygon() or shape.is_box():
            poly = shape.polygon if shape.is_polygon() else db.Polygon(shape.box)
            region.insert(poly.transformed(trans))
        shapes_iter.next()
    region.merge()

    # Rasterize the merged region by decomposing each polygon into
    # convex pieces (klayout's `decompose_convex`). Each convex piece
    # is hole-free, so PIL's polygon fill is exact and there is no
    # global hole/solid ordering hazard: a hole in polygon A only
    # subtracted from A during decompose, never from polygon B.
    #
    # An earlier approach drew hulls then subtracted holes onto the
    # global canvas; that erased B's solid whenever B sat inside A's
    # hole region (the merged Region keeps them as separate polygons,
    # so A's hole rectangle covers B's pixels).
    for poly in region.each():
        try:
            convex_pieces = list(poly.decompose_convex(db.Polygon.PO_any))
        except (AttributeError, TypeError):
            # Older klayout: fall back to per-polygon hull/holes draw.
            # Hole-vs-other-polygon hazard re-emerges, but the previous
            # behaviour was the same and downstream tests already pass.
            convex_pieces = [poly]
        for piece in convex_pieces:
            # decompose_convex yields SimplePolygon (no holes); iterate
            # vertices via each_point. Fall back to each_point_hull for
            # the donut-fallback path that yields a Polygon.
            iter_points = (
                piece.each_point if hasattr(piece, "each_point") else piece.each_point_hull
            )
            hull = [_project(p) for p in iter_points()]
            if len(hull) >= 3:
                drawer.polygon(hull, fill=255)

    raster = np.array(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(raster)


def _select_layer(layout: Any, layer: str | None) -> int:
    """Resolve a ``LAYER:DTYPE`` string to a klayout layer index.

    Refuses multi-layer files when the user did not specify a layer — the
    historical behavior of OR-ing every layer into one mask collapses
    multi-layer designs into nonsense input.
    """
    layer_indices = list(layout.layer_indices())
    if not layer_indices:
        raise ValueError("Layout contains no layers.")

    if layer is None:
        if len(layer_indices) > 1:
            available = ", ".join(
                f"{layout.get_info(idx).layer}:{layout.get_info(idx).datatype}"
                for idx in layer_indices
            )
            raise ValueError(
                f"Layout has {len(layer_indices)} layers; pass --layer LAYER:DTYPE "
                f"(available: [{available}])."
            )
        return int(layer_indices[0])

    if ":" not in layer:
        raise ValueError(f"--layer must be 'LAYER:DTYPE' (e.g. '1:0'); got {layer!r}")
    try:
        layer_num_s, dtype_s = layer.split(":", 1)
        layer_num = int(layer_num_s)
        dtype = int(dtype_s)
    except ValueError:
        raise ValueError(
            f"--layer must be 'LAYER:DTYPE' with integer components; got {layer!r}"
        ) from None

    for idx in layer_indices:
        info = layout.get_info(idx)
        if info.layer == layer_num and info.datatype == dtype:
            return int(idx)
    raise ValueError(f"Layer {layer!r} not found in layout.")
