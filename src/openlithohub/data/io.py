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
        px = int((point.x - bbox.left) * pixels_per_dbu)
        py_math = int((point.y - bbox.bottom) * pixels_per_dbu)
        py = (h_px - 1) - py_math
        return (max(0, min(px, w_px - 1)), max(0, min(py, h_px - 1)))

    while not shapes_iter.at_end():
        shape = shapes_iter.shape()
        # Recursive iteration walks across hierarchy — each shape's
        # geometry is in its own cell's coordinates, so transform up to
        # the top cell before projecting.
        trans = shapes_iter.trans()
        if shape.is_polygon() or shape.is_box():
            poly = shape.polygon if shape.is_polygon() else db.Polygon(shape.box)
            poly = poly.transformed(trans)

            hull = [_project(p) for p in poly.each_point_hull()]
            if len(hull) >= 3:
                drawer.polygon(hull, fill=255)
            for hole_idx in range(poly.holes()):
                hole = [_project(p) for p in poly.each_point_hole(hole_idx)]
                if len(hole) >= 3:
                    drawer.polygon(hole, fill=0)
        shapes_iter.next()

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
