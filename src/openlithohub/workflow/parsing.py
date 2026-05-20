"""Layout file parsing via KLayout Python API."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_layout(
    path: str | Path,
    *,
    lef_files: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Parse an OASIS / GDSII / DEF / LEF layout file.

    Returns dictionary with 'cells', 'layers', 'bounding_box', and '_layout' handle.

    DEF (Design Exchange Format, IEEE 1481) is the standard placed-and-routed
    layout dump produced by Innovus / ICC2 / OpenROAD. DEF carries placement
    + routing geometry but **not** cell internals — those live in companion
    LEF files. Pass ``lef_files=[...]`` to feed cell abstracts so DEF
    components resolve to real polygons rather than empty placeholders.

    LEF-only inputs (``.lef``) are also accepted, for tools that want to
    inspect cell abstracts without a placed design.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Layout file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in (".oas", ".gds", ".gds2", ".oasis", ".def", ".lef"):
        raise ValueError(f"Unsupported layout format: {suffix}. Use .oas / .gds / .def / .lef")

    try:
        import klayout.db as db
    except ImportError:
        raise ImportError(
            "klayout is required for layout parsing. "
            "Install with: pip install openlithohub[workflow]"
        ) from None

    layout = db.Layout()
    if suffix in (".def", ".lef"):
        # DEF/LEF reader: stream LEF files in via the reader options so
        # cell abstracts resolve when we read the DEF. Without LEF, a DEF
        # parse yields component placements but no internal geometry —
        # which is rarely what callers want from `parse_layout`.
        opts = db.LoadLayoutOptions()
        if lef_files:
            opts.lefdef_config.lef_files = [str(Path(p)) for p in lef_files]
            # When the caller hands us explicit LEF files, suppress
            # KLayout's auto-rescan of the DEF directory for *.lef —
            # otherwise the same LEF gets parsed twice and KLayout
            # raises "Duplicate MACRO" before we ever see the geometry.
            opts.lefdef_config.read_lef_with_def = False
        layout.read(str(path), opts)
    else:
        layout.read(str(path))

    cells: list[dict[str, Any]] = []
    for cell_idx in range(layout.cells()):
        cell = layout.cell(cell_idx)
        cells.append(
            {
                "name": cell.name,
                "index": cell_idx,
                "instance_count": cell.child_instances(),
            }
        )

    layers: list[dict[str, Any]] = []
    for layer_idx in layout.layer_indices():
        info = layout.get_info(layer_idx)
        layers.append(
            {
                "layer": info.layer,
                "datatype": info.datatype,
                "name": info.name if info.name else f"{info.layer}/{info.datatype}",
            }
        )

    top_cells = list(layout.top_cells())
    if not top_cells:
        raise ValueError(f"Layout {path.name} has no top cells.")
    top_cell = top_cells[0]
    bbox = top_cell.bbox()
    bounding_box = {
        "x_min": bbox.left,
        "y_min": bbox.bottom,
        "x_max": bbox.right,
        "y_max": bbox.top,
        "dbu": layout.dbu,
    }

    return {
        "cells": cells,
        "layers": layers,
        "bounding_box": bounding_box,
        "_layout": layout,
    }
