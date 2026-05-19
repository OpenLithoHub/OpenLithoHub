"""Layout file parsing via KLayout Python API."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_layout(path: str | Path) -> dict[str, Any]:
    """Parse an OASIS or GDSII layout file.

    Returns dictionary with 'cells', 'layers', 'bounding_box', and '_layout' handle.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Layout file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in (".oas", ".gds", ".gds2", ".oasis"):
        raise ValueError(f"Unsupported layout format: {suffix}. Use .oas or .gds")

    try:
        import klayout.db as db
    except ImportError:
        raise ImportError(
            "klayout is required for layout parsing. "
            "Install with: pip install openlithohub[workflow]"
        ) from None

    layout = db.Layout()
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
