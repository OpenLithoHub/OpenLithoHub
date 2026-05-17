"""Layout file parsing via KLayout Python API."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_layout(path: str | Path) -> dict[str, Any]:
    """Parse an OASIS or GDSII layout file.

    Args:
        path: Path to .oas or .gds file.

    Returns:
        Dictionary with 'cells', 'layers', 'bounding_box', and raw layout handle.
    """
    raise NotImplementedError(
        "Layout parsing not yet implemented. "
        "Planned: use klayout.db.Layout to read .oas/.gds, "
        "enumerate cells and layers, compute bounding boxes. "
        "Requires: pip install openlithohub[workflow]"
    )
