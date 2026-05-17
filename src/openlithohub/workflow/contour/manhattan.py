"""Manhattan (staircase) contour extraction for traditional VSB writers."""

from __future__ import annotations

import torch


def extract_manhattan_contour(
    mask: torch.Tensor,
    pixel_size_nm: float = 1.0,
) -> list[list[tuple[float, float]]]:
    """Extract Manhattan (rectilinear) polygon contours from a binary mask.

    Produces staircase-shaped polygons suitable for VSB mask writers.

    Args:
        mask: Binary mask tensor (H, W).
        pixel_size_nm: Physical pixel size for coordinate scaling.

    Returns:
        List of polygons, each as a list of (x_nm, y_nm) vertices.
    """
    raise NotImplementedError(
        "Manhattan contour extraction not yet implemented. "
        "Planned: use marching squares for initial contour, "
        "then snap all vertices to grid (Manhattanize). "
        "Reference: EasyMRC Manhattanization (TODAES'25)."
    )
