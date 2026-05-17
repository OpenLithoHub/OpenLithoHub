"""Full-chip tiling strategy for distributed processing."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Tile:
    """A single tile from a layout partition."""

    tensor: torch.Tensor
    origin_x: int
    origin_y: int
    width: int
    height: int
    overlap: int


def tile_layout(
    layout_tensor: torch.Tensor,
    tile_size: int = 2048,
    overlap: int = 128,
) -> list[Tile]:
    """Partition a full-chip layout tensor into overlapping tiles.

    Args:
        layout_tensor: Full layout as tensor (H, W).
        tile_size: Size of each square tile in pixels.
        overlap: Overlap between adjacent tiles for seamless stitching.

    Returns:
        List of Tile objects covering the full layout.
    """
    raise NotImplementedError(
        "Layout tiling not yet implemented. "
        "Planned: sliding window with configurable overlap, "
        "edge padding for boundary tiles, "
        "metadata tracking for reassembly after per-tile optimization."
    )
