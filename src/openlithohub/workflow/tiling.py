"""Full-chip tiling strategy for distributed processing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


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

    Uses a sliding window with configurable overlap. Boundary tiles are
    zero-padded to maintain uniform tile dimensions.

    Args:
        layout_tensor: Full layout as tensor (H, W).
        tile_size: Size of each square tile in pixels.
        overlap: Overlap between adjacent tiles for seamless stitching.

    Returns:
        List of Tile objects covering the full layout.

    Raises:
        ValueError: If overlap >= tile_size or tile_size <= 0.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")

    h, w = layout_tensor.shape[-2], layout_tensor.shape[-1]
    step = tile_size - overlap
    tiles: list[Tile] = []

    y = 0
    while y < h:
        x = 0
        while x < w:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile_data = layout_tensor[..., y:y_end, x:x_end]

            actual_h = y_end - y
            actual_w = x_end - x

            if actual_h < tile_size or actual_w < tile_size:
                pad_bottom = tile_size - actual_h
                pad_right = tile_size - actual_w
                tile_data = functional.pad(tile_data, (0, pad_right, 0, pad_bottom), value=0.0)

            tiles.append(Tile(
                tensor=tile_data,
                origin_x=x,
                origin_y=y,
                width=actual_w,
                height=actual_h,
                overlap=overlap,
            ))

            x += step
        y += step

    return tiles
