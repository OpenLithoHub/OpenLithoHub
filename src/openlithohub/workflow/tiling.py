"""Full-chip tiling strategy for distributed processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
    *anchored* to the layout edge (origin pulled back so the tile fits
    entirely inside the layout) so the model sees real layout context
    instead of zero-padding artefacts. The resulting boundary tiles overlap
    further with their neighbours, which ``stitch_tiles`` blends correctly
    via its weight-map normalization.

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
    seen_origins: set[tuple[int, int]] = set()

    # Layout smaller than tile_size in an axis ⇒ a single tile in that axis
    # is sufficient. Without this guard, the sliding window still emits one
    # tile per `step` along the axis, all zero-padded duplicates of the
    # entire layout, multiplying forward-model work for no coverage gain.
    h_iter_cap = 1 if h < tile_size else h
    w_iter_cap = 1 if w < tile_size else w

    y = 0
    while y < h_iter_cap:
        x = 0
        while x < w_iter_cap:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            actual_h = y_end - y
            actual_w = x_end - x

            if actual_h < tile_size and h >= tile_size:
                # Anchor the tile to the bottom edge so its full extent is
                # filled with real layout, not zero pad. This pulls the
                # origin back; the extra coverage overlaps the previous row
                # and is handled by stitch_tiles' weight-map blending.
                y_origin = h - tile_size
                tile_h_real = tile_size
            else:
                y_origin = y
                tile_h_real = actual_h

            if actual_w < tile_size and w >= tile_size:
                x_origin = w - tile_size
                tile_w_real = tile_size
            else:
                x_origin = x
                tile_w_real = actual_w

            y_real_end = y_origin + tile_h_real
            x_real_end = x_origin + tile_w_real
            tile_data = layout_tensor[..., y_origin:y_real_end, x_origin:x_real_end]

            if tile_h_real < tile_size or tile_w_real < tile_size:
                # Layout smaller than tile_size in some axis — must zero-pad,
                # there is no real layout to anchor to.
                pad_bottom = tile_size - tile_h_real
                pad_right = tile_size - tile_w_real
                tile_data = functional.pad(tile_data, (0, pad_right, 0, pad_bottom), value=0.0)

            # When the sliding window's last position past the layout edge
            # gets anchored back to the same origin as a previous tile, skip
            # it — emitting two tiles with identical origins doubles forward-
            # model work and does not change the stitched output.
            origin_key = (x_origin, y_origin)
            if origin_key not in seen_origins:
                seen_origins.add(origin_key)
                tiles.append(
                    Tile(
                        tensor=tile_data,
                        origin_x=x_origin,
                        origin_y=y_origin,
                        width=tile_w_real,
                        height=tile_h_real,
                        overlap=overlap,
                    )
                )

            x += step
        y += step

    return tiles


def stitch_tiles(
    tiles: list[tuple[Tile, torch.Tensor]],
    output_shape: tuple[int, int],
) -> torch.Tensor:
    """Reassemble optimized tiles into a full-chip tensor.

    Uses linear blending in overlap regions to avoid seam artifacts.

    Args:
        tiles: List of (original_tile, optimized_tensor) pairs.
        output_shape: (H, W) of the full output tensor.

    Returns:
        Stitched tensor of shape output_shape.
    """
    h, w = output_shape
    device = tiles[0][1].device if tiles else torch.device("cpu")
    output = torch.zeros(h, w, device=device)
    weight_map = torch.zeros(h, w, device=device)

    for tile, result in tiles:
        tile_h = tile.height
        tile_w = tile.width
        result_2d = result[..., :tile_h, :tile_w]
        if result_2d.ndim > 2:
            result_2d = result_2d.squeeze()

        blend = torch.ones(tile_h, tile_w, device=device)

        if tile.overlap > 0:
            ramp = torch.linspace(0.0, 1.0, tile.overlap, device=device)

            if tile.origin_x > 0:
                left_ramp = ramp.unsqueeze(0).expand(tile_h, -1)
                ol = min(tile.overlap, tile_w)
                blend[:, :ol] *= left_ramp[:, :ol]

            if tile.origin_y > 0:
                top_ramp = ramp.unsqueeze(1).expand(-1, tile_w)
                ol = min(tile.overlap, tile_h)
                blend[:ol, :] *= top_ramp[:ol, :]

            if tile.origin_x + tile_w < w:
                right_ramp = ramp.flip(0).unsqueeze(0).expand(tile_h, -1)
                ol = min(tile.overlap, tile_w)
                blend[:, -ol:] *= right_ramp[:, -ol:]

            if tile.origin_y + tile_h < h:
                bottom_ramp = ramp.flip(0).unsqueeze(1).expand(-1, tile_w)
                ol = min(tile.overlap, tile_h)
                blend[-ol:, :] *= bottom_ramp[-ol:, :]

        y_end = tile.origin_y + tile_h
        x_end = tile.origin_x + tile_w
        output[tile.origin_y : y_end, tile.origin_x : x_end] += result_2d * blend
        weight_map[tile.origin_y : y_end, tile.origin_x : x_end] += blend

    nonzero = weight_map > 0
    output[nonzero] /= weight_map[nonzero]
    return output


def tiled_ilt_with_consistency(
    mask: torch.Tensor,
    tile_size: int,
    ilt_fn: Callable[[torch.Tensor], torch.Tensor],
    overlap: int = 16,
    n_iterations: int = 10,
) -> dict:
    """Run tiled ILT and measure cross-tile consistency.

    Partitions ``mask`` into tiles, applies ``ilt_fn`` independently to each
    tile for ``n_iterations``, stitches the results back together, and
    evaluates boundary consistency metrics.

    Args:
        mask: Full-chip mask tensor ``(H, W)``.
        tile_size: Tile size in pixels.
        ilt_fn: Callable that takes a tile mask ``(H_tile, W_tile)`` and
            returns an optimised mask of the same shape. Called once per tile
            (not iteratively — ``n_iterations`` controls a simple iterative
            refinement loop if the caller passes a stateless function).
        overlap: Overlap between adjacent tiles.
        n_iterations: Number of refinement iterations per tile. Each iteration
            applies ``ilt_fn`` to the current tile result.

    Returns:
        Dictionary with:

        - ``'mask'``: stitched optimised mask ``(H, W)``.
        - ``'tiles'``: list of original ``Tile`` objects.
        - ``'tile_results'``: list of per-tile optimised tensors.
        - ``'consistency'``: output of :func:`tile_boundary_consistency`.
    """
    from openlithohub.benchmark.metrics.tiling_consistency import tile_boundary_consistency

    if mask.ndim > 2:
        mask = mask.squeeze()

    h, w = mask.shape
    tiles = tile_layout(mask, tile_size=tile_size, overlap=overlap)

    tile_results: list[torch.Tensor] = []
    for tile in tiles:
        current = tile.tensor.clone()
        for _ in range(n_iterations):
            current = ilt_fn(current)
        tile_results.append(current)

    stitched = stitch_tiles(
        [(t, r) for t, r in zip(tiles, tile_results)],
        (h, w),
    )

    consistency = tile_boundary_consistency(tiles, tile_results, overlap=overlap)

    return {
        "mask": stitched,
        "tiles": tiles,
        "tile_results": tile_results,
        "consistency": consistency,
    }
