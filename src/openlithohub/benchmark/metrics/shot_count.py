"""Shot count estimation for mask manufacturing cost."""

from __future__ import annotations

import torch

from openlithohub._utils.tensor_ops import ensure_2d

# Typical beam rates for write time estimation
_MBMW_BEAM_RATE = 1.0e9  # shots/second for multi-beam writers
_VSB_SHOT_TIME_S = 20.0e-9  # 20ns per shot for VSB writers


def estimate_shot_count(
    mask: torch.Tensor,
    writer_type: str = "mbmw",
    min_shot_size_nm: float = 5.0,
    pixel_size_nm: float = 1.0,
) -> dict[str, int | float]:
    """Estimate the number of shots needed to write a mask.

    Shot count is a direct proxy for mask writing time and manufacturing cost.

    For multi-beam mask writers (MBMW), each foreground pixel corresponds to
    one beam exposure position. Shot count equals the number of foreground pixels
    scaled by the ratio of pixel area to beam grid area.

    For variable shaped beam (VSB) writers, shots are rectangular exposures.
    The estimate uses the mask complexity (perimeter/area ratio) to approximate
    the number of rectangles needed.

    Args:
        mask: Binary mask tensor (H, W).
        writer_type: 'vsb' (variable shaped beam) or 'mbmw' (multi-beam).
        min_shot_size_nm: Minimum addressable shot dimension.
        pixel_size_nm: Physical pixel size in nanometers.

    Returns:
        Dictionary with 'shot_count' and 'estimated_write_time_s'.

    Raises:
        ValueError: If writer_type is not 'mbmw' or 'vsb'.
    """
    if writer_type not in ("mbmw", "vsb"):
        raise ValueError(f"writer_type must be 'mbmw' or 'vsb', got '{writer_type}'")

    m = ensure_2d(mask)
    binary = (m > 0.5).float()

    foreground_pixels = int(binary.sum().item())

    if foreground_pixels == 0:
        return {"shot_count": 0, "estimated_write_time_s": 0.0}

    if writer_type == "mbmw":
        return _estimate_mbmw(binary, foreground_pixels, min_shot_size_nm, pixel_size_nm)
    return _estimate_vsb(binary, foreground_pixels, min_shot_size_nm, pixel_size_nm)


def _estimate_mbmw(
    binary: torch.Tensor,
    foreground_pixels: int,
    min_shot_size_nm: float,
    pixel_size_nm: float,
) -> dict[str, int | float]:
    """MBMW estimation: each beam grid cell covering foreground is one shot."""
    grid_pitch_px = max(1.0, min_shot_size_nm / pixel_size_nm)
    shots_per_pixel = 1.0 / (grid_pitch_px * grid_pitch_px)
    shot_count = max(1, int(foreground_pixels * shots_per_pixel + 0.5))
    write_time_s = shot_count / _MBMW_BEAM_RATE
    return {"shot_count": shot_count, "estimated_write_time_s": write_time_s}


def _estimate_vsb(
    binary: torch.Tensor,
    foreground_pixels: int,
    min_shot_size_nm: float,
    pixel_size_nm: float,
) -> dict[str, int | float]:
    """VSB estimation: rectilinear-polygon rectangle-decomposition lower bound.

    For axis-aligned rectilinear polygons, the minimum number of
    axis-aligned rectangles needed to tile a single hole-free component is
    bounded above by ``ceil(h / 2) + 1`` where ``h`` is the count of
    reflex (concave) vertices (Lipski et al. 1979 / Imai-Asano 1986). The
    simpler ``h + 1`` bound double-counts paired concave corners — a plus
    shape has 4 reflex corners but can be tiled with 3 rectangles
    (horizontal bar + two stubs), not 5.

    Computing the optimum requires a max bipartite matching of good
    chords; we use the practical ``ceil(h / 2) + 1`` per-component
    approximation, then add a write-field cap split by ``max_shot_area``.

    A perfect rectangle has zero concave corners → 1 shot regardless of
    aspect ratio (the previous ``perimeter² / area`` heuristic reported
    50 shots for a 100×2 stripe and 4 shots for a 100×100 square — both
    wrong).
    """
    from openlithohub._utils.morphology import connected_components

    # Pad once with a one-pixel background border so 2×2 corner kernels
    # never spill off the canvas — equivalent to the zero-pad boundary
    # convention used elsewhere in this module.
    h, w = binary.shape
    bg = torch.zeros((h + 2, w + 2), dtype=binary.dtype, device=binary.device)
    bg[1:-1, 1:-1] = binary > 0.5
    b = bg

    # 2×2 block sum: each interior pixel position counts how many of the
    # four pixels meeting at that corner are foreground. A concave corner
    # is where exactly three of the four are foreground (the fourth is
    # the "bite" out of the rectilinear contour). Convex corners have
    # exactly one foreground pixel, edges have two, interior has four.
    block = b[:-1, :-1] + b[1:, :-1] + b[:-1, 1:] + b[1:, 1:]
    concave_mask = block == 3

    labels, n_components = connected_components(binary > 0.5, connectivity=4)
    if n_components == 0:
        return {"shot_count": 0, "estimated_write_time_s": 0.0}

    # Per-component concave-corner counts. The 2x2 block grid is offset
    # by one pixel from the foreground grid (each block cell straddles
    # four pixels); the ``labels`` tensor lives on the original grid, so
    # we attribute each concave corner to the foreground pixel at the
    # same (y+1, x+1) offset (the 2x2 block at (y, x) corresponds to the
    # padded array, so subtract one to map back, then clamp).
    rect_decomposition = 0
    if concave_mask.any():
        # Map each concave corner back to a representative foreground pixel
        # to look up its component label. The 2x2 block at padded coord
        # (yp, xp) covers pixels (yp-1..yp, xp-1..xp); pick the top-left
        # foreground pixel in that block as the representative.
        ys_pad, xs_pad = torch.where(concave_mask)
        # Convert padded coords to original-grid coords; clamp to valid range.
        ys_rep = torch.clamp(ys_pad - 1, 0, h - 1)
        xs_rep = torch.clamp(xs_pad - 1, 0, w - 1)
        rep_labels = labels[ys_rep, xs_rep]
        # Some representatives may land on a background pixel of the
        # padded view; fall back to neighbour lookups.
        for offset_y, offset_x in ((0, 0), (0, 1), (1, 0), (1, 1)):
            mask = rep_labels < 0
            if not mask.any():
                break
            ys_alt = torch.clamp(ys_pad - 1 + offset_y, 0, h - 1)
            xs_alt = torch.clamp(xs_pad - 1 + offset_x, 0, w - 1)
            alt_labels = labels[ys_alt[mask], xs_alt[mask]]
            rep_labels = rep_labels.clone()
            rep_labels[mask] = alt_labels
        valid = rep_labels >= 0
        valid_labels = rep_labels[valid]
        if valid_labels.numel() > 0:
            unique_labels, counts = torch.unique(valid_labels, return_counts=True)
            # Per-component contribution: ceil(h_i / 2) + 1.
            for h_i in counts.tolist():
                rect_decomposition += (int(h_i) + 1) // 2 + 1
            # Components with no concave corners contribute 1 each.
            rect_decomposition += n_components - int(unique_labels.numel())
        else:
            rect_decomposition = n_components
    else:
        rect_decomposition = n_components

    # VSB write-field cap: a single rectangle cannot exceed the writer's
    # max field. Approximate by area. This only matters for large solid
    # regions; concave-heavy patterns are already shot-bound.
    max_shot_area_px = max(1.0, (min_shot_size_nm * 1024) ** 2 / pixel_size_nm**2)
    field_cap_extra = max(0, int(foreground_pixels / max_shot_area_px + 0.5) - n_components)

    shot_count = max(1, rect_decomposition + field_cap_extra)
    write_time_s = shot_count * _VSB_SHOT_TIME_S
    return {"shot_count": shot_count, "estimated_write_time_s": write_time_s}
