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
    """VSB estimation: approximate rectangle decomposition via complexity heuristic."""
    h, w = binary.shape
    b = binary > 0.5
    # Boundary detection compares each foreground pixel against its 4 neighbours
    # under zero-padding (pixels outside the canvas are treated as background).
    # `torch.roll` wraps circularly, which would mark border-touching shapes as
    # bordered on the opposite edge as well — inflating perimeter for cropped
    # tiles, which is the common case.
    zero_row = torch.zeros((1, w), dtype=b.dtype, device=b.device)
    zero_col = torch.zeros((h, 1), dtype=b.dtype, device=b.device)
    up = torch.cat([b[1:, :], zero_row], dim=0)
    down = torch.cat([zero_row, b[:-1, :]], dim=0)
    left = torch.cat([b[:, 1:], zero_col], dim=1)
    right = torch.cat([zero_col, b[:, :-1]], dim=1)
    boundary = ((b != up) | (b != down) | (b != left) | (b != right)) & b
    perimeter_pixels = int(boundary.sum().item())

    # Complexity ratio: higher perimeter/area means more shots needed
    # For a simple rectangle: perimeter ~ 4*sqrt(area), needing 1 shot
    # For complex shapes: more rectangular decomposition needed
    if perimeter_pixels == 0:
        shot_count = 1
    else:
        max_shot_area_nm2 = min_shot_size_nm * min_shot_size_nm * 1024
        complexity = (perimeter_pixels * perimeter_pixels) / max(1, foreground_pixels)
        avg_shot_area_px = max(1.0, foreground_pixels / max(1.0, complexity * 0.25))
        max_area_px = max_shot_area_nm2 / (pixel_size_nm * pixel_size_nm)
        avg_shot_area_px = min(avg_shot_area_px, max_area_px)
        shot_count = max(1, int(foreground_pixels / avg_shot_area_px + 0.5))

    write_time_s = shot_count * _VSB_SHOT_TIME_S
    return {"shot_count": shot_count, "estimated_write_time_s": write_time_s}
