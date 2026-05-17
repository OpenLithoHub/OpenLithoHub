"""Shot count estimation for mask manufacturing cost."""

from __future__ import annotations

import torch


def estimate_shot_count(
    mask: torch.Tensor,
    writer_type: str = "mbmw",
    min_shot_size_nm: float = 5.0,
) -> dict[str, int | float]:
    """Estimate the number of shots needed to write a mask.

    Shot count is a direct proxy for mask writing time and manufacturing cost.

    Args:
        mask: Binary mask tensor (H, W).
        writer_type: 'vsb' (variable shaped beam) or 'mbmw' (multi-beam).
        min_shot_size_nm: Minimum addressable shot dimension.

    Returns:
        Dictionary with 'shot_count', 'estimated_write_time_s'.
    """
    raise NotImplementedError(
        "Shot count estimation not yet implemented. "
        "Planned: for VSB, decompose mask into non-overlapping rectangles; "
        "for MBMW, estimate based on pattern complexity and beam grid. "
        "Reference: OASIS.MBW 2.1 shot decomposition rules."
    )
