"""Edge Placement Error (EPE) computation."""

from __future__ import annotations

import torch


def compute_epe(
    predicted: torch.Tensor,
    target: torch.Tensor,
    pixel_size_nm: float = 1.0,
) -> dict[str, float]:
    """Compute Edge Placement Error between predicted and target contours.

    Args:
        predicted: Binary mask of predicted edges (H, W).
        target: Binary mask of target/reference edges (H, W).
        pixel_size_nm: Physical size of each pixel in nanometers.

    Returns:
        Dictionary with 'epe_mean_nm', 'epe_max_nm', 'epe_std_nm'.
    """
    raise NotImplementedError(
        "EPE computation not yet implemented. "
        "Planned: extract edge pixels via Sobel operator, compute "
        "minimum distance between edge point sets, scale by pixel_size_nm. "
        "Reference: LithoBench EPE definition (NeurIPS'23)."
    )
