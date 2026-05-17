"""Data transforms for resolution alignment and normalization."""

from __future__ import annotations

import torch


def align_resolution(
    tensor: torch.Tensor,
    source_pixel_nm: float,
    target_pixel_nm: float,
) -> torch.Tensor:
    """Resample a tensor to match target pixel resolution.

    Args:
        tensor: Input tensor (H, W) or (C, H, W).
        source_pixel_nm: Current pixel size in nanometers.
        target_pixel_nm: Desired pixel size in nanometers.

    Returns:
        Resampled tensor at the target resolution.
    """
    raise NotImplementedError(
        "Resolution alignment not yet implemented. "
        "Planned: use torch.nn.functional.interpolate with "
        "scale_factor = source_pixel_nm / target_pixel_nm."
    )


def normalize_to_binary(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Threshold a continuous tensor to binary (0/1)."""
    return (tensor > threshold).float()
