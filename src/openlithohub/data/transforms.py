"""Data transforms for resolution alignment and normalization."""

from __future__ import annotations

import torch
import torch.nn.functional as f


def align_resolution(
    tensor: torch.Tensor,
    source_pixel_nm: float,
    target_pixel_nm: float,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resample a tensor to match target pixel resolution.

    Args:
        tensor: Input tensor (H, W) or (C, H, W).
        source_pixel_nm: Current pixel size in nanometers.
        target_pixel_nm: Desired pixel size in nanometers.
        mode: Interpolation mode ('bilinear', 'nearest', 'bicubic').

    Returns:
        Resampled tensor at the target resolution.
    """
    if source_pixel_nm <= 0 or target_pixel_nm <= 0:
        raise ValueError("Pixel sizes must be positive")

    scale = source_pixel_nm / target_pixel_nm
    if abs(scale - 1.0) < 1e-6:
        return tensor

    ndim = tensor.ndim
    if ndim == 2:
        x = tensor.unsqueeze(0).unsqueeze(0)
    elif ndim == 3:
        x = tensor.unsqueeze(0)
    else:
        raise ValueError(f"Expected 2D (H,W) or 3D (C,H,W) tensor, got {ndim}D")

    align_corners = None if mode == "nearest" else False
    x = f.interpolate(x, scale_factor=scale, mode=mode, align_corners=align_corners)

    if ndim == 2:
        return x.squeeze(0).squeeze(0)
    return x.squeeze(0)


def normalize_to_binary(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Threshold a continuous tensor to binary (0/1)."""
    return (tensor > threshold).float()
