"""Morphological operations for binary mask analysis."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def binary_erosion(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Erode a binary mask using a square structuring element.

    Args:
        mask: Binary tensor (H, W) with values in {0, 1}.
        radius: Erosion radius in pixels (kernel size = 2*radius + 1).

    Returns:
        Eroded binary mask of the same shape.
    """
    if radius <= 0:
        return mask.clone()
    kernel_size = 2 * radius + 1
    inp = mask.float().unsqueeze(0).unsqueeze(0)
    inverted = 1.0 - inp
    dilated_inv = functional.max_pool2d(inverted, kernel_size=kernel_size, stride=1, padding=radius)
    return (1.0 - dilated_inv).squeeze(0).squeeze(0)


def binary_dilation(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Dilate a binary mask using a square structuring element.

    Args:
        mask: Binary tensor (H, W) with values in {0, 1}.
        radius: Dilation radius in pixels (kernel size = 2*radius + 1).

    Returns:
        Dilated binary mask of the same shape.
    """
    if radius <= 0:
        return mask.clone()
    kernel_size = 2 * radius + 1
    inp = mask.float().unsqueeze(0).unsqueeze(0)
    dilated = functional.max_pool2d(inp, kernel_size=kernel_size, stride=1, padding=radius)
    return dilated.squeeze(0).squeeze(0)


def distance_transform(mask: torch.Tensor) -> torch.Tensor:
    """Compute L-infinity distance transform of a binary mask.

    For each foreground pixel (value=1), computes the distance to the nearest
    background pixel using the chessboard (L-infinity) metric. Background pixels
    get distance 0.

    Uses iterative erosion — each iteration peels one pixel layer, so the
    iteration count at which a pixel vanishes equals its distance. Implemented
    via max_pool2d for GPU-friendly vectorized execution.

    Args:
        mask: Binary tensor (H, W) with values in {0, 1}.

    Returns:
        Distance transform tensor (H, W) with integer distances (as float).
    """
    fg = (mask > 0.5).float()
    if fg.sum() == 0:
        return torch.zeros_like(fg)

    # If entire mask is foreground, distance is determined by distance to edge
    h, w = fg.shape
    if fg.sum() == h * w:
        y_dist = torch.arange(h, dtype=fg.dtype, device=fg.device).unsqueeze(1).expand(h, w)
        y_dist = torch.minimum(y_dist, (h - 1) - y_dist)
        x_dist = torch.arange(w, dtype=fg.dtype, device=fg.device).unsqueeze(0).expand(h, w)
        x_dist = torch.minimum(x_dist, (w - 1) - x_dist)
        return torch.minimum(y_dist, x_dist) + 1.0

    dist = torch.zeros_like(fg)
    current = fg.unsqueeze(0).unsqueeze(0)
    max_iterations = min(h, w) // 2 + 1

    for _ in range(max_iterations):
        if current.sum() == 0:
            break
        dist += current.squeeze(0).squeeze(0)
        inverted = 1.0 - current
        dilated_inv = functional.max_pool2d(inverted, kernel_size=3, stride=1, padding=1)
        current = 1.0 - dilated_inv

    return dist
