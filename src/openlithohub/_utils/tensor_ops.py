"""Common tensor manipulation helpers."""

from __future__ import annotations

import torch


def ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 2D (H, W), taking channel 0 if multi-channel."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"Expected 2-4D tensor, got {tensor.ndim}D")
