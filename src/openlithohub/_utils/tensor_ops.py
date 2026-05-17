"""Common tensor manipulation helpers."""

from __future__ import annotations

import torch


def ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 2D (H, W), squeezing batch/channel dims if needed."""
    if tensor.ndim == 4:
        return tensor.squeeze(0).squeeze(0)
    if tensor.ndim == 3:
        return tensor.squeeze(0)
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"Expected 2-4D tensor, got {tensor.ndim}D")
