"""Common tensor manipulation helpers."""

from __future__ import annotations

import torch


def ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 2D (H, W).

    Accepts 2D (H, W), 3D (1, H, W), or 4D (1, 1, H, W). Singleton batch and
    channel dimensions are squeezed. Raises ``ValueError`` if any non-spatial
    dimension has size > 1 — silently dropping batched samples corrupts
    metrics. Callers that need batch support must iterate explicitly.
    """
    if tensor.ndim == 4:
        if tensor.shape[0] != 1 or tensor.shape[1] != 1:
            raise ValueError(
                "ensure_2d expects (1, 1, H, W); got "
                f"{tuple(tensor.shape)}. Iterate over the batch in the caller."
            )
        tensor = tensor[0, 0]
    elif tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(
                "ensure_2d expects (1, H, W); got "
                f"{tuple(tensor.shape)}. Iterate over the batch in the caller."
            )
        tensor = tensor[0]
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"Expected 2-4D tensor, got {tensor.ndim}D")
