"""Rich display functions for lithography tensors in Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import torch


def display_mask(
    mask: torch.Tensor,
    title: str = "Mask",
    pixel_size_nm: float = 1.0,
    figsize: tuple[int, int] = (6, 6),
) -> Any:
    """Display a binary mask tensor as an image in Jupyter.

    Args:
        mask: Binary mask tensor (H, W).
        title: Plot title.
        pixel_size_nm: Pixel size for axis labeling.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object (also displayed inline in Jupyter).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for display functions. "
            "Install with: pip install openlithohub[jupyter]"
        ) from None

    arr = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    h, w = arr.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    extent = [0, w * pixel_size_nm, h * pixel_size_nm, 0]
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1, extent=extent, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    plt.tight_layout()
    return fig


def display_comparison(
    predicted: torch.Tensor,
    target: torch.Tensor,
    title: str = "Predicted vs Target",
    pixel_size_nm: float = 1.0,
    figsize: tuple[int, int] = (12, 5),
) -> Any:
    """Display side-by-side comparison of predicted and target masks.

    Also shows an overlay where:
    - Green = target edges
    - Red = predicted edges
    - Yellow = overlapping edges

    Args:
        predicted: Predicted binary mask (H, W).
        target: Target binary mask (H, W).
        title: Overall figure title.
        pixel_size_nm: Pixel size for axis labeling.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError(
            "matplotlib is required for display functions. "
            "Install with: pip install openlithohub[jupyter]"
        ) from None

    pred_arr = (
        predicted.detach().cpu().numpy() if isinstance(predicted, torch.Tensor) else predicted
    )
    tgt_arr = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else target
    h, w = pred_arr.shape[:2]
    extent = [0, w * pixel_size_nm, h * pixel_size_nm, 0]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(pred_arr, cmap="gray", vmin=0, vmax=1, extent=extent, interpolation="nearest")
    axes[0].set_title("Predicted")

    axes[1].imshow(tgt_arr, cmap="gray", vmin=0, vmax=1, extent=extent, interpolation="nearest")
    axes[1].set_title("Target")

    # Overlay: R=predicted edges, G=target edges
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    pred_bin = (pred_arr > 0.5).astype(np.float32)
    tgt_bin = (tgt_arr > 0.5).astype(np.float32)
    overlay[:, :, 0] = pred_bin * (1.0 - tgt_bin)  # Red: predicted only
    overlay[:, :, 1] = tgt_bin * (1.0 - pred_bin)  # Green: target only
    overlay[:, :, 2] = 0.0
    # Yellow where both
    both = pred_bin * tgt_bin
    overlay[:, :, 0] += both * 0.8
    overlay[:, :, 1] += both * 0.8
    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay, extent=extent, interpolation="nearest")
    axes[2].set_title("Overlay (R=pred, G=tgt, Y=both)")

    for ax in axes:
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")

    fig.suptitle(title)
    plt.tight_layout()
    return fig
