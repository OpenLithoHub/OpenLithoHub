"""EPE distance heatmap and MRC violation overlay renderers.

Used by the HF Playground to surface *where* the mask is failing — a single
EPE-mean number is unhelpful when the user wants to know which routing
turned the metric red.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from openlithohub._utils.morphology import (
    binary_dilation,
    binary_erosion,
    distance_transform,
)
from openlithohub.benchmark.metrics.epe import _extract_edges


def _to_2d(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr.astype(np.float32))
    return arr.float()


def plot_epe_heatmap(
    predicted: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    pixel_size_nm: float = 1.0,
    ax: plt.Axes | None = None,
    title: str = "EPE Heatmap",
) -> plt.Figure:
    """Render predicted-edge pixels coloured by their distance (in nm) to the nearest target edge.

    The colormap is "turbo" capped at 5×pixel_size_nm or the actual max,
    whichever is smaller — keeps the dynamic range readable when one
    isolated edge segment is far away.
    """
    pred_t = _to_2d(predicted)
    tgt_t = _to_2d(target)

    pred_edges = _extract_edges(pred_t)
    tgt_edges = _extract_edges(tgt_t)

    # Distance transform of the *background* of the target edges = distance
    # from any pixel to the nearest target edge.
    edge_bg = (tgt_edges < 0.5).float()
    dist_to_target_edge = distance_transform(edge_bg)
    pred_edge_distance_nm = dist_to_target_edge * float(pixel_size_nm)

    # Mask everything that isn't a predicted edge pixel.
    masked = np.full(pred_t.shape, np.nan, dtype=np.float32)
    pe = pred_edges.numpy().astype(bool)
    masked[pe] = pred_edge_distance_nm.numpy()[pe]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
    else:
        fig = ax.figure

    ax.imshow(tgt_t.numpy(), cmap="gray", interpolation="nearest", alpha=0.4)
    finite = masked[np.isfinite(masked)]
    vmax = float(np.nanmax(finite)) if finite.size else float(pixel_size_nm)
    vmax = min(vmax, 5.0 * float(pixel_size_nm)) if vmax > 0 else float(pixel_size_nm)
    im = ax.imshow(masked, cmap="turbo", interpolation="nearest", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Edge displacement (nm)")
    return fig


def plot_mrc_overlay(
    mask: np.ndarray | torch.Tensor,
    *,
    min_width_nm: float = 40.0,
    min_spacing_nm: float = 40.0,
    pixel_size_nm: float = 1.0,
    ax: plt.Axes | None = None,
    title: str = "MRC Violations",
) -> plt.Figure:
    """Render the mask in grayscale with width and spacing violations highlighted in red.

    Width-violation pixels (foreground that disappears under morphological
    opening at radius_width) are flagged in pure red. Spacing violations
    (background that disappears under opening at radius_spacing) are
    flagged in red-orange so the two failure modes are visually distinct.
    """
    m = _to_2d(mask)
    binary = (m > 0.5).float()
    h, w = binary.shape

    radius_width = int(math.floor(min_width_nm / (2.0 * pixel_size_nm)))
    radius_spacing = int(math.floor(min_spacing_nm / (2.0 * pixel_size_nm)))

    rgba = np.stack(
        [binary.numpy(), binary.numpy(), binary.numpy(), np.ones((h, w), dtype=np.float32)],
        axis=-1,
    )

    width_count = 0
    spacing_count = 0

    if binary.sum() > 0 and radius_width >= 1:
        opened = binary_dilation(binary_erosion(binary, radius=radius_width), radius=radius_width)
        width_violation = ((binary > 0.5) & (opened < 0.5)).numpy()
        width_count = int(width_violation.sum())
        rgba[width_violation] = [1.0, 0.0, 0.0, 1.0]

    if binary.sum() > 0 and (1.0 - binary).sum() > 0 and radius_spacing >= 1:
        bg = (binary < 0.5).float()
        eroded_bg = binary_erosion(bg, radius=radius_spacing)
        opened_bg = binary_dilation(eroded_bg, radius=radius_spacing)
        spacing_violation = ((bg > 0.5) & (opened_bg < 0.5)).numpy()
        spacing_count = int(spacing_violation.sum())
        rgba[spacing_violation] = [1.0, 0.5, 0.0, 1.0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
    else:
        fig = ax.figure

    ax.imshow(rgba, interpolation="nearest")
    subtitle = f"width={width_count}px, spacing={spacing_count}px"
    ax.set_title(f"{title} ({subtitle})")
    ax.axis("off")
    return fig
