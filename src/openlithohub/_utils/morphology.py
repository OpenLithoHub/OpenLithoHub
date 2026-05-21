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
    # Worst-case chessboard distance for a near-full mask is bounded by
    # max(h, w) // 2 + 1 — using min() under-clips elongated foreground.
    # We add the early `current.sum() == 0` short-circuit to keep the cost
    # in line with the actual distance reached.
    max_iterations = max(h, w) // 2 + 1

    for _ in range(max_iterations):
        if current.sum() == 0:
            break
        dist += current.squeeze(0).squeeze(0)
        inverted = 1.0 - current
        dilated_inv = functional.max_pool2d(inverted, kernel_size=3, stride=1, padding=1)
        current = 1.0 - dilated_inv

    return dist


def connected_components(mask: torch.Tensor, connectivity: int = 8) -> tuple[torch.Tensor, int]:
    """Label connected components of a binary mask via iterative neighborhood min.

    Each foreground pixel is initially assigned a unique label (its flat index).
    The labels are then iteratively replaced by the minimum label in the
    pixel's neighborhood (intersected with foreground). Convergence happens in
    O(component-diameter) iterations and runs entirely on GPU via min-pool
    primitives — orders of magnitude faster than per-pixel BFS for large
    masks.

    Args:
        mask: Binary tensor (H, W) with values in {0, 1}.
        connectivity: 4 (von Neumann) or 8 (Moore). Default 8 matches the
            erosion-based components in `mrc._connected_component_areas` and
            the FG/BG labelling in `stochastic._nominal_state`. The 4-connectivity
            kernel rules out diagonal merges, which matches `drc._check_min_area`'s
            historical behaviour.

    Returns:
        labels: int64 tensor (H, W). Background pixels are -1; foreground pixels
            share an integer label per component (labels are not necessarily
            contiguous — caller may remap with ``unique`` if dense IDs are needed).
        num_components: number of distinct connected components.
    """
    fg = mask > 0.5
    h, w = fg.shape
    if not fg.any():
        return torch.full((h, w), -1, dtype=torch.int64, device=fg.device), 0

    flat_idx = torch.arange(h * w, device=fg.device, dtype=torch.int64).reshape(h, w)
    sentinel = h * w + 1
    labels = torch.where(fg, flat_idx, torch.full_like(flat_idx, sentinel))

    if connectivity == 4:
        # 4-connectivity via cross-shaped neighborhood. Operates on int64 via
        # explicit shift-and-pad rather than float32 max_pool2d — float32's
        # 24-bit mantissa would silently collide for masks larger than ~16
        # megapixels.
        prev_sum = labels.sum().item() + 1
        while True:
            up = functional.pad(labels[1:, :], (0, 0, 0, 1), value=sentinel)
            down = functional.pad(labels[:-1, :], (0, 0, 1, 0), value=sentinel)
            left = functional.pad(labels[:, 1:], (0, 1, 0, 0), value=sentinel)
            right = functional.pad(labels[:, :-1], (1, 0, 0, 0), value=sentinel)
            stacked = torch.stack([labels, up, down, left, right], dim=0)
            new_labels = stacked.amin(dim=0)
            new_labels = torch.where(fg, new_labels, torch.full_like(new_labels, sentinel))
            curr_sum = int(new_labels.sum().item())
            labels = new_labels
            if curr_sum == prev_sum:
                break
            prev_sum = curr_sum
    elif connectivity == 8:
        # 8-connectivity = 3x3 min over neighborhood. Computed on int64 via
        # nine shifted copies stacked along dim 0 — avoids float32 max_pool2d
        # mantissa collisions on >~16 megapixel masks.
        prev_sum = labels.sum().item() + 1
        while True:
            shifts = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    src = labels
                    if dy == 1:
                        src = src[1:, :]
                    elif dy == -1:
                        src = src[:-1, :]
                    if dx == 1:
                        src = src[:, 1:]
                    elif dx == -1:
                        src = src[:, :-1]
                    pad_left = 1 if dx == -1 else 0
                    pad_right = 1 if dx == 1 else 0
                    pad_top = 1 if dy == -1 else 0
                    pad_bottom = 1 if dy == 1 else 0
                    shifts.append(
                        functional.pad(
                            src,
                            (pad_left, pad_right, pad_top, pad_bottom),
                            value=sentinel,
                        )
                    )
            stacked = torch.stack(shifts, dim=0)
            new_labels = stacked.amin(dim=0)
            new_labels = torch.where(fg, new_labels, torch.full_like(new_labels, sentinel))
            curr_sum = int(new_labels.sum().item())
            labels = new_labels
            if curr_sum == prev_sum:
                break
            prev_sum = curr_sum
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    labels = torch.where(fg, labels, torch.full_like(labels, -1))
    fg_labels = labels[fg]
    num_components = int(torch.unique(fg_labels).numel())
    return labels, num_components
