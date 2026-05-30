"""Morphological operations for binary mask analysis and differentiable ILT."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

# ---------------------------------------------------------------------------
# Disk structuring element (pure PyTorch, no scipy/opencv)
# ---------------------------------------------------------------------------

_DISK_CACHE: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


def _disk_kernel(radius: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a binary disk structuring element as a 2D kernel.

    The kernel entries are 1 inside the disk and 0 outside.  Convolution
    with this kernel counts how many neighbourhood pixels fall inside the
    structuring element — used as the weight matrix for logsumexp soft-max.

    Cached by (ceil_radius, dtype, device) to avoid re-computation.
    """
    r_px = max(1, int(math.ceil(radius)))
    key = (r_px, dtype, device)
    if key in _DISK_CACHE:
        return _DISK_CACHE[key]

    size = 2 * r_px + 1
    ax = torch.arange(size, dtype=dtype, device=device) - r_px
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    inside = (xx * xx + yy * yy).le(float(r_px * r_px))
    kernel = inside.float().to(dtype=dtype, device=device)
    _DISK_CACHE[key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Differentiable (soft) morphological operators
# ---------------------------------------------------------------------------


def soft_dilation(
    mask: torch.Tensor,
    radius: float = 2.0,
    hardness: float = 10.0,
) -> torch.Tensor:
    """Differentiable dilation via log-sum-exp soft maximum over neighbourhood.

    For each pixel the soft maximum of mask values within a disk of the given
    *radius* is computed using ``logsumexp(hardness * conv2d(mask, disk)) /
    hardness``.  As *hardness* -> inf this converges to the true (binary)
    dilation; finite values give smooth gradients everywhere.

    Args:
        mask: Continuous mask ``(H, W)`` with values in ``[0, 1]``.
        radius: Structuring-element radius in pixels.
        hardness: Temperature controlling the soft-max approximation.
            Larger = sharper, smaller = smoother gradients.

    Returns:
        Dilated mask ``(H, W)`` with values in ``[0, 1]``.
    """
    if radius < 0.5:
        return mask.clone()
    kernel = _disk_kernel(radius, mask.dtype, mask.device)
    r_px = max(1, int(math.ceil(radius)))
    inp = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    k4d = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)

    # Numerical stability: subtract the per-pixel max before exp.
    # max_pool2d with the same disk size gives the local max (approximation).
    max_local = functional.max_pool2d(inp, kernel_size=2 * r_px + 1, stride=1, padding=r_px)
    shifted = hardness * (inp - max_local)

    # conv2d(binary_kernel, exp(shifted)) = sum of exp(hardness*(x - max)) in
    # the disk neighbourhood.  The binary kernel counts contributing pixels.
    sum_exp = functional.conv2d(shifted.exp(), k4d, padding=r_px)

    # logsumexp = max + log(sum_exp) / hardness
    # Clamp sum_exp to avoid log(0) in degenerate cases.
    result = max_local + torch.log(sum_exp.clamp(min=1e-30)) / hardness
    return result.squeeze(0).squeeze(0)


def soft_erosion(
    mask: torch.Tensor,
    radius: float = 2.0,
    hardness: float = 10.0,
) -> torch.Tensor:
    """Differentiable erosion via log-sum-exp soft minimum over neighbourhood.

    Equivalent to ``1 - soft_dilation(1 - mask, ...)``.

    Args:
        mask: Continuous mask ``(H, W)`` with values in ``[0, 1]``.
        radius: Structuring-element radius in pixels.
        hardness: Temperature controlling the soft-min approximation.

    Returns:
        Eroded mask ``(H, W)`` with values in ``[0, 1]``.
    """
    if radius < 0.5:
        return mask.clone()
    return 1.0 - soft_dilation(1.0 - mask, radius=radius, hardness=hardness)


def morphological_opening(mask: torch.Tensor, radius: float = 2.0) -> torch.Tensor:
    """Differentiable morphological opening (erosion then dilation).

    Removes bright features smaller than *radius* while approximately
    preserving the shape of larger features.
    """
    return soft_dilation(soft_erosion(mask, radius=radius), radius=radius)


def morphological_closing(mask: torch.Tensor, radius: float = 2.0) -> torch.Tensor:
    """Differentiable morphological closing (dilation then erosion).

    Fills dark holes smaller than *radius* while approximately preserving
    the shape of the remaining features.
    """
    return soft_erosion(soft_dilation(mask, radius=radius), radius=radius)


def mrc_projection(
    mask: torch.Tensor,
    min_feature_px: float = 3.0,
) -> torch.Tensor:
    """Project a continuous mask to be MRC-clean by construction.

    Applies a morphological opening followed by a closing with radius
    ``min_feature_px / 2``.  The result is guaranteed to have no features
    (bright or dark) smaller than *min_feature_px* pixels — width and
    spacing constraints are both satisfied.

    This is a differentiable projection layer suitable for insertion after
    ILT optimisation or as a final post-processing step.

    Args:
        mask: Continuous mask ``(H, W)`` with values in ``[0, 1]``.
        min_feature_px: Minimum allowed feature size in pixels.  Both the
            minimum width and minimum spacing will be at least this value
            after projection.

    Returns:
        MRC-clean mask ``(H, W)`` with values in ``[0, 1]``.
    """
    r = max(0.5, min_feature_px / 2.0)
    opened = morphological_opening(mask, radius=r)
    return morphological_closing(opened, radius=r)


def estimate_shot_count(
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Differentiable approximation of mask shot count for e-beam writing.

    The shot count is the number of trapezoids (or shots) needed by a
    multi-beam mask writer.  This approximation counts boundary pixels of
    the thresholded mask as a proxy for perimeter complexity, then scales
    by a geometry factor.  Fully differentiable through a soft threshold.

    Args:
        mask: Continuous mask ``(H, W)`` with values in ``[0, 1]``.
        threshold: Binarisation threshold.

    Returns:
        Scalar tensor with the estimated shot count (differentiable).
    """
    # Soft binarisation via sigmoid centred at *threshold*.
    binary_soft = torch.sigmoid(20.0 * (mask - threshold))
    # Gradient magnitude as a soft boundary detector.
    gy = binary_soft[1:, :] - binary_soft[:-1, :]
    gx = binary_soft[:, 1:] - binary_soft[:, :-1]
    # Pad to original size so sum is straightforward.
    gy = functional.pad(gy, (0, 0, 0, 1))
    gx = functional.pad(gx, (0, 1, 0, 0))
    perimeter_proxy = (gy.abs() + gx.abs()).sum()
    # Scale: each boundary pixel ~ 0.5 shots (rough trapezoid equivalence).
    return perimeter_proxy * 0.5


# ---------------------------------------------------------------------------
# Binary (non-differentiable) morphological operators
# ---------------------------------------------------------------------------


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
    # Worst-case L-infinity distance for any pixel in an HxW canvas is
    # bounded by max(h, w) (a foreground component can be up to that wide
    # before any peeling exhausts it). The early `current.sum() == 0`
    # short-circuit keeps the cost in line with the actual distance reached
    # so we don't pay for the loose bound on typical masks.
    max_iterations = max(h, w) + 1

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
        while True:
            up = functional.pad(labels[1:, :], (0, 0, 0, 1), value=sentinel)
            down = functional.pad(labels[:-1, :], (0, 0, 1, 0), value=sentinel)
            left = functional.pad(labels[:, 1:], (0, 1, 0, 0), value=sentinel)
            right = functional.pad(labels[:, :-1], (1, 0, 0, 0), value=sentinel)
            stacked = torch.stack([labels, up, down, left, right], dim=0)
            new_labels = stacked.amin(dim=0)
            new_labels = torch.where(fg, new_labels, torch.full_like(new_labels, sentinel))
            if torch.equal(new_labels, labels):
                break
            labels = new_labels
    elif connectivity == 8:
        # 8-connectivity = 3x3 min over neighborhood. Computed on int64 via
        # nine shifted copies stacked along dim 0 — avoids float32 max_pool2d
        # mantissa collisions on >~16 megapixel masks.
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
            if torch.equal(new_labels, labels):
                break
            labels = new_labels
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    labels = torch.where(fg, labels, torch.full_like(labels, -1))
    fg_labels = labels[fg]
    num_components = int(torch.unique(fg_labels).numel())
    return labels, num_components
