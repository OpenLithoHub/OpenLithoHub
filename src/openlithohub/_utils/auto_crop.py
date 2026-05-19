"""Auto-Crop: locate the most visually complex region of a large mask.

Used by the HF Playground to accept arbitrarily large uploads (full-chip
OASIS rasterizations, multi-megapixel PNGs) without OOM-ing the free
container's distance-transform path. The crop window is chosen by an
edge-density + foreground-density score so the user sees their dense
routing area, not blank silicon.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .morphology import binary_dilation, binary_erosion

BBox = tuple[int, int, int, int]  # (y0, x0, y1, x1), exclusive on the high end


def _edge_map(mask: torch.Tensor) -> torch.Tensor:
    """Cheap binary edge approximation: dilation XOR erosion."""
    return (binary_dilation(mask, radius=1) - binary_erosion(mask, radius=1)).clamp_(0.0, 1.0)


def score_complexity(mask: torch.Tensor, *, window_px: int) -> torch.Tensor:
    """Per-pixel complexity score via boxcar mean of edges + foreground.

    Output is the same shape as ``mask``. Higher = denser / busier.
    """
    fg = (mask > 0.5).float()
    edges = _edge_map(fg)
    # 0.7 weight on edges, 0.3 on raw foreground — edges dominate so SRAM-style
    # repeating arrays score above empty-with-one-blob regions.
    signal = 0.7 * edges + 0.3 * fg
    sig4d = signal.unsqueeze(0).unsqueeze(0)
    # Force an odd kernel so conv2d output matches the input shape exactly.
    k = window_px if window_px % 2 == 1 else window_px + 1
    kernel = torch.ones((1, 1, k, k), dtype=sig4d.dtype, device=sig4d.device)
    boxcar = functional.conv2d(sig4d, kernel, padding=k // 2) / float(k * k)
    return boxcar.squeeze(0).squeeze(0)


def find_most_complex_window(mask: torch.Tensor, *, window_size: int) -> BBox:
    """Return the bbox (y0, x0, y1, x1) of the densest ``window_size`` square.

    The window is clamped to fit inside the mask. For masks smaller than
    ``window_size`` on either axis, returns the full extent.
    """
    h, w = mask.shape
    if h <= window_size and w <= window_size:
        return (0, 0, h, w)

    score = score_complexity(mask, window_px=min(window_size // 4, max(h, w) // 8) or 1)
    # Restrict the argmax search to centres that keep a full window in-bounds.
    half = window_size // 2
    y_lo, y_hi = half, max(half, h - (window_size - half))
    x_lo, x_hi = half, max(half, w - (window_size - half))
    interior = score[y_lo:y_hi, x_lo:x_hi]
    flat_idx = int(torch.argmax(interior).item())
    cy = y_lo + flat_idx // interior.shape[1]
    cx = x_lo + flat_idx % interior.shape[1]
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(h, y0 + window_size)
    x1 = min(w, x0 + window_size)
    y0 = max(0, y1 - window_size)
    x0 = max(0, x1 - window_size)
    return (y0, x0, y1, x1)


def auto_crop(mask: torch.Tensor, *, target_size: int) -> tuple[torch.Tensor, BBox]:
    """Crop ``mask`` to a ``target_size`` square at the densest location.

    No-op (returns the original tensor + a full-extent bbox) when the input
    is already within budget on both axes.
    """
    h, w = mask.shape
    if h <= target_size and w <= target_size:
        return mask, (0, 0, h, w)
    y0, x0, y1, x1 = find_most_complex_window(mask, window_size=target_size)
    return mask[y0:y1, x0:x1], (y0, x0, y1, x1)
