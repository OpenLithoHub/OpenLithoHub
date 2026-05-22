"""Data transforms for resolution alignment and normalization."""

from __future__ import annotations

import torch
import torch.nn.functional as f


def align_resolution(
    tensor: torch.Tensor,
    source_pixel_nm: float,
    target_pixel_nm: float,
    mode: str = "bilinear",
    *,
    binarize: bool = False,
    binarize_threshold: float = 0.5,
) -> torch.Tensor:
    """Resample a tensor to match target pixel resolution.

    Args:
        tensor: Input tensor (H, W), (C, H, W), or (N, C, H, W).
        source_pixel_nm: Current pixel size in nanometers.
        target_pixel_nm: Desired pixel size in nanometers.
        mode: Interpolation mode ('bilinear', 'nearest', 'bicubic').
        binarize: If True, threshold the resampled output back to {0, 1}
            with ``> binarize_threshold``. Use this when the input is a
            binary mask: bilinear / bicubic interpolation produces
            grayscale fringes along edges that downstream raster ops
            (DRC, MRC, contour trace) treat as foreground, inflating
            metrics by a few pixel widths. Skip when the input is a
            continuous field (aerial intensity, density map).
        binarize_threshold: Cutoff used when ``binarize=True``.

    Returns:
        Resampled tensor at the target resolution; ndim matches input.

    Notes:
        Output spatial dimensions are computed as
        ``round(H * source / target)`` and passed to ``F.interpolate``
        via ``size=``. The earlier ``scale_factor=`` form left the exact
        output size to the framework's rounding policy, which differs
        between PyTorch versions and between modes — explicit ``size``
        keeps a (1024, 1024) layout aligning to a (2048, 2048) grid at
        2× upsample regardless of build.
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
    elif ndim == 4:
        x = tensor
    else:
        raise ValueError(f"Expected 2D (H,W), 3D (C,H,W), or 4D (N,C,H,W) tensor, got {ndim}D")

    h_in, w_in = int(x.shape[-2]), int(x.shape[-1])
    h_out = max(1, int(round(h_in * scale)))
    w_out = max(1, int(round(w_in * scale)))

    align_corners = None if mode == "nearest" else False
    x = f.interpolate(x, size=(h_out, w_out), mode=mode, align_corners=align_corners)

    if binarize:
        x = (x > binarize_threshold).to(x.dtype)

    if ndim == 2:
        return x.squeeze(0).squeeze(0)
    if ndim == 3:
        return x.squeeze(0)
    return x


def normalize_to_binary(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Threshold a continuous tensor to binary (0/1)."""
    return (tensor > threshold).float()
