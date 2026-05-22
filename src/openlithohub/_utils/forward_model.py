"""Simplified aerial image forward model using Gaussian PSF convolution.

Padding contract
----------------
All convolutions in this module MUST use circular (periodic) padding, not
zero-padding. This is not stylistic — the Hopkins source-mask formulation
treats the mask as a tile in a periodic illumination, and zero-padding the
input introduces spurious dim-aerial fringes near the frame edge that
silently degrade EPE / PV-band metrics on layouts with features close to
the boundary. If you add a new conv-based simulator here, route it through
``_circular_pad_clamped`` (or an equivalent ``mode="circular"`` call) — do
not switch to ``F.conv2d``'s default zero-pad as a "simplification".
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as functional


def _build_gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32, device=device) - radius
    g1d = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
    kernel = g1d.unsqueeze(1) * g1d.unsqueeze(0)
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def simulate_aerial_image(
    mask: torch.Tensor,
    sigma_px: float,
    dose: float = 1.0,
) -> torch.Tensor:
    """Simulate aerial image via Gaussian PSF convolution.

    Approximates Hopkins diffraction with a single Gaussian point spread function.

    Accepts ``(H, W)`` for single-image use and ``(B, 1, H, W)`` for batched
    forward passes — the output preserves the input rank.

    Uses circular (periodic) padding to match the Hopkins forward model's
    convention. OPC treats the mask as a tile of an infinite layout, so
    zero-padding at the border would introduce spurious dim-aerial fringes
    that the Hopkins path does not have.
    """
    if sigma_px < 1e-6:
        return mask.float() * dose

    kernel = _build_gaussian_kernel(sigma_px, mask.device)
    padding = kernel.shape[-1] // 2

    squeezed = False
    if mask.ndim == 2:
        inp = mask.float().unsqueeze(0).unsqueeze(0)
        squeezed = True
    elif mask.ndim == 4 and mask.shape[1] == 1:
        inp = mask.float()
    else:
        raise ValueError(f"Expected mask shape (H,W) or (B,1,H,W); got {tuple(mask.shape)}")

    inp_padded = _circular_pad_clamped(inp, padding)
    aerial = functional.conv2d(inp_padded, kernel)
    if squeezed:
        aerial = aerial.squeeze(0).squeeze(0)
    return aerial * dose


def _circular_pad_clamped(inp: torch.Tensor, padding: int) -> torch.Tensor:
    """Circular pad an (N, C, H, W) tensor by ``padding`` on every side.

    PyTorch's circular pad refuses pad sizes >= the corresponding image
    dimension. For very small masks (typical of unit tests) we tile the
    padding in steps that respect that constraint.
    """
    if padding == 0:
        return inp
    out = inp
    remaining_h = padding
    remaining_w = padding
    while remaining_h > 0 or remaining_w > 0:
        cur_h = out.shape[-2]
        cur_w = out.shape[-1]
        step_h = min(remaining_h, cur_h - 1) if remaining_h > 0 else 0
        step_w = min(remaining_w, cur_w - 1) if remaining_w > 0 else 0
        if step_h == 0 and step_w == 0:
            # Image is 1 px wide/tall in some axis — circular pad cannot extend
            # it (PyTorch refuses pad sizes >= dim). Fall back to replicate so
            # the unit-test path still runs, but warn so production callers are
            # not silently downgraded from circular padding (the Hopkins forward
            # model's contract — see module docstring).
            step_h = remaining_h if cur_h == 1 else 0
            step_w = remaining_w if cur_w == 1 else 0
            warnings.warn(
                "_circular_pad_clamped: input has a 1-pixel-wide axis; falling "
                "back to replicate padding. This violates the module's circular-"
                "padding contract and will introduce edge artifacts on a real "
                "layout. See forward_model.py module docstring.",
                RuntimeWarning,
                stacklevel=2,
            )
            out = functional.pad(out, (step_w, step_w, step_h, step_h), mode="replicate")
            remaining_h -= step_h
            remaining_w -= step_w
            continue
        out = functional.pad(out, (step_w, step_w, step_h, step_h), mode="circular")
        remaining_h -= step_h
        remaining_w -= step_w
    return out


def apply_resist_threshold(
    aerial_image: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Apply resist threshold to produce binary resist pattern.

    The 0.5 default is a generic mid-intensity cutoff for ad-hoc use; the
    canonical ICCAD16 / LithoBench cutoff is 0.225 (see
    [Yang2023_LithoBench, §3.2, p.5] and ``SimulatorConfig.threshold``).
    Pass ``threshold=0.225`` when reproducing benchmark numbers.

    This is **constant threshold resist (CTR) without diffusion** — the
    sigmoid-on-aerial simplification documented in
    ``docs/architecture.md → Resist Model Simplification``. Real per-node
    CTR parameters are foundry-confidential and cannot ship in an
    open-source repo; benchmark-relative comparison is unaffected, but
    absolute wafer prediction is not in scope.
    """
    return (aerial_image >= threshold).float()
