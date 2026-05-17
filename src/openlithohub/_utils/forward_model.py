"""Simplified aerial image forward model using Gaussian PSF convolution."""

from __future__ import annotations

import math

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
    """
    if sigma_px < 1e-6:
        return mask.float() * dose

    kernel = _build_gaussian_kernel(sigma_px, mask.device)
    inp = mask.float().unsqueeze(0).unsqueeze(0)
    padding = kernel.shape[-1] // 2
    aerial = functional.conv2d(inp, kernel, padding=padding).squeeze(0).squeeze(0)
    return aerial * dose


def apply_resist_threshold(
    aerial_image: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Apply resist threshold to produce binary resist pattern."""
    return (aerial_image >= threshold).float()
