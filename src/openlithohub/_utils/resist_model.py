"""Chemically-amplified resist simulation with acid diffusion."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from openlithohub._utils.forward_model import _build_gaussian_kernel


def simulate_resist(
    aerial_image: torch.Tensor,
    acid_diffusion_length_nm: float = 5.0,
    pixel_size_nm: float = 1.0,
    threshold: float = 0.5,
    quencher_concentration: float = 0.1,
) -> torch.Tensor:
    """Simulate chemically-amplified resist response with acid diffusion.

    Models a physically-motivated resist development process:
    1. Photoacid generation proportional to aerial image intensity
    2. Acid diffusion via Gaussian blur (diffusion length determines spread)
    3. Quencher neutralization (constant subtraction)
    4. Threshold to binary resist pattern

    Args:
        aerial_image: Aerial image intensity (H, W), values in [0, 1].
        acid_diffusion_length_nm: Acid diffusion length in nanometers.
        pixel_size_nm: Physical pixel size for unit conversion.
        threshold: Development threshold for binary output.
        quencher_concentration: Base quencher level subtracted from acid.

    Returns:
        Binary resist pattern (H, W), 1.0 where resist remains.
    """
    acid = aerial_image.clone()

    sigma_diffusion_px = acid_diffusion_length_nm / max(pixel_size_nm, 1e-6)
    if sigma_diffusion_px > 0.1:
        kernel = _build_gaussian_kernel(sigma_diffusion_px, acid.device)
        inp = acid.unsqueeze(0).unsqueeze(0)
        padding = kernel.shape[-1] // 2
        acid = functional.conv2d(inp, kernel, padding=padding).squeeze(0).squeeze(0)

    acid = (acid - quencher_concentration).clamp(min=0.0)
    return (acid >= threshold).float()


def simulate_resist_soft(
    aerial_image: torch.Tensor,
    acid_diffusion_length_nm: float = 5.0,
    pixel_size_nm: float = 1.0,
    threshold: float = 0.5,
    quencher_concentration: float = 0.1,
    steepness: float = 50.0,
) -> torch.Tensor:
    """Differentiable resist simulation using sigmoid instead of hard threshold.

    Same physics as `simulate_resist` but uses a smooth sigmoid for the
    development step, making it suitable for gradient-based optimization.
    """
    acid = aerial_image.clone()

    sigma_diffusion_px = acid_diffusion_length_nm / max(pixel_size_nm, 1e-6)
    if sigma_diffusion_px > 0.1:
        kernel = _build_gaussian_kernel(sigma_diffusion_px, acid.device)
        inp = acid.unsqueeze(0).unsqueeze(0)
        padding = kernel.shape[-1] // 2
        acid = functional.conv2d(inp, kernel, padding=padding).squeeze(0).squeeze(0)

    acid = (acid - quencher_concentration).clamp(min=0.0)
    return torch.sigmoid(steepness * (acid - threshold))
