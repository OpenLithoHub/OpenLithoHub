"""Process Variation Band (PV Band) computation."""

from __future__ import annotations

import torch

from openlithohub._utils.forward_model import apply_resist_threshold, simulate_aerial_image
from openlithohub._utils.morphology import distance_transform
from openlithohub._utils.tensor_ops import ensure_2d


def compute_pvband(
    mask: torch.Tensor,
    nominal_dose: float = 1.0,
    dose_variation: float = 0.05,
    defocus_range_nm: float = 20.0,
    pixel_size_nm: float = 1.0,
) -> dict[str, float]:
    """Compute Process Variation Band width for a given mask.

    PV Band measures the perpendicular distance between the resist contours
    at process window extremes. Uses a simplified Gaussian forward model to
    simulate aerial images at four dose/focus corners, then reports the
    band's local thickness — twice the distance from each band-interior pixel
    to the nearest non-band pixel (i.e. either the outer or inner contour).

    The factor of two converts "distance to the nearest contour" (half-width
    at the band's centerline) into the full perpendicular contour-to-contour
    distance that the literature publishes. Without it the metric under-reports
    by 2× and cannot be compared with other papers' PV-band numbers.
    """
    m = ensure_2d(mask)
    binary = (m > 0.5).float()

    sigma_nominal = 2.0
    sigma_defocus = defocus_range_nm / (2.0 * pixel_size_nm)

    dose_high = nominal_dose * (1.0 + dose_variation)
    dose_low = nominal_dose * (1.0 - dose_variation)
    sigma_high = sigma_nominal + sigma_defocus
    sigma_low = max(0.5, sigma_nominal - sigma_defocus * 0.5)

    corners = [
        (dose_high, sigma_high),
        (dose_high, sigma_low),
        (dose_low, sigma_high),
        (dose_low, sigma_low),
    ]

    outer_envelope = torch.zeros_like(binary)
    inner_envelope = torch.ones_like(binary)

    for dose, sigma in corners:
        aerial = simulate_aerial_image(binary, sigma_px=sigma, dose=dose)
        resist = apply_resist_threshold(aerial, threshold=0.5)
        outer_envelope = torch.maximum(outer_envelope, resist)
        inner_envelope = torch.minimum(inner_envelope, resist)

    band = (outer_envelope - inner_envelope).clamp(min=0.0)
    band_pixels = band.sum().item()
    if band_pixels < 1.0:
        return {"pvband_mean_nm": 0.0, "pvband_max_nm": 0.0}

    band_binary = (band > 0.5).float()
    dist_map = distance_transform(band_binary)

    band_mask = band_binary > 0.5
    if not band_mask.any():
        return {"pvband_mean_nm": 0.0, "pvband_max_nm": 0.0}

    distances = dist_map[band_mask] * pixel_size_nm
    pvband_mean = float(distances.mean().item()) * 2.0
    pvband_max = float(distances.max().item()) * 2.0
    return {"pvband_mean_nm": pvband_mean, "pvband_max_nm": pvband_max}
