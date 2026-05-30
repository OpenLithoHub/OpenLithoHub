"""Process Variation Band (PV Band) computation.

Two forward-model paths are available:

1. **Default — fast Gaussian-PSF aerial-image approximation** at four
   dose/focus corners. Cheap diagnostic that runs in inner loops and on
   every commit; this is what the baseline tables in
   ``baselines/results.md`` and the README report. The Gaussian model
   is calibrated so the absolute PV Band number tracks the SOCS result
   at the published Neural-ILT corners — both are stable signals of
   process-window robustness, but they are not interchangeable
   numerically.

2. **SOCS-faithful — `simulator=` keyword** (added 2026-05-23). When a
   :class:`BaseSimulator` instance is passed, this metric drives the
   simulator at each ``(dose, defocus)`` corner via
   :meth:`BaseSimulator.with_config`, takes the binarised resist
   contour at each corner, and reports outer-vs-inner band thickness
   from the same kernels :func:`compute_l2_error` uses. This closes
   the "Gaussian PVB ≠ SOCS PVB" reproducibility footgun for paper
   authors comparing OPC numbers across implementations: when you
   need PVB derived from the same SOCS kernels as L2/EPE, pass the
   same configured simulator instance.

   This path is opt-in to keep existing baseline numbers stable —
   passing ``simulator=`` *will* change the absolute number reported.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from openlithohub._utils.forward_model import apply_resist_threshold, simulate_aerial_image
from openlithohub._utils.morphology import distance_transform
from openlithohub._utils.tensor_ops import ensure_2d

if TYPE_CHECKING:
    from openlithohub.simulators.base import BaseSimulator


def compute_pvband(
    mask: torch.Tensor,
    nominal_dose: float = 1.0,
    dose_variation: float = 0.05,
    defocus_range_nm: float = 20.0,
    pixel_size_nm: float = 1.0,
    simulator: BaseSimulator | None = None,
    resist_diffusion_nm: float = 0.0,
    quencher: float = 0.0,
) -> dict[str, float]:
    """Compute Process Variation Band width for a given mask.

    PV Band measures the perpendicular distance between the resist
    contours at process window extremes.

    With ``simulator=None`` (default) the cheap Gaussian-PSF approximation
    is used — see module docstring path (1).

    With ``simulator=<BaseSimulator instance>`` the simulator is driven
    at four ``(dose × defocus)`` corners via ``with_config``, and the
    band is computed from the same kernels — see module docstring path
    (2). The simulator's existing ``defocus_nm`` is used as the
    nominal centre; ``±defocus_range_nm/2`` is applied at the corners.

    The factor of two converts "distance to the nearest contour"
    (half-width at the band's centerline) into the full perpendicular
    contour-to-contour distance that the literature publishes.
    """
    m = ensure_2d(mask)
    binary = (m > 0.5).float()

    if simulator is None:
        outer_envelope, inner_envelope = _gaussian_pw_envelopes(
            binary,
            nominal_dose,
            dose_variation,
            defocus_range_nm,
            pixel_size_nm,
            resist_diffusion_nm=resist_diffusion_nm,
            quencher=quencher,
        )
    else:
        outer_envelope, inner_envelope = _simulator_pw_envelopes(
            binary,
            simulator,
            nominal_dose,
            dose_variation,
            defocus_range_nm,
            resist_diffusion_nm=resist_diffusion_nm,
            quencher=quencher,
        )

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


def _gaussian_pw_envelopes(
    binary: torch.Tensor,
    nominal_dose: float,
    dose_variation: float,
    defocus_range_nm: float,
    pixel_size_nm: float,
    resist_diffusion_nm: float = 0.0,
    quencher: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        resist = apply_resist_threshold(
            aerial,
            threshold=0.5,
            resist_diffusion_nm=resist_diffusion_nm,
            pixel_size_nm=pixel_size_nm,
            quencher=quencher,
        )
        outer_envelope = torch.maximum(outer_envelope, resist)
        inner_envelope = torch.minimum(inner_envelope, resist)
    return outer_envelope, inner_envelope


def _simulator_pw_envelopes(
    binary: torch.Tensor,
    simulator: BaseSimulator,
    nominal_dose: float,
    dose_variation: float,
    defocus_range_nm: float,
    resist_diffusion_nm: float = 0.0,
    quencher: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drive ``simulator`` at four PW corners and aggregate resist contours.

    Each corner gets a sibling simulator via ``with_config`` so cached
    SOCS kernels are reused along the dose axis (kernels depend on
    defocus, not on dose — see ``HopkinsSimulator._hparams_match``). The
    binarised resist contour at each corner is OR-ed into the outer
    envelope and AND-ed into the inner envelope; the band is their
    difference.
    """
    base_cfg = simulator.config
    base_dose = base_cfg.dose
    base_defocus = base_cfg.defocus_nm
    half_range = defocus_range_nm * 0.5

    dose_high = base_dose * nominal_dose * (1.0 + dose_variation)
    dose_low = base_dose * nominal_dose * (1.0 - dose_variation)
    defocus_high = base_defocus + half_range
    defocus_low = base_defocus - half_range

    corners = [
        (dose_high, defocus_high),
        (dose_high, defocus_low),
        (dose_low, defocus_high),
        (dose_low, defocus_low),
    ]

    outer_envelope = torch.zeros_like(binary)
    inner_envelope = torch.ones_like(binary)
    for dose, defocus_nm in corners:
        corner_cfg = replace(base_cfg, dose=dose, defocus_nm=defocus_nm)
        corner_sim = simulator.with_config(corner_cfg)
        result = corner_sim.simulate(binary)
        if result.resist is not None:
            resist = result.resist.to(binary.dtype)
        else:
            threshold = corner_cfg.threshold
            resist = apply_resist_threshold(
                result.aerial,
                threshold=threshold,
                resist_diffusion_nm=resist_diffusion_nm,
                pixel_size_nm=simulator.config.pixel_size_nm,
                quencher=quencher,
            ).to(binary.dtype)
        outer_envelope = torch.maximum(outer_envelope, resist)
        inner_envelope = torch.minimum(inner_envelope, resist)
    return outer_envelope, inner_envelope
