"""EUV 3D-mask shadow-effect proxy metric.

Real EUV mask 3D simulation (rigorous Maxwell) is expensive and lives in
commercial tools like HyperLith / EM-Suite. This module ships a cheap
proxy that captures the dominant first-order effect: **shadowing-induced
bias** that depends on feature orientation relative to the chief-ray
direction.

What we model
-------------

For an EUV reflective mask at a non-zero chief-ray angle of incidence
(typically 6° in NXE:3400-class scanners), the absorber casts a
geometric shadow whose magnitude depends on:

* absorber thickness (≈70 nm Ta-based, ≈30 nm low-n attenuated PSM);
* angle of incidence;
* feature orientation (horizontal vs vertical lines respond
  differently — the well-known H–V CD bias).

We compute a per-pixel **shadow displacement field** and convolve the
binary mask with an anisotropic shadow kernel, then compare the resulting
"3D-corrected" aerial against a thin-mask aerial. The L2 residual between
the two is a reasonable proxy for "how much rigorous 3D simulation would
disagree with the Hopkins thin-mask result on this layout".

This is a **proxy**, not a substitute, for rigorous 3D-mask EMF
simulation. Its purpose is to flag layouts that are at risk of large 3D
errors at evaluation time without paying the cost of a Maxwell solver.
For papers that require ground-truth 3D, hook a real simulator via
:class:`openlithohub.simulators.BaseSimulator`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from openlithohub._utils.tensor_ops import ensure_2d
from openlithohub.simulators import HopkinsSimulator, SimulatorConfig


@dataclass(frozen=True)
class Mask3DParams:
    """Parameters for the EUV 3D-mask shadow proxy.

    Attributes:
        absorber_thickness_nm: Absorber stack height. 70 nm = Ta-based,
            30 nm = low-n attenuated PSM.
        chief_ray_angle_deg: Chief-ray angle of incidence at the mask.
            6° for NXE:3400-class scanners.
        chief_ray_azimuth_deg: Azimuth of the chief ray (0° = +x). Sets
            the shadow direction.
        pixel_size_nm: Mask-side pixel pitch.
    """

    absorber_thickness_nm: float = 70.0
    chief_ray_angle_deg: float = 6.0
    chief_ray_azimuth_deg: float = 0.0
    pixel_size_nm: float = 1.0


def _shadow_kernel(params: Mask3DParams, device: torch.device) -> torch.Tensor:
    """Anisotropic line-segment kernel along the chief-ray azimuth.

    The shadow length is ``thickness * tan(angle)``. We discretise as a
    soft line segment: a 1-pixel-wide box of length
    ``ceil(shadow_px) + 1``, oriented along the azimuth, normalised to
    unit sum so it preserves dose.
    """

    shadow_nm = params.absorber_thickness_nm * math.tan(math.radians(params.chief_ray_angle_deg))
    shadow_px = max(shadow_nm / max(params.pixel_size_nm, 1e-6), 1e-6)
    radius = max(1, int(math.ceil(shadow_px)))
    size = 2 * radius + 1

    coords = torch.arange(size, dtype=torch.float32, device=device) - radius
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    az = math.radians(params.chief_ray_azimuth_deg)
    along = xx * math.cos(az) + yy * math.sin(az)
    perp = -xx * math.sin(az) + yy * math.cos(az)

    along_mask = (along >= 0) & (along <= shadow_px)
    perp_mask = perp.abs() <= 0.5
    kernel = (along_mask & perp_mask).to(torch.float32)
    if kernel.sum() == 0:
        kernel[radius, radius] = 1.0
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def apply_3d_shadow(
    mask: torch.Tensor,
    params: Mask3DParams | None = None,
) -> torch.Tensor:
    """Apply the 3D-shadow proxy operator to a binary mask.

    Args:
        mask: ``(H, W)`` real-valued mask in ``[0, 1]``.
        params: Shadow parameters; defaults to NXE:3400-like.

    Returns:
        Same-shape mask with the shadow operator applied. The result is
        no longer strictly binary — it represents the effective
        attenuation seen by the optical model.
    """

    p = params or Mask3DParams()
    m = ensure_2d(mask)
    kernel = _shadow_kernel(p, m.device)
    inp = m.unsqueeze(0).unsqueeze(0)
    radius = kernel.shape[-1] // 2
    padded = functional.pad(inp, [radius] * 4, mode="circular")
    shadowed = functional.conv2d(padded, kernel)
    return shadowed.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def compute_3d_mask_residual(
    mask: torch.Tensor,
    params: Mask3DParams | None = None,
    sim_config: SimulatorConfig | None = None,
) -> dict[str, float]:
    """Quantify expected disagreement between thin-mask and 3D-mask aerials.

    Runs the bundled Hopkins/SOCS simulator twice — once on the input
    mask (thin-mask assumption) and once on the shadow-corrected mask —
    and reports the L2 and L_inf residuals plus the H–V CD-bias proxy.

    Args:
        mask: ``(H, W)`` real mask.
        params: Shadow parameters.
        sim_config: Optional simulator config; defaults to EUV-ish
            (13.5 nm, NA 0.33).

    Returns:
        Dict with keys ``residual_l2``, ``residual_linf``, and
        ``hv_bias_nm`` (positive when horizontal lines print wider than
        vertical lines after 3D-shadow correction).
    """

    p = params or Mask3DParams()
    cfg = sim_config or SimulatorConfig(
        wavelength_nm=13.5,
        na=0.33,
        sigma=0.7,
        pixel_size_nm=p.pixel_size_nm,
    )
    sim = HopkinsSimulator(cfg)

    m = ensure_2d(mask).to(torch.float32)
    aerial_thin = sim.simulate(m).aerial
    aerial_3d = sim.simulate(apply_3d_shadow(m, p)).aerial

    diff = (aerial_3d - aerial_thin).detach()
    residual_l2 = float(diff.pow(2).mean().sqrt().item())
    residual_linf = float(diff.abs().max().item())

    horizontal_p = Mask3DParams(
        absorber_thickness_nm=p.absorber_thickness_nm,
        chief_ray_angle_deg=p.chief_ray_angle_deg,
        chief_ray_azimuth_deg=0.0,
        pixel_size_nm=p.pixel_size_nm,
    )
    vertical_p = Mask3DParams(
        absorber_thickness_nm=p.absorber_thickness_nm,
        chief_ray_angle_deg=p.chief_ray_angle_deg,
        chief_ray_azimuth_deg=90.0,
        pixel_size_nm=p.pixel_size_nm,
    )
    aerial_h = sim.simulate(apply_3d_shadow(m, horizontal_p)).aerial
    aerial_v = sim.simulate(apply_3d_shadow(m, vertical_p)).aerial
    hv_bias_nm = float((aerial_h - aerial_v).mean().item() * cfg.pixel_size_nm)

    return {
        "residual_l2": residual_l2,
        "residual_linf": residual_linf,
        "hv_bias_nm": hv_bias_nm,
    }
