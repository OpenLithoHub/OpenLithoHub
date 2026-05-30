"""Process-window-aware OPC ("PW-OPC") workflow.

Production OPC must hold up across a dose/focus *process window*, not just at
nominal exposure. Optimising against the nominal corner alone tends to produce
masks that look great in the lab and fail in the fab.

This module supplies the corner-sweep machinery: define a small set of dose +
defocus corners, run the existing forward model at each, and aggregate the
fidelity loss as a weighted mean. It is a drop-in replacement for the nominal
``F.mse_loss(resist, target)`` line that today lives inside every ILT inner
loop — see ``models.levelset_ilt.LevelSetILTModel`` for the canonical
integration site.

The corner enumeration mirrors the four-corner scheme used by
``benchmark.metrics.pvband.compute_pvband`` so that what we *optimise* and what
we *measure* live in the same world.

Physical caveats (issue #27)
----------------------------

The corner sweep here is a fast inner-loop diagnostic, not rigorous PW
simulation:

* **Defocus is modelled by widening the Gaussian PSF only.** A real
  defocused pupil loses contrast (the MTF dips and re-rings — the
  textbook Bossung curves) — the energy is *redistributed* into the
  pupil's nulls, not merely smeared by a wider real-space kernel. A
  Gaussian preserves total energy, so this proxy under-estimates
  defocus-induced contrast loss; PW corners that are dim in reality
  look only blurry here. For headline PW numbers, drive the Hopkins
  path with measured-source / Zernike-pupil I/O (see ``optics.py``)
  rather than this proxy.
* **Dose enters as a multiplicative scale on the aerial image.** That
  is correct *if* the resist threshold is dose-pinned (the intensity
  the resist clears at scales linearly with dose). The
  ``HopkinsSimulator`` does this internally (issue #52), but this fast
  path uses a *fixed* threshold passed by the caller — so a ±5% dose
  corner cleanly shifts the resist contour rather than being silently
  cancelled. Pair it with ``threshold=0.225`` (LithoBench-canonical)
  for headline alignment with the Hopkins benchmark numbers.

If you require physically-rigorous defocus, use a Hopkins SOCS forward
sim with a defocus Zernike (Z4) and treat this module as the
training-loop proxy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.resist_model import apply_differentiable_resist


@dataclass(frozen=True)
class ProcessWindowCorner:
    """One dose/focus corner in the optimisation sweep.

    ``sigma_px`` is the Gaussian-PSF width in pixels — defocus translates into
    sigma upstream of this dataclass (callers convert nm → px), keeping this
    structure agnostic to physical units.
    """

    dose: float
    sigma_px: float
    weight: float = 1.0


DEFAULT_PW_CORNERS: tuple[ProcessWindowCorner, ...] = (
    ProcessWindowCorner(dose=1.00, sigma_px=2.0, weight=2.0),
    ProcessWindowCorner(dose=1.05, sigma_px=2.5, weight=1.0),
    ProcessWindowCorner(dose=0.95, sigma_px=2.5, weight=1.0),
    ProcessWindowCorner(dose=1.05, sigma_px=1.5, weight=1.0),
    ProcessWindowCorner(dose=0.95, sigma_px=1.5, weight=1.0),
)


def pw_aerial_images(
    mask: torch.Tensor,
    corners: Sequence[ProcessWindowCorner] = DEFAULT_PW_CORNERS,
) -> list[torch.Tensor]:
    """Simulate the aerial image at every corner.

    Returned tensors share rank with ``mask`` (``(H,W)`` in, ``(H,W)`` out).
    Autograd-connected — gradients flow back to ``mask``.
    """
    return [simulate_aerial_image(mask, sigma_px=c.sigma_px, dose=c.dose) for c in corners]


def pw_fidelity_loss(
    mask: torch.Tensor,
    target: torch.Tensor,
    *,
    corners: Sequence[ProcessWindowCorner] = DEFAULT_PW_CORNERS,
    threshold: float = 0.5,
    steepness: float = 50.0,
    resist_diffusion_nm: float = 0.0,
    pixel_size_nm: float = 1.0,
    quencher: float = 0.0,
) -> torch.Tensor:
    """Weighted-mean MSE between simulated resist and target across PW corners.

    Each corner contributes ``weight * MSE(resist_corner, target)``; the result
    is divided by the sum of weights so the scalar magnitude stays comparable
    to the nominal-only baseline.

    With ``corners=(ProcessWindowCorner(dose=1.0, sigma_px=σ, weight=1.0),)``
    this reduces to the existing nominal-only loss, which is how the call-site
    in ``LevelSetILTModel`` keeps backward compatibility.

    .. note::
       Threshold default is 0.5 (legacy API — changing it would silently
       shift every training trajectory built on this module). Pass
       ``threshold=0.225`` to align with the LithoBench / Yang2023
       resist-clearing convention used by ``compute_l2_error`` /
       ``compute_wafer_epe``.
    """
    if len(corners) == 0:
        raise ValueError("pw_fidelity_loss requires at least one corner")

    total = mask.new_zeros(())
    weight_sum = 0.0
    for corner in corners:
        aerial = simulate_aerial_image(mask, sigma_px=corner.sigma_px, dose=corner.dose)
        resist = apply_differentiable_resist(
            aerial, threshold=threshold, steepness=steepness,
            resist_diffusion_nm=resist_diffusion_nm,
            pixel_size_nm=pixel_size_nm, quencher=quencher,
        )
        total = total + corner.weight * functional.mse_loss(resist, target)
        weight_sum += corner.weight

    if weight_sum <= 0.0:
        raise ValueError("pw_fidelity_loss corner weights must sum to a positive value")

    return total / weight_sum
