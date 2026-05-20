"""SRAF non-printing penalty.

Sub-Resolution Assist Features should bias the diffraction pattern around main
features without ever clearing the resist threshold themselves. A printed SRAF
shows up on the wafer as a stray defect, which is a yield killer.

This module provides a differentiable penalty that callers add to their ILT or
OPC training loss. It is complementary to (not a substitute for) the curvilinear
MRC loss requested in issue #8 — that one polices mask geometry, this one
polices the *aerial-image* response inside SRAF regions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def sraf_print_penalty(
    aerial_image: torch.Tensor,
    sraf_mask: torch.Tensor,
    *,
    print_threshold: float = 0.30,
    margin: float = 0.05,
) -> torch.Tensor:
    """Differentiable penalty for SRAFs whose aerial intensity risks printing.

    For every pixel inside ``sraf_mask``, penalise the amount by which the
    aerial intensity exceeds ``print_threshold - margin``. Squared-ReLU keeps
    the gradient growing as the violation deepens, which empirically converges
    faster than plain L1 inside ILT inner loops.

    Args:
        aerial_image: Simulated aerial image. Either ``(H, W)`` or
            ``(B, 1, H, W)`` — the rank is mirrored from
            ``simulate_aerial_image``'s contract.
        sraf_mask: Binary tensor of the same shape as ``aerial_image``. ``1``
            indicates a pixel that belongs to an SRAF region (caller-provided —
            SRAF placement / detection is handled upstream, see issue #6).
        print_threshold: Resist-clearing threshold. Defaults to ``0.30``,
            comfortably below the nominal ``0.50`` so SRAFs are punished
            *before* they become bright enough to actually develop.
        margin: Safety headroom subtracted from ``print_threshold`` before the
            comparison — encourages the optimiser to hold SRAFs below the
            danger zone with a buffer.

    Returns:
        Scalar ``torch.Tensor`` (autograd-connected). ``0`` when no SRAF pixel
        exceeds the budget; positive otherwise.
    """
    if aerial_image.shape != sraf_mask.shape:
        raise ValueError(
            f"aerial_image and sraf_mask must share shape; got "
            f"{tuple(aerial_image.shape)} vs {tuple(sraf_mask.shape)}"
        )

    sraf_float = sraf_mask.to(aerial_image.dtype)
    sraf_pixel_count = sraf_float.sum()
    if float(sraf_pixel_count.detach()) < 1.0:
        return aerial_image.new_zeros(())

    budget = print_threshold - margin
    excess = functional.relu(aerial_image - budget)
    weighted = (excess.pow(2) * sraf_float).sum()
    return weighted / sraf_pixel_count.clamp(min=1.0)
