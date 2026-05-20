"""Differentiable Mask Rule Check (MRC) loss for curvilinear masks.

Companion to ``benchmark.compliance.mrc.check_curvilinear_mrc`` (post-hoc
binary verdict) and ``sraf.sraf_print_penalty`` (aerial-image-side penalty).
This module gives optimisers a smooth, differentiable signal so curvilinear
ILT / level-set / Neural-ILT models can learn to respect MRC during training
instead of being scored on it afterwards.

Drop into a training loop:

    loss = epe_loss + alpha * curvilinear_mrc_loss(mask, pdk="asap7")

See issue #8 for motivation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from openlithohub.synth.pdk import PdkRules, get_pdk

__all__ = ["curvilinear_mrc_loss"]


def _soft_erosion(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Differentiable erosion via min-pool (= -max_pool of the negated input).

    Equivalent to ``binary_erosion`` from ``_utils.morphology`` on a binarised
    input, but propagates gradients through continuous mask values. Operates
    on ``(B, 1, H, W)`` for vectorised pooling.

    Uses replicate padding so features touching the canvas edge are not
    artificially eroded by zero-padding — keeps the loss honest at the
    boundary, where naive zero-pad would treat the off-canvas region as
    background and flag every edge-adjacent feature as a width violation.
    """
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    padded = functional.pad(mask, (radius, radius, radius, radius), mode="replicate")
    return -functional.max_pool2d(-padded, kernel_size=kernel, stride=1, padding=0)


def _soft_dilation(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Differentiable dilation via max-pool. ``(B, 1, H, W)`` in / out.

    Replicate-padded to match :func:`_soft_erosion` — see that docstring.
    """
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    padded = functional.pad(mask, (radius, radius, radius, radius), mode="replicate")
    return functional.max_pool2d(padded, kernel_size=kernel, stride=1, padding=0)


def _opening_residual(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """``relu(mask - opening(mask, radius))`` — pixels that fail the width opening.

    Morphological opening keeps only features wide enough to host the
    structuring element. Anything ``mask`` claims that the opening drops
    is a width violation, in proportion to how confidently the mask claimed
    it. ReLU clamps the (rare) numerical-noise negatives.
    """
    opened = _soft_dilation(_soft_erosion(mask, radius), radius)
    return functional.relu(mask - opened)


def _resolve_rules(
    pdk: PdkRules | str | None,
    min_width_nm: float | None,
    min_spacing_nm: float | None,
    min_curvature_radius_nm: float,
    pixel_size_nm: float | None,
) -> tuple[float, float, float, float]:
    """Resolve the active rule values from ``pdk`` + per-rule overrides.

    Explicit kwargs win over ``pdk`` so callers can sweep a single rule
    without forking the preset. Returns
    ``(min_width_nm, min_spacing_nm, min_curvature_radius_nm, pixel_size_nm)``.
    """
    if isinstance(pdk, str):
        pdk = get_pdk(pdk)

    if pdk is None:
        if min_width_nm is None or min_spacing_nm is None or pixel_size_nm is None:
            raise ValueError(
                "curvilinear_mrc_loss requires either `pdk` or explicit "
                "(min_width_nm, min_spacing_nm, pixel_size_nm). Got pdk=None "
                f"with min_width_nm={min_width_nm!r}, "
                f"min_spacing_nm={min_spacing_nm!r}, "
                f"pixel_size_nm={pixel_size_nm!r}."
            )
        return (
            float(min_width_nm),
            float(min_spacing_nm),
            float(min_curvature_radius_nm),
            float(pixel_size_nm),
        )

    return (
        float(min_width_nm if min_width_nm is not None else pdk.min_width_nm),
        float(min_spacing_nm if min_spacing_nm is not None else pdk.min_spacing_nm),
        float(min_curvature_radius_nm),
        float(pixel_size_nm if pixel_size_nm is not None else pdk.pixel_size_nm),
    )


def _ensure_b1hw(mask: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Coerce ``(H, W)`` or ``(B, 1, H, W)`` into ``(B, 1, H, W)``.

    Returns the 4D view and a flag telling the caller it added the batch
    axis (currently unused — kept so future callers can unwrap if needed).
    """
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0), True
    if mask.dim() == 4 and mask.shape[1] == 1:
        return mask, False
    raise ValueError(
        f"curvilinear_mrc_loss expects mask shape (H, W) or (B, 1, H, W); got {tuple(mask.shape)}."
    )


def curvilinear_mrc_loss(
    mask: torch.Tensor,
    pdk: PdkRules | str | None = None,
    *,
    min_width_nm: float | None = None,
    min_spacing_nm: float | None = None,
    min_curvature_radius_nm: float = 20.0,
    pixel_size_nm: float | None = None,
    weight_min_cd: float = 1.0,
    weight_min_spacing: float = 1.0,
    weight_min_curvature: float = 1.0,
) -> torch.Tensor:
    """Differentiable MRC penalty for curvilinear masks.

    Three additive terms, each non-negative and zero on a clean mask:

    * **Min-CD** — soft morphological opening with structuring radius
      ``r = floor(min_width_nm / (2 * pixel_size_nm))``. Pixels the mask
      claims that the opening drops contribute ``relu(mask - opening)``,
      summed and normalised by area. This mirrors the binary check in
      ``compliance.mrc.check_mrc`` so the loss and the verdict agree on
      what a violation is.
    * **Min-spacing** — same opening applied to ``1 - mask``; gaps too
      narrow to host the structuring element get penalised.
    * **Min-curvature** — boundary-band integral of the squared image
      gradient. ``‖∇mask‖²`` peaks at sharp transitions, so any region
      where the local gradient magnitude exceeds the curvature budget
      ``1 / min_curvature_radius_nm`` (in per-nm units) is squared-ReLU
      penalised. The "boundary band" is the symmetric difference
      ``dilation(mask, 1) - erosion(mask, 1)``, restricting the cost to
      pixels actually on a contour and keeping the loss well-defined for
      large flat interior regions.

    Args:
        mask: Continuous mask in ``[0, 1]``. Either ``(H, W)`` or
            ``(B, 1, H, W)``.
        pdk: PDK rules to source defaults from. May be a ``PdkRules`` instance
            or a preset name (e.g. ``"asap7"``, ``"freepdk45"``). If ``None``,
            ``min_width_nm``, ``min_spacing_nm``, and ``pixel_size_nm`` must
            all be supplied explicitly.
        min_width_nm: Override for ``pdk.min_width_nm``.
        min_spacing_nm: Override for ``pdk.min_spacing_nm``.
        min_curvature_radius_nm: Minimum allowed local radius of curvature.
            Defaults to ``20 nm`` — looser than typical e-beam writer specs
            so the term doesn't dominate early in training.
        pixel_size_nm: Override for ``pdk.pixel_size_nm``.
        weight_min_cd: Weight for the min-CD term.
        weight_min_spacing: Weight for the min-spacing term.
        weight_min_curvature: Weight for the min-curvature term.

    Returns:
        Scalar ``torch.Tensor`` (autograd-connected). Zero on a fully
        rule-respecting mask, positive otherwise.
    """
    if mask.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError(f"curvilinear_mrc_loss requires a floating-point mask, got {mask.dtype}.")

    width_nm, spacing_nm, curvature_nm, pixel_nm = _resolve_rules(
        pdk, min_width_nm, min_spacing_nm, min_curvature_radius_nm, pixel_size_nm
    )
    if pixel_nm <= 0:
        raise ValueError(f"pixel_size_nm must be positive, got {pixel_nm}.")

    m, _ = _ensure_b1hw(mask)
    m = m.clamp(0.0, 1.0)

    radius_width = max(0, int(width_nm // (2.0 * pixel_nm)))
    radius_spacing = max(0, int(spacing_nm // (2.0 * pixel_nm)))

    # Loss is averaged per pixel per sample, then meaned over batch — keeps
    # the magnitude comparable across resolutions and batch sizes.
    spatial = float(m.shape[-1] * m.shape[-2])

    cd_term = m.new_zeros(())
    if weight_min_cd != 0.0 and radius_width >= 1:
        cd_residual = _opening_residual(m, radius_width)
        cd_term = cd_residual.flatten(1).sum(dim=1).mean() / spatial

    spacing_term = m.new_zeros(())
    if weight_min_spacing != 0.0 and radius_spacing >= 1:
        spacing_residual = _opening_residual(1.0 - m, radius_spacing)
        spacing_term = spacing_residual.flatten(1).sum(dim=1).mean() / spatial

    curvature_term = m.new_zeros(())
    if weight_min_curvature != 0.0 and curvature_nm > 0.0:
        # Sobel-like central differences in nm⁻¹ — divide finite differences
        # by ``pixel_nm`` so the gradient magnitude is in physical units and
        # the threshold ``1 / curvature_nm`` is directly meaningful.
        gy = (m[..., 2:, :] - m[..., :-2, :]) / (2.0 * pixel_nm)
        gx = (m[..., :, 2:] - m[..., :, :-2]) / (2.0 * pixel_nm)
        # Crop to the common interior so gx and gy align spatially.
        gy = gy[..., :, 1:-1]
        gx = gx[..., 1:-1, :]
        grad_sq = gy * gy + gx * gx

        # Boundary band: symmetric difference of 1-px dilation and erosion.
        # Detached so it acts as a soft mask, not a target — the gradient
        # signal flows through ``grad_sq``, not through where the band lives.
        with torch.no_grad():
            band = (_soft_dilation(m, 1) - _soft_erosion(m, 1))[..., 1:-1, 1:-1]
            band = band.clamp(0.0, 1.0)

        threshold = (1.0 / curvature_nm) ** 2
        excess = functional.relu(grad_sq - threshold)
        curvature_term = (excess * band).flatten(1).sum(dim=1).mean() / spatial

    return (
        weight_min_cd * cd_term
        + weight_min_spacing * spacing_term
        + weight_min_curvature * curvature_term
    )
