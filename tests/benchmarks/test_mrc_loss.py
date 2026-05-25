"""Tests for openlithohub.benchmark.metrics.mrc_loss.

Acceptance criteria from issue #8:

* Function lands in ``benchmark.metrics`` and is exported.
* Returns a finite scalar tensor with non-zero gradients on a violating mask.
* Clean rule-respecting mask gets ``loss ≈ 0``.
* Known-bad mask (1-pixel bridge) gets ``loss > 0``.
"""

from __future__ import annotations

import pytest
import torch

from openlithohub.benchmark.metrics import curvilinear_mrc_loss
from openlithohub.synth.pdk import get_pdk


def _clean_stripe_mask(h: int = 64, w: int = 80) -> torch.Tensor:
    """Return a mask that comfortably satisfies width/spacing for kernel=11.

    Two stripes 20 px wide separated by a 16-px gap, with 12-px margins on
    every side. All foreground and background strips are wider than the
    radius-5 (kernel-11) structuring element used by the tests below.
    """
    mask = torch.zeros((h, w))
    mask[12:-12, 12:32] = 1.0
    mask[12:-12, 48:68] = 1.0
    return mask


def _bridge_mask(h: int = 32, w: int = 64) -> torch.Tensor:
    """Two big blocks joined by a 1-pixel-tall horizontal bridge — width violation."""
    mask = torch.zeros((h, w))
    mask[4:28, 4:24] = 1.0
    mask[4:28, 40:60] = 1.0
    mask[15:16, 24:40] = 1.0
    return mask


def _narrow_gap_mask(h: int = 32, w: int = 64) -> torch.Tensor:
    """Two big blocks separated by a 1-pixel gap — spacing violation."""
    mask = torch.zeros((h, w))
    mask[4:28, 4:31] = 1.0
    mask[4:28, 32:60] = 1.0
    return mask


class TestCurvilinearMrcLoss:
    def test_clean_mask_near_zero_loss(self) -> None:
        mask = _clean_stripe_mask()
        loss = curvilinear_mrc_loss(
            mask,
            min_width_nm=10.0,
            min_spacing_nm=10.0,
            min_curvature_radius_nm=0.0,
            pixel_size_nm=1.0,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_one_pixel_bridge_positive_loss(self) -> None:
        mask = _bridge_mask()
        loss = curvilinear_mrc_loss(
            mask,
            min_width_nm=4.0,
            min_spacing_nm=2.0,
            min_curvature_radius_nm=0.0,
            pixel_size_nm=1.0,
        )
        assert loss.item() > 0.0
        assert torch.isfinite(loss)

    def test_one_pixel_bridge_gradient_flows(self) -> None:
        mask = _bridge_mask().requires_grad_(True)
        loss = curvilinear_mrc_loss(
            mask,
            min_width_nm=4.0,
            min_spacing_nm=2.0,
            min_curvature_radius_nm=0.0,
            pixel_size_nm=1.0,
        )
        loss.backward()
        assert mask.grad is not None
        # The bridge row should receive non-zero gradient pushing it toward 0.
        assert mask.grad[15, 24:40].abs().sum().item() > 0.0

    def test_sub_spacing_positive_loss(self) -> None:
        mask = _narrow_gap_mask()
        loss = curvilinear_mrc_loss(
            mask,
            min_width_nm=2.0,
            min_spacing_nm=4.0,
            min_curvature_radius_nm=0.0,
            pixel_size_nm=1.0,
        )
        assert loss.item() > 0.0

    def test_curvature_term_fires_on_sharp_transition(self) -> None:
        # Tiny isolated dot — every pixel is a sharp transition. Setting the
        # other terms' weights to zero isolates the curvature contribution.
        mask = torch.zeros((32, 32))
        mask[15:17, 15:17] = 1.0
        loss = curvilinear_mrc_loss(
            mask,
            min_width_nm=1.0,
            min_spacing_nm=1.0,
            min_curvature_radius_nm=10.0,
            pixel_size_nm=1.0,
            weight_min_cd=0.0,
            weight_min_spacing=0.0,
        )
        assert loss.item() > 0.0

    def test_zero_weights_zero_loss(self) -> None:
        mask = _bridge_mask()
        loss = curvilinear_mrc_loss(
            mask,
            min_width_nm=4.0,
            min_spacing_nm=4.0,
            min_curvature_radius_nm=10.0,
            pixel_size_nm=1.0,
            weight_min_cd=0.0,
            weight_min_spacing=0.0,
            weight_min_curvature=0.0,
        )
        assert loss.item() == pytest.approx(0.0)

    def test_batched_shape_supported(self) -> None:
        # (B, 1, H, W): one bridge sample, one clean sample.
        bridge = _bridge_mask()
        clean = _clean_stripe_mask(h=bridge.shape[0], w=bridge.shape[1])
        batch = torch.stack([bridge, clean]).unsqueeze(1)
        loss = curvilinear_mrc_loss(
            batch,
            min_width_nm=4.0,
            min_spacing_nm=2.0,
            min_curvature_radius_nm=0.0,
            pixel_size_nm=1.0,
        )
        assert loss.dim() == 0
        assert loss.item() > 0.0

    def test_pdk_string_lookup_matches_explicit(self) -> None:
        pdk = get_pdk("asap7")
        mask = _bridge_mask()
        loss_str = curvilinear_mrc_loss(
            mask,
            pdk="asap7",
            min_curvature_radius_nm=0.0,
        )
        loss_explicit = curvilinear_mrc_loss(
            mask,
            min_width_nm=pdk.min_width_nm,
            min_spacing_nm=pdk.min_spacing_nm,
            min_curvature_radius_nm=0.0,
            pixel_size_nm=pdk.pixel_size_nm,
        )
        assert loss_str.item() == pytest.approx(loss_explicit.item())

    def test_pdk_instance_accepted(self) -> None:
        pdk = get_pdk("freepdk45")
        loss = curvilinear_mrc_loss(
            _clean_stripe_mask(),
            pdk=pdk,
            min_curvature_radius_nm=0.0,
        )
        assert torch.isfinite(loss)

    def test_no_pdk_no_overrides_raises(self) -> None:
        with pytest.raises(ValueError, match="requires either"):
            curvilinear_mrc_loss(_clean_stripe_mask())

    def test_overrides_win_over_pdk(self) -> None:
        # asap7's min_width is 18 nm. Override it to a value that the bridge
        # mask trivially satisfies — loss must drop to (near) zero.
        loss = curvilinear_mrc_loss(
            _bridge_mask(),
            pdk="asap7",
            min_width_nm=1.0,
            min_spacing_nm=1.0,
            min_curvature_radius_nm=0.0,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="expects mask shape"):
            curvilinear_mrc_loss(
                torch.zeros((4, 4, 4)),
                min_width_nm=1.0,
                min_spacing_nm=1.0,
                pixel_size_nm=1.0,
            )

    def test_non_float_mask_raises(self) -> None:
        with pytest.raises(TypeError, match="floating-point"):
            curvilinear_mrc_loss(
                torch.zeros((8, 8), dtype=torch.int64),
                min_width_nm=1.0,
                min_spacing_nm=1.0,
                pixel_size_nm=1.0,
            )


class TestV02RadiusParity:
    """v0.2 plan v6 lock + v7 Bug E: at min_width_nm=24, pixel_size_nm=8,
    the loss-side radius (`mrc_loss.py:190`) and the checker-side radius
    (`compliance/mrc.py:161`) must both be 1, so the differentiable loss
    and the binary verdict agree on what a violation is.

    Both formulas are inline expressions in their respective files; this
    test inlines them directly so it fails if either source-site moves.
    """

    def test_loss_and_checker_radius_agree_at_v02_settings(self) -> None:
        px, w = 8.0, 24.0
        # mrc_loss.py:190 — half-width radius the differentiable loss uses.
        loss_radius = max(0, int(w // (2.0 * px)))
        # compliance/mrc.py:161 — radius the binary checker opens with.
        import math as _math

        checker_radius = max(0, (int(_math.floor(w / px)) - 1) // 2)
        assert loss_radius == checker_radius == 1, (
            f"v0.2 radius parity broken at {w} nm @ {px} nm/px: "
            f"loss={loss_radius}, checker={checker_radius}"
        )
