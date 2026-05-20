"""Tests for openlithohub.workflow.process_window."""

from __future__ import annotations

import pytest
import torch

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.resist_model import differentiable_threshold
from openlithohub.workflow.process_window import (
    DEFAULT_PW_CORNERS,
    ProcessWindowCorner,
    pw_aerial_images,
    pw_fidelity_loss,
)


def _square_target(size: int = 16) -> torch.Tensor:
    target = torch.zeros(size, size)
    target[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1.0
    return target


class TestProcessWindowCorner:
    def test_default_corner_count(self) -> None:
        assert len(DEFAULT_PW_CORNERS) == 5

    def test_default_includes_nominal(self) -> None:
        nominal = [c for c in DEFAULT_PW_CORNERS if c.dose == 1.0 and c.sigma_px == 2.0]
        assert len(nominal) == 1
        # Nominal carries heavier weight than the off-nominal corners.
        assert nominal[0].weight > 1.0


class TestPwAerialImages:
    def test_returns_one_image_per_corner(self) -> None:
        mask = _square_target()
        corners = (
            ProcessWindowCorner(dose=1.0, sigma_px=2.0),
            ProcessWindowCorner(dose=0.9, sigma_px=2.5),
        )
        images = pw_aerial_images(mask, corners)
        assert len(images) == 2
        for img in images:
            assert img.shape == mask.shape

    def test_dose_scales_intensity(self) -> None:
        mask = _square_target()
        bright = pw_aerial_images(mask, (ProcessWindowCorner(dose=2.0, sigma_px=2.0),))[0]
        dim = pw_aerial_images(mask, (ProcessWindowCorner(dose=1.0, sigma_px=2.0),))[0]
        assert bright.mean().item() > dim.mean().item()


class TestPwFidelityLoss:
    def test_reduces_to_nominal_when_one_corner(self) -> None:
        mask = _square_target().requires_grad_(True)
        target = _square_target()

        single = (ProcessWindowCorner(dose=1.0, sigma_px=2.0, weight=1.0),)
        pw_loss = pw_fidelity_loss(mask, target, corners=single, threshold=0.5, steepness=50.0)

        aerial = simulate_aerial_image(mask, sigma_px=2.0, dose=1.0)
        resist = differentiable_threshold(aerial, threshold=0.5, steepness=50.0)
        nominal_loss = torch.nn.functional.mse_loss(resist, target)

        assert pw_loss.item() == pytest.approx(nominal_loss.item(), rel=1e-5)

    def test_gradient_flows(self) -> None:
        mask = _square_target().requires_grad_(True)
        target = _square_target()
        loss = pw_fidelity_loss(mask, target, corners=DEFAULT_PW_CORNERS)
        loss.backward()
        assert mask.grad is not None
        assert mask.grad.abs().sum().item() > 0.0

    def test_corner_weights_honored(self) -> None:
        """A heavily-weighted bad corner should dominate over a lightly-weighted good one."""
        mask = _square_target()
        target = _square_target()

        good = ProcessWindowCorner(dose=1.0, sigma_px=2.0, weight=1.0)
        bad = ProcessWindowCorner(dose=0.5, sigma_px=4.0, weight=1.0)

        loss_balanced = pw_fidelity_loss(mask, target, corners=(good, bad))
        loss_bad_heavy = pw_fidelity_loss(
            mask,
            target,
            corners=(good, ProcessWindowCorner(dose=0.5, sigma_px=4.0, weight=10.0)),
        )
        assert loss_bad_heavy.item() > loss_balanced.item()

    def test_empty_corners_raises(self) -> None:
        mask = _square_target()
        target = _square_target()
        with pytest.raises(ValueError, match="at least one corner"):
            pw_fidelity_loss(mask, target, corners=())

    def test_zero_weight_sum_raises(self) -> None:
        mask = _square_target()
        target = _square_target()
        with pytest.raises(ValueError, match="positive value"):
            pw_fidelity_loss(
                mask,
                target,
                corners=(ProcessWindowCorner(dose=1.0, sigma_px=2.0, weight=0.0),),
            )
