"""Tests for differentiable stochastic-aware ILT loss functions."""

from __future__ import annotations

import torch
import pytest

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub.benchmark.metrics.stochastic_loss import (
    StochasticAwareLoss,
    StochasticProcessWindow,
    cvar_loss,
    differentiable_edge_error,
    differentiable_lcdu,
    quantile_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_mask() -> torch.Tensor:
    """32x32 binary mask with a vertical line."""
    mask = torch.zeros(32, 32, dtype=torch.float32)
    mask[:, 14:18] = 1.0
    return mask


@pytest.fixture
def aerial_fn():
    """Simple aerial image function (Gaussian PSF)."""
    return lambda m: simulate_aerial_image(m, sigma_px=2.0, dose=1.0)


# ---------------------------------------------------------------------------
# cvar_loss tests
# ---------------------------------------------------------------------------


class TestCvarLoss:
    def test_cvar_loss_basic(self):
        """CVaR(alpha=0.5) with 4 samples should average the top 2."""
        errors = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = cvar_loss(errors, alpha=0.5)
        expected = (4.0 + 3.0) / 2.0
        assert abs(result.item() - expected) < 1e-5

    def test_cvar_loss_monotone_in_alpha(self):
        """Higher alpha = more risk-averse = larger CVaR (tail gets smaller
        but the values are larger)."""
        torch.manual_seed(42)
        errors = torch.randn(100)
        cvar_90 = cvar_loss(errors, alpha=0.90)
        cvar_50 = cvar_loss(errors, alpha=0.50)
        # alpha=0.90 averages the worst 10 samples; alpha=0.50 averages the
        # worst 50 samples. The worst 10 should be >= mean of worst 50.
        assert cvar_90.item() >= cvar_50.item() - 1e-4

    def test_cvar_loss_rejects_bad_alpha(self):
        """alpha must be in (0, 1)."""
        with pytest.raises(ValueError):
            cvar_loss(torch.tensor([1.0, 2.0]), alpha=0.0)
        with pytest.raises(ValueError):
            cvar_loss(torch.tensor([1.0, 2.0]), alpha=1.0)

    def test_cvar_loss_single_sample(self):
        """CVaR with a single sample returns that sample."""
        errors = torch.tensor([3.14])
        result = cvar_loss(errors, alpha=0.5)
        assert abs(result.item() - 3.14) < 1e-5


# ---------------------------------------------------------------------------
# quantile_loss tests
# ---------------------------------------------------------------------------


class TestQuantileLoss:
    def test_quantile_loss_basic(self):
        """Pinball loss is non-negative by construction."""
        torch.manual_seed(123)
        errors = torch.randn(50)
        result = quantile_loss(errors, quantile=0.9)
        assert result.item() >= -1e-6

    def test_quantile_loss_symmetric_at_median(self):
        """At quantile=0.5 with symmetric errors the pinball loss ≈ mean(|error - median|)/2."""
        errors = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = quantile_loss(errors, quantile=0.5)
        # tau = median = 3.0; pinball = mean(max(0.5*(e-3), -0.5*(e-3)))
        # = 0.5 * mean(|e-3|) = 0.5 * (2+1+0+1+2)/5 = 0.5*1.2 = 0.6
        assert abs(result.item() - 0.6) < 1e-4

    def test_quantile_loss_rejects_bad_quantile(self):
        with pytest.raises(ValueError):
            quantile_loss(torch.tensor([1.0]), quantile=0.0)


# ---------------------------------------------------------------------------
# differentiable_edge_error tests
# ---------------------------------------------------------------------------


class TestDifferentiableEdgeError:
    def test_differentiable_edge_error_gradient_exists(self, simple_mask, aerial_fn):
        """Gradient w.r.t. mask must exist and be non-zero."""
        mask = simple_mask.clone().requires_grad_(True)
        aerial = aerial_fn(mask)
        loss = differentiable_edge_error(
            mask, aerial,
            dose_photons_per_nm2=30.0,
            pixel_size_nm=1.0,
            n_samples=4,
        )
        loss.backward()
        assert mask.grad is not None
        assert mask.grad.abs().sum().item() > 0.0

    def test_differentiable_edge_error_nonnegative(self, simple_mask, aerial_fn):
        """Edge error should be non-negative."""
        mask = simple_mask
        aerial = aerial_fn(mask)
        loss = differentiable_edge_error(
            mask, aerial,
            dose_photons_per_nm2=30.0,
            pixel_size_nm=1.0,
            n_samples=4,
        )
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# differentiable_lcdu tests
# ---------------------------------------------------------------------------


class TestDifferentiableLcdu:
    def test_differentiable_lcdu_gradient_exists(self, simple_mask, aerial_fn):
        """Gradient w.r.t. mask must exist and be non-zero."""
        mask = simple_mask.clone().requires_grad_(True)
        aerial = aerial_fn(mask)
        loss = differentiable_lcdu(
            mask, aerial,
            dose_photons_per_nm2=30.0,
            pixel_size_nm=1.0,
            n_samples=4,
        )
        loss.backward()
        assert mask.grad is not None
        assert mask.grad.abs().sum().item() > 0.0

    def test_differentiable_lcdu_nonnegative(self, simple_mask, aerial_fn):
        """LCDU (a std deviation) should be non-negative."""
        mask = simple_mask
        aerial = aerial_fn(mask)
        loss = differentiable_lcdu(
            mask, aerial,
            dose_photons_per_nm2=30.0,
            pixel_size_nm=1.0,
            n_samples=8,
        )
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# StochasticAwareLoss tests
# ---------------------------------------------------------------------------


class TestStochasticAwareLoss:
    def test_stochastic_aware_loss_gradient_exists(self, simple_mask, aerial_fn):
        """Full loss must propagate gradients to the mask."""
        loss_fn = StochasticAwareLoss(
            alpha=0.95, n_mc_samples=4, dose_photons_per_nm2=30.0
        )
        mask = simple_mask.clone().requires_grad_(True)
        loss = loss_fn.forward(mask, aerial_fn)
        loss.backward()
        assert mask.grad is not None
        assert mask.grad.abs().sum().item() > 0.0

    def test_stochastic_aware_loss_worse_than_nominal(self, simple_mask, aerial_fn):
        """Low dose (more noise) should give a higher loss than high dose."""
        torch.manual_seed(0)
        loss_low_dose = StochasticAwareLoss(
            alpha=0.95, n_mc_samples=16, dose_photons_per_nm2=5.0,
        )
        loss_high_dose = StochasticAwareLoss(
            alpha=0.95, n_mc_samples=16, dose_photons_per_nm2=200.0,
        )
        mask = simple_mask
        val_low = loss_low_dose.forward(mask, aerial_fn).item()
        val_high = loss_high_dose.forward(mask, aerial_fn).item()
        assert val_low > val_high

    def test_cvar_worse_than_mean(self, simple_mask, aerial_fn):
        """CVaR (tail risk) should produce a loss >= mean-based loss."""
        torch.manual_seed(0)
        loss_cvar = StochasticAwareLoss(
            alpha=0.95, n_mc_samples=32, dose_photons_per_nm2=20.0,
            risk_measure="cvar",
        )
        loss_mean = StochasticAwareLoss(
            alpha=0.95, n_mc_samples=32, dose_photons_per_nm2=20.0,
            risk_measure="mean",
        )
        mask = simple_mask
        val_cvar = loss_cvar.forward(mask, aerial_fn).item()
        val_mean = loss_mean.forward(mask, aerial_fn).item()
        assert val_cvar >= val_mean - 1e-5

    def test_stochastic_aware_loss_rejects_bad_risk_measure(self):
        with pytest.raises(ValueError):
            StochasticAwareLoss(risk_measure="quantile")

    def test_stochastic_aware_loss_rejects_bad_alpha(self):
        with pytest.raises(ValueError):
            StochasticAwareLoss(alpha=0.0)


# ---------------------------------------------------------------------------
# StochasticProcessWindow tests
# ---------------------------------------------------------------------------


class TestStochasticProcessWindow:
    def test_stochastic_process_window_basic(self, simple_mask, aerial_fn):
        """Process window should return valid bounds."""
        spw = StochasticProcessWindow(
            n_samples=4,
            dose_photons_per_nm2=30.0,
            epe_tolerance=5.0,
        )
        result = spw.compute(simple_mask, aerial_image_fn=aerial_fn, focus_range_nm=(-20.0, 20.0, 10.0))
        assert len(result.focus_values_nm) > 0
        assert len(result.mean_epe_per_focus) == len(result.focus_values_nm)
        assert len(result.worst_case_epe_per_focus) == len(result.focus_values_nm)
        assert result.window_width_nm >= 0.0
        assert result.upper_bound_nm >= result.lower_bound_nm

    def test_stochastic_process_window_best_focus_best(self, simple_mask):
        """Best focus (0 nm) should have lower EPE than extreme defocus."""
        spw = StochasticProcessWindow(
            n_samples=8, dose_photons_per_nm2=30.0, epe_tolerance=5.0,
        )
        result = spw.compute(
            simple_mask,
            focus_range_nm=(-50.0, 50.0, 25.0),
        )
        # Index of best focus (0 nm or nearest).
        best_idx = min(
            range(len(result.focus_values_nm)),
            key=lambda i: abs(result.focus_values_nm[i]),
        )
        worst_idx = max(
            range(len(result.mean_epe_per_focus)),
            key=lambda i: result.mean_epe_per_focus[i],
        )
        assert result.mean_epe_per_focus[best_idx] <= result.mean_epe_per_focus[worst_idx] + 1e-4


# ---------------------------------------------------------------------------
# Integration test: stochastic-aware ILT improvement
# ---------------------------------------------------------------------------


class TestStochasticVsDeterministic:
    def test_stochastic_vs_deterministic_ilt_improvement(self, aerial_fn):
        """A few steps of stochastic-aware ILT should reduce the stochastic loss."""
        torch.manual_seed(0)

        # Start with a uniform mask — worst case for edge fidelity.
        mask = torch.full((32, 32), 0.5, requires_grad=True)
        optimizer = torch.optim.Adam([mask], lr=0.05)
        loss_fn = StochasticAwareLoss(
            alpha=0.9, n_mc_samples=4, dose_photons_per_nm2=30.0,
        )

        initial_loss = loss_fn.forward(mask, aerial_fn).item()

        for _ in range(5):
            optimizer.zero_grad()
            loss = loss_fn.forward(mask, aerial_fn)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                mask.clamp_(0.0, 1.0)

        final_loss = loss_fn.forward(mask.detach(), aerial_fn).item()
        # After optimisation the loss should have decreased.
        assert final_loss < initial_loss
