"""Tests for openlithohub._utils.resist_physics."""

import torch
import torch.nn as nn

from openlithohub._utils.resist_physics import (
    FidelityResult,
    GradientFidelityGate,
    PhysicalResistModel,
)


class TestPhysicalResistOutputRange:
    def test_output_in_zero_one(self) -> None:
        model = PhysicalResistModel()
        aerial = torch.rand(32, 32)
        out = model(aerial)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_extreme_inputs(self) -> None:
        model = PhysicalResistModel()
        # Very large positive input -> sigmoid saturates near 1
        aerial_high = torch.full((8, 8), 100.0)
        out_high = model(aerial_high)
        assert out_high.max() <= 1.0

        # Zero input -> after subtracting quencher -> negative -> sigmoid near 0
        aerial_low = torch.zeros(8, 8)
        out_low = model(aerial_low)
        assert out_low.min() >= 0.0


class TestPhysicalResistGradientExists:
    def test_gradient_flows(self) -> None:
        model = PhysicalResistModel()
        aerial = torch.rand(16, 16, requires_grad=True)
        out = model(aerial)
        loss = out.sum()
        loss.backward()
        assert aerial.grad is not None
        assert aerial.grad.abs().sum() > 0.0

    def test_gradient_has_correct_shape(self) -> None:
        model = PhysicalResistModel()
        aerial = torch.rand(16, 16, requires_grad=True)
        out = model(aerial)
        loss = out.sum()
        loss.backward()
        assert aerial.grad.shape == aerial.shape


class TestPhysicalResistMonotoneInDose:
    def test_higher_dose_higher_resist(self) -> None:
        model = PhysicalResistModel(acid_diffusion_nm=0.0)
        low_dose = torch.full((8, 8), 0.3)
        high_dose = torch.full((8, 8), 0.7)
        out_low = model(low_dose)
        out_high = model(high_dose)
        assert (out_high >= out_low).all()

    def test_monotone_with_diffusion(self) -> None:
        model = PhysicalResistModel(acid_diffusion_nm=3.0, pixel_size_nm=1.0)
        low_dose = torch.full((16, 16), 0.3)
        high_dose = torch.full((16, 16), 0.7)
        out_low = model(low_dose)
        out_high = model(high_dose)
        # On average higher dose gives higher resist
        assert out_high.mean() >= out_low.mean()


class TestAcidDiffusionSmooths:
    def test_diffusion_reduces_variance(self) -> None:
        model_no_diff = PhysicalResistModel(acid_diffusion_nm=0.0)
        model_with_diff = PhysicalResistModel(acid_diffusion_nm=8.0, pixel_size_nm=1.0)

        aerial = torch.zeros(32, 32)
        aerial[10:22, 10:22] = 1.0

        out_no = model_no_diff(aerial)
        out_diff = model_with_diff(aerial)

        # Diffused output should have lower variance (smoother)
        assert out_diff.var() <= out_no.var()

    def test_diffusion_preserves_mean(self) -> None:
        model_no = PhysicalResistModel(acid_diffusion_nm=0.0, quencher_concentration=0.0)
        model_diff = PhysicalResistModel(acid_diffusion_nm=4.0, pixel_size_nm=1.0, quencher_concentration=0.0)

        aerial = torch.rand(32, 32)
        # Acid diffusion is energy-conserving (Gaussian kernel sums to 1)
        acid_no = model_no._acid_generation(aerial)
        acid_diff = model_diff._acid_diffusion(acid_no)
        # Means should be close (not exact due to sigmoid nonlinearity,
        # but acid field means should match)
        assert torch.allclose(acid_no.mean(), acid_diff.mean(), atol=1e-4)


class TestGradientFidelityGatePassesForConsistent:
    def test_identical_functions_pass(self) -> None:
        identity = nn.Identity()

        class SurrogateWrapper(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return identity(x)

        class HFWrapper(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return identity(x)

        gate = GradientFidelityGate(
            surrogate_fn=SurrogateWrapper(),
            high_fidelity_fn=HFWrapper(),
        )
        mask = torch.rand(4, 4)
        target = torch.rand(4, 4)
        result = gate.verify(mask, target)
        assert isinstance(result, FidelityResult)
        assert result.passed

    def test_consistent_approximation_passes(self) -> None:
        """Surrogate is a slight smoothing of the high-fidelity — should pass."""
        hf_fn = lambda x: torch.sigmoid(10.0 * x)
        surr_fn = lambda x: torch.sigmoid(10.0 * x) + 0.0  # identical

        gate = GradientFidelityGate(surrogate_fn=surr_fn, high_fidelity_fn=hf_fn)
        mask = torch.rand(6, 6)
        target = torch.bernoulli(torch.full((6, 6), 0.5))
        result = gate.verify(mask, target)
        assert result.passed
        assert result.surrogate_gradient_cosine > 0.99


class TestGradientFidelityGateCatchesDivergence:
    def test_divergent_functions_fail(self) -> None:
        """Surrogate with opposite gradient direction should fail."""
        hf_fn = lambda x: torch.sigmoid(5.0 * x)
        # Surrogate has inverted gradient
        surr_fn = lambda x: -torch.sigmoid(5.0 * x)

        gate = GradientFidelityGate(
            surrogate_fn=surr_fn,
            high_fidelity_fn=hf_fn,
            cosine_threshold=0.5,
        )
        mask = torch.rand(4, 4)
        target = torch.rand(4, 4)
        result = gate.verify(mask, target)
        assert not result.passed

    def test_scaled_gradient_detected(self) -> None:
        """Surrogate with much larger gradient magnitude should fail max-component check."""
        hf_fn = lambda x: torch.sigmoid(5.0 * x)
        surr_fn = lambda x: 3.0 * torch.sigmoid(5.0 * x)

        gate = GradientFidelityGate(
            surrogate_fn=surr_fn,
            high_fidelity_fn=hf_fn,
            cosine_threshold=0.5,
            max_component_threshold=0.1,
        )
        mask = torch.rand(4, 4)
        target = torch.rand(4, 4)
        result = gate.verify(mask, target)
        assert not result.passed


class TestGradientCosineAboveThreshold:
    def test_cosine_metric_close_to_one(self) -> None:
        hf_fn = lambda x: torch.sigmoid(10.0 * x)
        surr_fn = lambda x: torch.sigmoid(10.0 * x)

        gate = GradientFidelityGate(surrogate_fn=surr_fn, high_fidelity_fn=hf_fn)
        mask = torch.rand(8, 8)
        target = torch.rand(8, 8)
        result = gate.verify(mask, target)
        assert result.surrogate_gradient_cosine > 0.99


class TestBenchmarkReturnsStatistics:
    def test_benchmark_structure(self) -> None:
        fn = lambda x: torch.sigmoid(5.0 * x)
        gate = GradientFidelityGate(surrogate_fn=fn, high_fidelity_fn=fn)
        stats = gate.benchmark(n_masks=3, size=4)

        assert stats["n_masks"] == 3
        assert "surrogate_cosine_mean" in stats
        assert "surrogate_cosine_min" in stats
        assert "fd_cosine_mean" in stats
        assert "fd_cosine_min" in stats
        assert "max_component_error_mean" in stats
        assert "max_component_error_max" in stats
        assert "all_passed" in stats
        assert "n_passed" in stats

    def test_benchmark_all_passed_for_identical(self) -> None:
        fn = lambda x: torch.sigmoid(5.0 * x)
        gate = GradientFidelityGate(surrogate_fn=fn, high_fidelity_fn=fn)
        stats = gate.benchmark(n_masks=3, size=4)
        assert stats["all_passed"] is True
        assert stats["n_passed"] == 3
        assert stats["surrogate_cosine_min"] > 0.99
