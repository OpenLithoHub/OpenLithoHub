"""Tests for openlithohub._utils.forward_model."""

import pytest
import torch

from openlithohub._utils.forward_model import (
    _build_gaussian_kernel,
    _circular_pad_clamped,
    apply_resist_threshold,
    simulate_aerial_image,
)


class TestBuildGaussianKernel:
    def test_kernel_sums_to_one(self) -> None:
        kernel = _build_gaussian_kernel(2.0, torch.device("cpu"))
        assert abs(kernel.sum().item() - 1.0) < 1e-5

    def test_kernel_is_symmetric(self) -> None:
        kernel = _build_gaussian_kernel(3.0, torch.device("cpu"))
        k = kernel.squeeze()
        assert torch.allclose(k, k.flip(0), atol=1e-6)
        assert torch.allclose(k, k.flip(1), atol=1e-6)

    def test_kernel_shape_depends_on_sigma(self) -> None:
        k1 = _build_gaussian_kernel(1.0, torch.device("cpu"))
        k2 = _build_gaussian_kernel(5.0, torch.device("cpu"))
        assert k2.shape[-1] > k1.shape[-1]

    def test_kernel_is_4d(self) -> None:
        kernel = _build_gaussian_kernel(2.0, torch.device("cpu"))
        assert kernel.ndim == 4
        assert kernel.shape[0] == 1
        assert kernel.shape[1] == 1


class TestSimulateAerialImage:
    def test_zero_sigma_returns_scaled_mask(self) -> None:
        mask = torch.ones(16, 16)
        result = simulate_aerial_image(mask, sigma_px=0.0, dose=2.0)
        assert torch.allclose(result, mask * 2.0)

    def test_blurring_reduces_contrast(self) -> None:
        mask = torch.zeros(32, 32)
        mask[14:18, 14:18] = 1.0  # small 4x4 feature
        aerial = simulate_aerial_image(mask, sigma_px=3.0)
        # Peak intensity is reduced due to PSF spreading
        assert aerial.max() < 1.0

    def test_dose_scaling(self) -> None:
        mask = torch.ones(16, 16)
        a1 = simulate_aerial_image(mask, sigma_px=1.0, dose=1.0)
        a2 = simulate_aerial_image(mask, sigma_px=1.0, dose=2.0)
        assert torch.allclose(a2, a1 * 2.0, atol=1e-5)

    def test_output_shape_matches_input(self) -> None:
        mask = torch.rand(64, 64)
        result = simulate_aerial_image(mask, sigma_px=2.0)
        assert result.shape == mask.shape

    def test_preserves_gradient(self) -> None:
        mask = torch.rand(16, 16, requires_grad=True)
        aerial = simulate_aerial_image(mask, sigma_px=2.0)
        loss = aerial.sum()
        loss.backward()
        assert mask.grad is not None
        assert mask.grad.shape == mask.shape


class TestApplyResistThreshold:
    def test_binary_output(self) -> None:
        aerial = torch.rand(32, 32)
        resist = apply_resist_threshold(aerial, threshold=0.5)
        unique_vals = resist.unique()
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_threshold_at_zero(self) -> None:
        aerial = torch.rand(16, 16) + 0.01
        resist = apply_resist_threshold(aerial, threshold=0.0)
        assert resist.sum() == resist.numel()

    def test_threshold_at_one(self) -> None:
        aerial = torch.rand(16, 16) * 0.99
        resist = apply_resist_threshold(aerial, threshold=1.0)
        assert resist.sum() == 0.0

    def test_custom_threshold(self) -> None:
        aerial = torch.tensor([[0.3, 0.7], [0.1, 0.9]])
        resist = apply_resist_threshold(aerial, threshold=0.5)
        expected = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        assert torch.equal(resist, expected)


class TestCircularPadClamped:
    """Issue #10: 1-px-wide axis used to silently fall back to replicate
    padding via a `warnings.warn`, but Python's default warning filter is
    once-per-location, so subsequent metric calls picked up replicate-pad
    edge fringes silently. The contract is now strict: raise instead of
    fall back so misconfigured inputs surface loudly."""

    def test_circular_pad_works_on_normal_input(self) -> None:
        inp = torch.rand(1, 1, 8, 8)
        out = _circular_pad_clamped(inp, padding=2)
        assert out.shape == (1, 1, 12, 12)

    def test_one_pixel_axis_raises(self) -> None:
        inp = torch.rand(1, 1, 1, 8)
        with pytest.raises(ValueError, match="1-pixel-wide axis"):
            _circular_pad_clamped(inp, padding=2)

    def test_one_pixel_both_axes_raises(self) -> None:
        inp = torch.rand(1, 1, 1, 1)
        with pytest.raises(ValueError, match="1-pixel-wide axis"):
            _circular_pad_clamped(inp, padding=1)
