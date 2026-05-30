"""Tests for thick-mask forward model and U-Net proxy."""

import pytest
import torch

from openlithohub._utils.forward_model import (
    simulate_aerial_image,
    simulate_aerial_image_abbe,
    simulate_aerial_image_thick_mask,
)
from openlithohub.models.thick_mask_proxy import ThickMaskProxy


class TestThickMaskPerturbation:
    def test_thick_mask_perturbation_shape(self) -> None:
        """Thick mask output has correct shape."""
        mask = torch.rand(32, 32)
        result = simulate_aerial_image_thick_mask(
            mask,
            sigma_px=2.0,
            dose=1.0,
            thickness_nm=70.0,
            refractive_index_n=1.5,
            refractive_index_k=0.3,
            wavelength_nm=193.0,
        )
        assert result.shape == mask.shape

    def test_thick_mask_perturbation_shape_batched(self) -> None:
        """Thick mask output preserves batched input shape."""
        mask = torch.rand(4, 1, 32, 32)
        result = simulate_aerial_image_thick_mask(
            mask,
            sigma_px=2.0,
            thickness_nm=50.0,
        )
        assert result.shape == mask.shape

    def test_thick_mask_differentiable(self) -> None:
        """Thick mask forward is differentiable."""
        mask = torch.rand(16, 16, requires_grad=True)
        aerial = simulate_aerial_image_thick_mask(
            mask,
            sigma_px=2.0,
            thickness_nm=70.0,
            refractive_index_n=1.5,
            refractive_index_k=0.3,
        )
        loss = aerial.sum()
        loss.backward()
        assert mask.grad is not None
        assert mask.grad.shape == mask.shape

    def test_thick_mask_vs_hopkins(self) -> None:
        """Thick mask result differs from thin mask (Hopkins) when thickness > 0."""
        mask = torch.zeros(32, 32)
        mask[12:20, 12:20] = 1.0

        hopkins = simulate_aerial_image(mask, sigma_px=2.0, dose=1.0)
        thick = simulate_aerial_image_thick_mask(
            mask,
            sigma_px=2.0,
            dose=1.0,
            thickness_nm=70.0,
            refractive_index_n=1.5,
            refractive_index_k=0.5,
        )
        assert not torch.allclose(hopkins, thick, atol=1e-4)

    def test_zero_thickness_recovers_hopkins(self) -> None:
        """Zero thickness gives same result as Hopkins."""
        mask = torch.rand(32, 32)
        hopkins = simulate_aerial_image(mask, sigma_px=2.0, dose=1.0)
        thick = simulate_aerial_image_thick_mask(
            mask,
            sigma_px=2.0,
            dose=1.0,
            thickness_nm=0.0,
        )
        assert torch.allclose(hopkins, thick, atol=1e-5)

    def test_zero_sigma_returns_scaled_mask(self) -> None:
        """Zero sigma bypasses convolution."""
        mask = torch.ones(16, 16)
        result = simulate_aerial_image_thick_mask(
            mask,
            sigma_px=0.0,
            dose=2.0,
            thickness_nm=70.0,
        )
        assert torch.allclose(result, mask * 2.0)


class TestAbbeForward:
    def test_abbe_forward_shape(self) -> None:
        """Abbe partial coherence forward has correct shape."""
        mask = torch.rand(32, 32)
        result = simulate_aerial_image_abbe(
            mask,
            sigma_px=2.0,
            n_source_points=8,
            partial_coherence=0.7,
        )
        assert result.shape == mask.shape

    def test_abbe_forward_shape_batched(self) -> None:
        """Abbe forward preserves batched input shape."""
        mask = torch.rand(2, 1, 32, 32)
        result = simulate_aerial_image_abbe(
            mask,
            sigma_px=2.0,
            n_source_points=4,
        )
        assert result.shape == mask.shape

    def test_abbe_output_nonnegative(self) -> None:
        """Abbe forward produces non-negative intensities."""
        mask = torch.rand(32, 32)
        result = simulate_aerial_image_abbe(
            mask,
            sigma_px=2.0,
            n_source_points=8,
        )
        assert (result >= -1e-6).all()

    def test_abbe_zero_sigma_returns_scaled_mask(self) -> None:
        """Zero sigma bypasses convolution."""
        mask = torch.ones(16, 16)
        result = simulate_aerial_image_abbe(
            mask,
            sigma_px=0.0,
            dose=2.0,
        )
        assert torch.allclose(result, mask * 2.0)

    def test_abbe_differentiable(self) -> None:
        """Abbe forward is differentiable."""
        mask = torch.rand(16, 16, requires_grad=True)
        aerial = simulate_aerial_image_abbe(
            mask,
            sigma_px=2.0,
            n_source_points=4,
        )
        loss = aerial.sum()
        loss.backward()
        assert mask.grad is not None


class TestThickMaskProxy:
    def test_thick_mask_proxy_training(self) -> None:
        """ThickMaskProxy can be trained and produces valid output."""
        proxy = ThickMaskProxy()
        losses = proxy.train_from_born(
            n_samples=4,
            image_size=16,
            sigma_px=2.0,
            n_epochs=5,
            lr=1e-3,
        )
        assert len(losses) == 5
        assert all(isinstance(loss, float) for loss in losses)

        proxy.eval()
        mask = torch.rand(1, 1, 16, 16)
        thickness = torch.tensor([70.0])
        with torch.no_grad():
            output = proxy(mask, thickness)
        assert output.shape == mask.shape
        assert torch.isfinite(output).all()

    def test_proxy_rejects_bad_mask_shape(self) -> None:
        """Proxy rejects non-batched mask input."""
        proxy = ThickMaskProxy()
        mask = torch.rand(16, 16)
        thickness = torch.tensor([70.0])
        with pytest.raises(ValueError, match="Expected mask shape"):
            proxy(mask, thickness)

    def test_proxy_thickness_broadcast(self) -> None:
        """Proxy handles 1-D thickness tensor."""
        proxy = ThickMaskProxy()
        proxy.eval()
        mask = torch.rand(2, 1, 16, 16)
        thickness = torch.tensor([50.0, 100.0])
        with torch.no_grad():
            output = proxy(mask, thickness)
        assert output.shape == mask.shape
