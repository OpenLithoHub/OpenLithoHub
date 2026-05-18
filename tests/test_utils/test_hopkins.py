"""Tests for openlithohub._utils.hopkins."""

from __future__ import annotations

import pytest
import torch

from openlithohub._utils.hopkins import (
    HopkinsParams,
    clear_kernel_cache,
    compute_socs_kernels,
    simulate_aerial_image_hopkins,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_kernel_cache()


class TestHopkinsParams:
    def test_default_values(self) -> None:
        p = HopkinsParams()
        assert p.wavelength_nm == 193.0
        assert p.na == 1.35
        assert p.sigma == 0.7
        assert p.illumination == "circular"

    def test_cache_key_changes_with_params(self) -> None:
        a = HopkinsParams(sigma=0.7).cache_key(64, "cpu", "torch.complex64")
        b = HopkinsParams(sigma=0.5).cache_key(64, "cpu", "torch.complex64")
        assert a != b

    def test_cache_key_changes_with_grid(self) -> None:
        p = HopkinsParams()
        assert p.cache_key(64, "cpu", "torch.complex64") != p.cache_key(
            128, "cpu", "torch.complex64"
        )

    def test_cache_key_changes_with_dtype(self) -> None:
        p = HopkinsParams()
        assert p.cache_key(64, "cpu", "torch.complex64") != p.cache_key(
            64, "cpu", "torch.complex128"
        )


class TestComputeSocsKernels:
    def test_returns_complex_kernels_and_real_weights(self) -> None:
        kernels, weights = compute_socs_kernels(
            HopkinsParams(num_kernels=4, pixel_size_nm=2.0), grid_size=64
        )
        assert kernels.dtype == torch.complex64
        assert weights.dtype == torch.float32
        assert kernels.shape[1] == kernels.shape[2] == 64
        assert weights.shape[0] == kernels.shape[0]

    def test_kernel_count_capped_by_num_kernels(self) -> None:
        kernels, _ = compute_socs_kernels(
            HopkinsParams(num_kernels=4, pixel_size_nm=2.0), grid_size=128
        )
        assert kernels.shape[0] <= 4

    def test_weights_sorted_descending(self) -> None:
        _, weights = compute_socs_kernels(
            HopkinsParams(num_kernels=8, pixel_size_nm=2.0), grid_size=128
        )
        diffs = (weights[1:] - weights[:-1]).tolist()
        assert all(d <= 1e-5 for d in diffs)

    def test_cache_hit_returns_same_object(self) -> None:
        params = HopkinsParams(num_kernels=4, pixel_size_nm=2.0)
        k1, w1 = compute_socs_kernels(params, 64)
        k2, w2 = compute_socs_kernels(params, 64)
        assert k1 is k2 and w1 is w2

    def test_annular_illumination(self) -> None:
        k, _ = compute_socs_kernels(
            HopkinsParams(
                num_kernels=4,
                pixel_size_nm=2.0,
                sigma=0.9,
                sigma_inner=0.6,
                illumination="annular",
            ),
            grid_size=64,
        )
        assert k.shape[0] >= 1

    def test_dipole_illumination(self) -> None:
        k, _ = compute_socs_kernels(
            HopkinsParams(
                num_kernels=4,
                pixel_size_nm=2.0,
                sigma=0.9,
                illumination="dipole",
            ),
            grid_size=64,
        )
        assert k.shape[0] >= 1


class TestSimulateAerialImageHopkins:
    def test_open_frame_intensity_is_unity(self) -> None:
        params = HopkinsParams(num_kernels=8, pixel_size_nm=2.0)
        kernels, weights = compute_socs_kernels(params, 64)
        mask = torch.ones(64, 64)
        aerial = simulate_aerial_image_hopkins(mask, kernels=kernels, weights=weights)
        assert aerial.shape == mask.shape
        assert torch.allclose(aerial, torch.ones_like(aerial), atol=1e-3)

    def test_output_shape_matches_input(self) -> None:
        params = HopkinsParams(num_kernels=4, pixel_size_nm=2.0)
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0
        aerial = simulate_aerial_image_hopkins(mask, params=params)
        assert aerial.shape == mask.shape

    def test_intensity_is_nonnegative(self) -> None:
        params = HopkinsParams(num_kernels=4, pixel_size_nm=2.0)
        mask = torch.rand(64, 64)
        aerial = simulate_aerial_image_hopkins(mask, params=params)
        assert (aerial >= -1e-6).all()

    def test_dose_scales_linearly(self) -> None:
        params = HopkinsParams(num_kernels=4, pixel_size_nm=2.0)
        kernels, weights = compute_socs_kernels(params, 64)
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0
        a1 = simulate_aerial_image_hopkins(mask, kernels=kernels, weights=weights, dose=1.0)
        a2 = simulate_aerial_image_hopkins(mask, kernels=kernels, weights=weights, dose=2.5)
        assert torch.allclose(a2, a1 * 2.5, atol=1e-5)

    def test_preserves_gradient(self) -> None:
        params = HopkinsParams(num_kernels=4, pixel_size_nm=2.0)
        kernels, weights = compute_socs_kernels(params, 32)
        mask = torch.rand(32, 32, requires_grad=True)
        aerial = simulate_aerial_image_hopkins(mask, kernels=kernels, weights=weights)
        aerial.sum().backward()
        assert mask.grad is not None
        assert mask.grad.shape == mask.shape
        assert torch.isfinite(mask.grad).all()
        assert (mask.grad.abs() > 0).any()

    def test_accepts_4d_input(self) -> None:
        params = HopkinsParams(num_kernels=4, pixel_size_nm=2.0)
        mask = torch.zeros(1, 1, 64, 64)
        mask[..., 20:44, 20:44] = 1.0
        aerial = simulate_aerial_image_hopkins(mask, params=params)
        assert aerial.shape == mask.shape

    def test_rejects_non_square(self) -> None:
        params = HopkinsParams(num_kernels=4, pixel_size_nm=2.0)
        mask = torch.zeros(32, 64)
        with pytest.raises(ValueError, match="square"):
            simulate_aerial_image_hopkins(mask, params=params)

    def test_requires_params_or_kernels(self) -> None:
        mask = torch.zeros(32, 32)
        with pytest.raises(ValueError, match="kernels"):
            simulate_aerial_image_hopkins(mask, params=None, kernels=None, weights=None)


class TestDtypeAndCompile:
    def test_cache_key_includes_dtype(self) -> None:
        """Different dtypes must not collide in the kernel cache."""
        params = HopkinsParams(num_kernels=4, pixel_size_nm=2.0)
        k64, _ = compute_socs_kernels(params, 64, dtype=torch.complex64)
        k128, _ = compute_socs_kernels(params, 64, dtype=torch.complex128)
        assert k64.dtype == torch.complex64
        assert k128.dtype == torch.complex128
        # Re-fetch to confirm cache returns the right dtype on second call.
        k64_again, _ = compute_socs_kernels(params, 64, dtype=torch.complex64)
        assert k64_again.dtype == torch.complex64

    def test_bf16_vs_fp32_numerical_consistency(self) -> None:
        """bf16 forward should match fp32 within bf16's mantissa tolerance."""
        params = HopkinsParams(num_kernels=4, pixel_size_nm=2.0)
        kernels, weights = compute_socs_kernels(params, 64)
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0
        aerial_fp32 = simulate_aerial_image_hopkins(
            mask, kernels=kernels, weights=weights, dtype=torch.float32
        )
        aerial_bf16 = simulate_aerial_image_hopkins(
            mask, kernels=kernels, weights=weights, dtype=torch.bfloat16
        )
        assert aerial_bf16.dtype == torch.bfloat16
        assert torch.allclose(aerial_bf16.float(), aerial_fp32, atol=5e-3, rtol=5e-2)

    def test_compiled_forward_matches_eager(self) -> None:
        """torch.compile-wrapped forward should be numerically identical to eager."""
        params = HopkinsParams(num_kernels=4, pixel_size_nm=2.0)
        kernels, weights = compute_socs_kernels(params, 32)
        mask = torch.zeros(32, 32)
        mask[10:22, 10:22] = 1.0
        eager = simulate_aerial_image_hopkins(mask, kernels=kernels, weights=weights)
        try:
            compiled = torch.compile(simulate_aerial_image_hopkins, dynamic=False)
            out = compiled(mask, kernels=kernels, weights=weights)
        except Exception as exc:  # noqa: BLE001 — torch.compile may not be available
            pytest.skip(f"torch.compile unavailable in this env: {exc}")
        assert torch.allclose(out, eager, atol=1e-5)
