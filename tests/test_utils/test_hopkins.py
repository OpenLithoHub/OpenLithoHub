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


class TestPolarJacobian:
    """Issue #29 regression: source samples must carry the polar `r·dr·dθ`
    Jacobian, otherwise the centre is over-weighted because every angular
    bin has the same sample count regardless of its physical area."""

    def test_circular_source_inner_third_not_overweighted(self) -> None:
        from openlithohub._utils.hopkins import _illumination_samples

        # 512 px @ 1 nm pitch resolves the source disk well enough that the
        # binning artifact is bounded; without the Jacobian, the inner third
        # held ~21% of the source intensity (measured pre-fix) instead of
        # the geometric ~11%. After the fix, it must come in below 15%.
        params = HopkinsParams(illumination="circular", sigma=0.7, pixel_size_nm=1.0)
        shifts, weights = _illumination_samples(params, 512, torch.device("cpu"))

        f_pupil = params.na / params.wavelength_nm
        fstep = 1.0 / (512 * params.pixel_size_nm)
        r_source_bins = f_pupil * params.sigma / fstep

        sy = shifts[:, 0].to(torch.float32)
        sx = shifts[:, 1].to(torch.float32)
        sy = torch.where(sy <= 256, sy, sy - 512)
        sx = torch.where(sx <= 256, sx, sx - 512)
        r = torch.sqrt(sy * sy + sx * sx)

        inner_w = float(weights[r <= r_source_bins / 3.0].sum())
        total_w = float(weights.sum())
        assert total_w == pytest.approx(1.0, abs=1e-6)
        assert inner_w / total_w < 0.15, (
            f"inner-third weight {inner_w / total_w:.3f} too large — Jacobian missing?"
        )


class TestPrecomputedKernelsF:
    """v0.2 P2-a regression: precomputed_kernels_f kwarg must produce
    bit-equivalent (within float tolerance) aerial images vs. the no-FFT
    path, so training with the precompute is numerically faithful to v0.1
    inference paths that don't pass it."""

    def test_precomputed_path_matches_inner_loop_path(self) -> None:
        from openlithohub._utils.hopkins import (
            HopkinsParams,
            compute_socs_kernels,
            simulate_aerial_image_hopkins,
        )

        torch.manual_seed(0)
        H = 64  # noqa: N806
        params = HopkinsParams(num_kernels=4, pixel_size_nm=8.0)
        device = torch.device("cpu")
        kernels, weights = compute_socs_kernels(params, H, device)

        kernels_c64 = kernels.to(torch.complex64)
        kernels_shifted = torch.fft.ifftshift(kernels_c64, dim=(-2, -1))
        kernels_f = torch.fft.fft2(kernels_shifted)

        mask = (torch.rand(H, H) > 0.5).float()

        out_inner = simulate_aerial_image_hopkins(mask, kernels=kernels, weights=weights)
        out_pre = simulate_aerial_image_hopkins(
            mask,
            kernels=kernels,
            weights=weights,
            precomputed_kernels_f=kernels_f,
        )
        assert torch.allclose(out_inner, out_pre, atol=1e-5), (
            f"precomputed path diverges: max diff {(out_inner - out_pre).abs().max().item():.3e}"
        )

    def test_precomputed_dtype_coerced_to_complex64(self) -> None:
        from openlithohub._utils.hopkins import (
            HopkinsParams,
            compute_socs_kernels,
            simulate_aerial_image_hopkins,
        )

        H = 64  # noqa: N806
        params = HopkinsParams(num_kernels=4, pixel_size_nm=8.0)
        device = torch.device("cpu")
        kernels, weights = compute_socs_kernels(params, H, device)
        kernels_shifted = torch.fft.ifftshift(kernels.to(torch.complex64), dim=(-2, -1))
        kernels_f_c128 = torch.fft.fft2(kernels_shifted).to(torch.complex128)

        mask = torch.zeros(H, H)
        mask[20:40, 20:40] = 1.0

        out_c128 = simulate_aerial_image_hopkins(
            mask, kernels=kernels, weights=weights, precomputed_kernels_f=kernels_f_c128
        )
        out_c64 = simulate_aerial_image_hopkins(
            mask,
            kernels=kernels,
            weights=weights,
            precomputed_kernels_f=kernels_f_c128.to(torch.complex64),
        )
        assert torch.allclose(out_c128, out_c64, atol=1e-5)

    def test_precomputed_kernels_f_count_mismatch_raises(self) -> None:
        from openlithohub._utils.hopkins import (
            HopkinsParams,
            compute_socs_kernels,
            simulate_aerial_image_hopkins,
        )

        H = 64  # noqa: N806
        params = HopkinsParams(num_kernels=4, pixel_size_nm=8.0)
        kernels, weights = compute_socs_kernels(params, H, torch.device("cpu"))

        wrong = torch.zeros((3, H, H), dtype=torch.complex64)
        with pytest.raises(ValueError, match="precomputed_kernels_f has K="):
            simulate_aerial_image_hopkins(
                torch.zeros(H, H),
                kernels=kernels,
                weights=weights,
                precomputed_kernels_f=wrong,
            )
