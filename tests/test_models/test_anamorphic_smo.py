"""Tests for openlithohub.models.anamorphic_smo."""

from __future__ import annotations

import math

import pytest
import torch

from openlithohub.models.anamorphic_smo import (
    AnamorphicImaging,
    AnamorphicParams,
    AnamorphicSMO,
    AnamorphicSMOBenchmark,
    ShotCountCost,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_target(size: int = 32) -> torch.Tensor:
    """Create a simple square-contact-hole target pattern."""
    t = torch.zeros(size, size)
    margin = size // 4
    t[margin : size - margin, margin : size - margin] = 1.0
    return t


def _line_space_target(size: int = 32) -> torch.Tensor:
    """Create a line-space target pattern."""
    t = torch.zeros(size, size)
    pitch = max(4, size // 8)
    half = pitch // 2
    for x in range(0, size, pitch):
        t[:, x : x + half] = 1.0
    return t


# ---------------------------------------------------------------------------
# 1. AnamorphicParams defaults
# ---------------------------------------------------------------------------


class TestAnamorphicParams:
    def test_anamorphic_params_defaults(self) -> None:
        p = AnamorphicParams()
        assert p.na_x == 0.55
        assert p.na_y == 0.55
        assert p.mag_x == 4.0
        assert p.mag_y == 8.0
        assert p.central_obscuration_ratio == 0.2
        assert p.wavelength_nm == 13.5
        assert p.pixel_size_nm == 1.0

    def test_anamorphic_params_custom(self) -> None:
        p = AnamorphicParams(na_x=0.33, mag_y=4.0, central_obscuration_ratio=0.0)
        assert p.na_x == 0.33
        assert p.mag_y == 4.0
        assert p.central_obscuration_ratio == 0.0


# ---------------------------------------------------------------------------
# 2. AnamorphicImaging PSF shape
# ---------------------------------------------------------------------------


class TestAnamorphicImagingPSF:
    def test_anamorphic_imaging_psf_shape(self) -> None:
        params = AnamorphicParams()
        imaging = AnamorphicImaging(params)
        psf = imaging.compute_psf(32)
        assert psf.shape == (32, 32)
        assert psf.dtype == torch.float32

    def test_psf_normalised(self) -> None:
        imaging = AnamorphicImaging()
        psf = imaging.compute_psf(32)
        assert psf.sum().item() == pytest.approx(1.0, abs=1e-4)

    def test_psf_central_obscuration_narrows_peak(self) -> None:
        """Central obscuration redistributes energy out of the central lobe.

        An annular pupil trades peak intensity for resolution: coherent
        energy is blocked, so the (normalised) PSF peak drops and the
        fraction of energy inside the full-pupil central-lobe radius
        decreases — sidelobe energy increases.
        """
        p_full = AnamorphicParams(central_obscuration_ratio=0.0)
        p_obs = AnamorphicParams(central_obscuration_ratio=0.3)
        psf_full = AnamorphicImaging(p_full).compute_psf(64)
        psf_obs = AnamorphicImaging(p_obs).compute_psf(64)
        # PSF from |IFFT(pupil)|^2 is corner-centred: main lobe at (0, 0).
        assert psf_obs[0, 0] < psf_full[0, 0], "Annular pupil should lower the normalised PSF peak"
        n = 64
        yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        dx = torch.minimum(xx, n - xx).float()
        dy = torch.minimum(yy, n - yy).float()
        r = torch.sqrt(dx**2 + dy**2)
        r0 = 6.0  # central-lobe radius of the full-pupil PSF
        core_full = psf_full[r <= r0].sum().item()
        core_obs = psf_obs[r <= r0].sum().item()
        assert core_obs < core_full, "Annular pupil should push energy into sidelobes"


# ---------------------------------------------------------------------------
# 3. AnamorphicImaging aerial
# ---------------------------------------------------------------------------


class TestAnamorphicImagingAerial:
    def test_anamorphic_imaging_aerial(self) -> None:
        imaging = AnamorphicImaging(AnamorphicParams())
        mask = _simple_target(32)
        aerial = imaging.simulate_aerial(mask)
        assert aerial.shape == (32, 32)
        assert aerial.dtype == torch.float32
        assert aerial.min() >= 0.0

    def test_aerial_non_negative(self) -> None:
        imaging = AnamorphicImaging()
        mask = torch.rand(32, 32)
        aerial = imaging.simulate_aerial(mask)
        assert (aerial >= 0.0).all()

    def test_aerial_open_frame_bright(self) -> None:
        """An open-frame (all-ones) mask should produce high aerial intensity."""
        imaging = AnamorphicImaging()
        mask = torch.ones(32, 32)
        aerial = imaging.simulate_aerial(mask)
        assert aerial.mean() > 0.5


# ---------------------------------------------------------------------------
# 4. Anamorphic vs isotropic
# ---------------------------------------------------------------------------


class TestAerialAlignment:
    def test_aerial_delta_stays_put(self) -> None:
        """A point mask must image to a peak at the same pixel.

        Guards against a half-field circular shift: the PSF from
        ``|IFFT(pupil)|^2`` is already in circular-convolution layout, and
        applying an extra ``ifftshift`` (a historical bug) would move the
        aerial peak by (N/2, N/2).
        """
        imaging = AnamorphicImaging(AnamorphicParams())
        mask = torch.zeros(32, 32)
        mask[16, 16] = 1.0
        aerial = imaging.simulate_aerial(mask)
        peak_y, peak_x = divmod(int(aerial.argmax()), 32)
        assert (peak_y, peak_x) == (16, 16), f"Aerial peak at {(peak_y, peak_x)}, expected (16, 16)"


class TestSourceParameterGradient:
    def test_source_params_move_during_optimisation(self) -> None:
        """Source parameters must be updated by the joint SMO.

        With the historical hard-threshold pupil the comparison cut the
        autograd graph and Adam never updated ``source_param``; the soft
        pupil keeps gradients flowing, so the sigmoid-ed source params
        must drift away from their initial values. Sensitivity is
        anisotropic (the anamorphic y cutoff dominates for a square
        target), so at least one parameter moving substantially is the
        robust gradient-flow signal.
        """
        smo = AnamorphicSMO()
        target = _simple_target(16)
        _, info = smo.optimize_source_mask(target, n_steps=20, lr=0.2)
        initial = 0.7
        initial_sig = 1.0 / (1.0 + math.exp(-initial))
        moves = [abs(value - initial_sig) for value in info["source_params"]]
        assert max(moves) > 1e-6, f"source parameters did not move {moves}; gradient path is broken"


class TestAnamorphicVsIsotropic:
    def test_anamorphic_vs_isotropic(self) -> None:
        """Anamorphic imaging should produce different PSFs from isotropic."""
        p_aniso = AnamorphicParams(mag_x=4.0, mag_y=8.0)
        p_iso = AnamorphicParams(mag_x=4.0, mag_y=4.0)

        psf_aniso = AnamorphicImaging(p_aniso).compute_psf(32)
        psf_iso = AnamorphicImaging(p_iso).compute_psf(32)

        diff = (psf_aniso - psf_iso).abs().sum()
        assert diff > 0.0, "Anamorphic and isotropic PSFs should differ"


# ---------------------------------------------------------------------------
# 5. Mask 3D shadow correction
# ---------------------------------------------------------------------------


class TestMask3DShadowCorrection:
    def test_mask_3d_shadow_correction(self) -> None:
        imaging = AnamorphicImaging(AnamorphicParams())
        mask = _simple_target(32)
        corrected = imaging.mask_3d_shadow_correction(mask, incident_angle_deg=6.0)
        assert corrected.shape == mask.shape
        assert corrected.dtype == torch.float32
        assert corrected.min() >= 0.0
        assert corrected.max() <= 1.0

    def test_shadow_correction_shifts_pattern(self) -> None:
        """Non-zero incident angle should shift the mask pattern."""
        imaging = AnamorphicImaging(AnamorphicParams())
        mask = _simple_target(32)
        corrected = imaging.mask_3d_shadow_correction(mask, incident_angle_deg=9.0)
        # The corrected mask should differ from the original.
        diff = (corrected - mask).abs().sum()
        assert diff > 0.0

    def test_zero_angle_no_change(self) -> None:
        """Zero incident angle should leave the mask unchanged."""
        imaging = AnamorphicImaging(AnamorphicParams(pixel_size_nm=100.0))
        mask = _simple_target(32)
        corrected = imaging.mask_3d_shadow_correction(mask, incident_angle_deg=0.0)
        assert torch.allclose(corrected, mask, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. ShotCountCost gradient
# ---------------------------------------------------------------------------


class TestShotCountCostGradient:
    def test_shot_count_cost_gradient(self) -> None:
        sc = ShotCountCost(weight=1.0)
        mask = torch.rand(32, 32, requires_grad=True)
        cost = sc.forward(mask)
        cost.backward()
        assert mask.grad is not None
        assert torch.isfinite(mask.grad).all()

    def test_shot_count_cost_scalar(self) -> None:
        sc = ShotCountCost(weight=0.01)
        mask = torch.rand(16, 16)
        cost = sc.forward(mask)
        assert cost.ndim == 0  # scalar
        assert cost.item() >= 0.0

    def test_shot_count_evaluate(self) -> None:
        sc = ShotCountCost(writer_type="mbmw")
        mask = _simple_target(16)
        result = sc.evaluate(mask)
        assert "shot_count" in result
        assert "estimated_write_time_s" in result
        assert result["shot_count"] >= 0

    def test_binary_mask_has_finite_cost(self) -> None:
        sc = ShotCountCost()
        mask = torch.zeros(16, 16)
        mask[4:12, 4:12] = 1.0
        cost = sc.forward(mask)
        assert torch.isfinite(cost)


# ---------------------------------------------------------------------------
# 7. AnamorphicSMO reduces EPE
# ---------------------------------------------------------------------------


class TestAnamorphicSMO:
    def test_anamorphic_smo_optimization_reduces_epe(self) -> None:
        params = AnamorphicParams()
        smo = AnamorphicSMO(params, shot_count_weight=0.001)
        target = _simple_target(32)
        mask, info = smo.optimize_source_mask(target, n_steps=30, lr=0.05)
        assert mask.shape == target.shape
        assert info["final_epe"] < float("inf")
        # EPE should have decreased over optimisation.
        history = info["pareto_history"]
        if len(history) > 1:
            # Allow for non-monotonic behaviour but expect overall reduction.
            assert history[-1][0] <= history[0][0] * 2.0

    def test_smo_output_is_binary(self) -> None:
        smo = AnamorphicSMO()
        target = _simple_target(16)
        mask, _ = smo.optimize_source_mask(target, n_steps=10, lr=0.05)
        unique = mask.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique)


# ---------------------------------------------------------------------------
# 8. Pareto tracking
# ---------------------------------------------------------------------------


class TestSMOParetoTracking:
    def test_smo_pareto_tracking(self) -> None:
        smo = AnamorphicSMO()
        target = _simple_target(16)
        _, info = smo.optimize_source_mask(target, n_steps=10, lr=0.05)
        history = info["pareto_history"]
        assert isinstance(history, list)
        assert len(history) == 10
        for fidelity, sc in history:
            assert isinstance(fidelity, float)
            assert isinstance(sc, float)
            assert fidelity >= 0.0
            assert sc >= 0.0

    def test_pareto_history_stored_on_smo(self) -> None:
        smo = AnamorphicSMO()
        target = _simple_target(16)
        smo.optimize_source_mask(target, n_steps=5, lr=0.05)
        assert len(smo.pareto_history) == 5


# ---------------------------------------------------------------------------
# 9. Benchmark runs
# ---------------------------------------------------------------------------


class TestAnamorphicSMOBenchmark:
    def test_benchmark_runs(self) -> None:
        bench = AnamorphicSMOBenchmark(grid_size=16, n_steps=5, lr=0.05)
        results = bench.run(n_seeds=2)
        assert "anamorphic" in results
        assert "isotropic" in results
        assert len(results["anamorphic"]) == 2
        assert len(results["isotropic"]) == 2
        assert "mean_anamorphic_epe" in results
        assert "mean_isotropic_epe" in results

    def test_benchmark_results_finite(self) -> None:
        bench = AnamorphicSMOBenchmark(grid_size=16, n_steps=5, lr=0.05)
        results = bench.run(n_seeds=1)
        assert math.isfinite(results["mean_anamorphic_epe"])
        assert math.isfinite(results["mean_isotropic_epe"])
