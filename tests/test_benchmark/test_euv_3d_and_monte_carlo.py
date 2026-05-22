"""Tests for EUV 3D-mask and Monte Carlo metrics."""

from __future__ import annotations

import pytest
import torch

from openlithohub.benchmark import (
    Mask3DParams,
    apply_3d_shadow,
    compute_3d_mask_residual,
    monte_carlo_failure_probability,
)
from openlithohub.simulators import HopkinsSimulator, SimulatorConfig


def _checkerboard_mask(size: int = 64) -> torch.Tensor:
    mask = torch.zeros(size, size)
    for r in range(0, size, 8):
        mask[r : r + 4, ::2] = 1.0
    return mask


class TestEuv3D:
    def test_apply_3d_shadow_preserves_shape_and_range(self) -> None:
        mask = _checkerboard_mask()
        out = apply_3d_shadow(mask)
        assert out.shape == mask.shape
        assert (out >= 0).all() and (out <= 1).all()

    def test_zero_thickness_is_identity(self) -> None:
        mask = _checkerboard_mask()
        params = Mask3DParams(absorber_thickness_nm=0.0, chief_ray_angle_deg=0.0)
        out = apply_3d_shadow(mask, params)
        assert torch.allclose(out, mask, atol=1e-5)

    def test_residual_grows_with_thickness(self) -> None:
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0
        cfg = SimulatorConfig(wavelength_nm=13.5, na=0.33, sigma=0.7, pixel_size_nm=1.0)
        thin = compute_3d_mask_residual(
            mask,
            Mask3DParams(absorber_thickness_nm=10.0, pixel_size_nm=1.0),
            cfg,
        )
        thick = compute_3d_mask_residual(
            mask,
            Mask3DParams(absorber_thickness_nm=200.0, pixel_size_nm=1.0),
            cfg,
        )
        assert thick["residual_l2"] > thin["residual_l2"]

    def test_residual_keys(self) -> None:
        mask = _checkerboard_mask()
        out = compute_3d_mask_residual(mask)
        assert {"residual_l2", "residual_linf", "hv_bias_nm"} == set(out)
        assert out["residual_l2"] >= 0

    def test_hv_bias_responds_to_horizontal_line_array(self) -> None:
        """Issue #24: hv_bias_nm should be in nanometres and reflect the
        H-vs-V printed-CD difference under the chief-ray shadow. Build a
        mask with horizontal-only stripes — its V-azimuth shadow
        narrows the lines along their length axis, the H-azimuth shadow
        leaves them untouched, so the H-azimuth printed contour is
        wider than the V-azimuth one. Sign: positive (H wider)."""
        # Horizontal stripes: rows of foreground separated by rows of bg.
        mask = torch.zeros(64, 64)
        for r in range(8, 64, 16):
            mask[r : r + 4, :] = 1.0
        out = compute_3d_mask_residual(
            mask,
            Mask3DParams(absorber_thickness_nm=200.0, chief_ray_angle_deg=10.0),
        )
        # Strong shadow → measurable, signed bias in nm.
        assert isinstance(out["hv_bias_nm"], float)
        # H-stripes shadowed along the H-azimuth (along the line) preserve
        # CD; shadowed along the V-azimuth (across the line) shrink CD —
        # so H_width > V_width and the bias is positive.
        assert out["hv_bias_nm"] > 0.0

    def test_hv_bias_zero_for_zero_thickness(self) -> None:
        """No absorber, no shadow, so H ≡ V and bias = 0."""
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0
        out = compute_3d_mask_residual(
            mask,
            Mask3DParams(absorber_thickness_nm=0.0, chief_ray_angle_deg=0.0),
        )
        assert out["hv_bias_nm"] == pytest.approx(0.0, abs=1e-9)


class TestMonteCarloFailure:
    def test_zero_jitter_yields_zero_failure(self) -> None:
        mask = _checkerboard_mask()
        sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
        result = monte_carlo_failure_probability(
            mask,
            sim,
            num_trials=5,
            dose_jitter_sigma=0.0,
            threshold_jitter_sigma=0.0,
            seed=0,
        )
        assert result.failure_probability == 0.0
        assert result.num_trials == 5

    def test_jitter_runs_and_returns_in_unit_interval(self) -> None:
        mask = _checkerboard_mask()
        sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
        result = monte_carlo_failure_probability(
            mask,
            sim,
            num_trials=4,
            dose_jitter_sigma=0.05,
            threshold_jitter_sigma=0.02,
            seed=42,
        )
        assert 0.0 <= result.bridge_probability <= 1.0
        assert 0.0 <= result.break_probability <= 1.0
        # failure_probability is the union of bridge + break (issue #55:
        # a trial with both gets counted on both axes but contributes
        # one failure), so it can never exceed 1.0.
        assert 0.0 <= result.failure_probability <= 1.0

    def test_simultaneous_bridge_and_break_both_recorded(self) -> None:
        """Issue #55: a trial that bridges one component pair AND breaks a
        third component must register on both axes — the prior net
        component-count delta would silently classify it as a no-op.
        Synthetic test exercising the bridge/break detector directly:
        nominal has 3 disjoint blobs; trial merges two and splits the
        third. Net component count is unchanged but both events occur.
        """
        from openlithohub.benchmark.metrics.monte_carlo import _bridge_and_break_versus

        nominal = torch.zeros(20, 30)
        nominal[2:8, 2:8] = 1
        nominal[2:8, 12:18] = 1
        nominal[2:8, 22:28] = 1

        trial = torch.zeros(20, 30)
        # First two blobs merge (bridge)
        trial[2:8, 2:18] = 1
        # Third blob splits into two (break)
        trial[2:8, 22:24] = 1
        trial[2:8, 26:28] = 1

        has_bridge, has_break = _bridge_and_break_versus(nominal, trial)
        assert has_bridge
        assert has_break

    def test_simulator_config_restored_after_run(self) -> None:
        mask = _checkerboard_mask()
        cfg = SimulatorConfig(pixel_size_nm=4.0, dose=1.0, threshold=0.225)
        sim = HopkinsSimulator(cfg)
        monte_carlo_failure_probability(
            mask,
            sim,
            num_trials=3,
            dose_jitter_sigma=0.1,
            threshold_jitter_sigma=0.1,
            seed=7,
        )
        assert sim.config.dose == pytest.approx(1.0)
        assert sim.config.threshold == pytest.approx(0.225)
