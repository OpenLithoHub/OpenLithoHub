"""Tests for conformal coverage gate for stochastic ILT acceptance."""
from __future__ import annotations

import pytest
import torch

from openlithohub.benchmark.metrics.coverage_gate import (
    CoverageResult,
    ProcessWindowPlotter,
    StochasticAcceptanceGate,
    StochasticSampler,
    ThroughFocusCoverageCalibrator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_mask() -> torch.Tensor:
    mask = torch.zeros(32, 32, dtype=torch.float32)
    mask[:, 14:18] = 1.0
    return mask


@pytest.fixture
def simple_target() -> torch.Tensor:
    target = torch.zeros(32, 32, dtype=torch.float32)
    target[:, 14:18] = 1.0
    return target


@pytest.fixture
def sampler() -> StochasticSampler:
    return StochasticSampler(dose_photons_per_nm2=30.0, pixel_size_nm=1.0)


# ---------------------------------------------------------------------------
# StochasticSampler
# ---------------------------------------------------------------------------


class TestStochasticSampler:
    def test_sample_epe_shape(self, sampler: StochasticSampler, simple_mask: torch.Tensor):
        torch.manual_seed(42)
        epe = sampler.sample_epe(simple_mask, n_samples=25)
        assert epe.shape == (25,)

    def test_sample_epe_nonnegative(self, sampler: StochasticSampler, simple_mask: torch.Tensor):
        torch.manual_seed(42)
        epe = sampler.sample_epe(simple_mask, n_samples=10)
        assert (epe >= 0).all()

    def test_sample_lcdu_shape(self, sampler: StochasticSampler, simple_mask: torch.Tensor):
        torch.manual_seed(42)
        lcdu = sampler.sample_lcdu(simple_mask, n_samples=30)
        assert lcdu.shape == (30,)

    def test_sample_lcdu_nonnegative(self, sampler: StochasticSampler, simple_mask: torch.Tensor):
        torch.manual_seed(42)
        lcdu = sampler.sample_lcdu(simple_mask, n_samples=10)
        assert (lcdu >= 0).all()

    def test_sample_epe_deterministic_with_seed(
        self, sampler: StochasticSampler, simple_mask: torch.Tensor
    ):
        torch.manual_seed(99)
        epe1 = sampler.sample_epe(simple_mask, n_samples=5)
        torch.manual_seed(99)
        epe2 = sampler.sample_epe(simple_mask, n_samples=5)
        assert torch.allclose(epe1, epe2)


# ---------------------------------------------------------------------------
# ThroughFocusCoverageCalibrator
# ---------------------------------------------------------------------------


class TestThroughFocusCoverageCalibrator:
    def test_calibrate_and_predict(
        self,
        sampler: StochasticSampler,
        simple_mask: torch.Tensor,
        simple_target: torch.Tensor,
    ):
        torch.manual_seed(42)
        calibrator = ThroughFocusCoverageCalibrator(
            sampler=sampler,
            n_calibration_samples=10,
            n_focus_points=3,
            focus_range_nm=(-20.0, 20.0),
        )
        calibrator.calibrate([simple_mask], [simple_target], alpha=0.1)
        result = calibrator.predict_coverage(simple_mask, simple_target, alpha=0.1)
        assert isinstance(result, CoverageResult)
        assert result.epe_band[0] <= result.epe_band[1]
        assert result.lcdu_band[0] <= result.lcdu_band[1]
        assert 0.0 <= result.empirical_coverage <= 1.0
        assert result.target_coverage == 0.9

    def test_predict_without_calibrate_raises(
        self,
        sampler: StochasticSampler,
        simple_mask: torch.Tensor,
        simple_target: torch.Tensor,
    ):
        calibrator = ThroughFocusCoverageCalibrator(sampler=sampler)
        with pytest.raises(RuntimeError, match="calibrate"):
            calibrator.predict_coverage(simple_mask, simple_target)

    def test_mismatched_masks_targets_raises(
        self,
        sampler: StochasticSampler,
        simple_mask: torch.Tensor,
        simple_target: torch.Tensor,
    ):
        calibrator = ThroughFocusCoverageCalibrator(sampler=sampler)
        with pytest.raises(ValueError, match="same length"):
            calibrator.calibrate([simple_mask], [simple_target, simple_target])

    def test_process_window_populated(
        self,
        sampler: StochasticSampler,
        simple_mask: torch.Tensor,
        simple_target: torch.Tensor,
    ):
        torch.manual_seed(42)
        calibrator = ThroughFocusCoverageCalibrator(
            sampler=sampler,
            n_calibration_samples=5,
            n_focus_points=2,
            focus_range_nm=(-10.0, 10.0),
        )
        calibrator.calibrate([simple_mask], [simple_target], alpha=0.1)
        result = calibrator.predict_coverage(simple_mask, simple_target, alpha=0.1)
        assert len(result.process_window) > 0
        assert all(isinstance(v, bool) for v in result.process_window.values())


# ---------------------------------------------------------------------------
# CoverageResult
# ---------------------------------------------------------------------------


class TestCoverageResult:
    def test_fields(self):
        result = CoverageResult(
            epe_band=(0.1, 0.5),
            lcdu_band=(0.0, 0.3),
            empirical_coverage=0.92,
            target_coverage=0.9,
            process_window={"f=0.0_d=1.00": True},
        )
        assert result.epe_band == (0.1, 0.5)
        assert result.lcdu_band == (0.0, 0.3)
        assert result.empirical_coverage == 0.92
        assert result.target_coverage == 0.9
        assert result.process_window["f=0.0_d=1.00"] is True

    def test_frozen(self):
        result = CoverageResult(
            epe_band=(0.0, 1.0),
            lcdu_band=(0.0, 1.0),
            empirical_coverage=0.95,
            target_coverage=0.9,
            process_window={},
        )
        with pytest.raises(AttributeError):
            result.empirical_coverage = 0.5


# ---------------------------------------------------------------------------
# StochasticAcceptanceGate
# ---------------------------------------------------------------------------


class TestStochasticAcceptanceGate:
    def _calibrated_gate(
        self, sampler: StochasticSampler, mask: torch.Tensor, target: torch.Tensor
    ) -> StochasticAcceptanceGate:
        calibrator = ThroughFocusCoverageCalibrator(
            sampler=sampler,
            n_calibration_samples=5,
            n_focus_points=2,
            focus_range_nm=(-10.0, 10.0),
        )
        calibrator.calibrate([mask], [target], alpha=0.1)
        return StochasticAcceptanceGate(calibrator=calibrator)

    def test_accepts_when_within_spec(
        self,
        sampler: StochasticSampler,
        simple_mask: torch.Tensor,
        simple_target: torch.Tensor,
    ):
        torch.manual_seed(42)
        gate = self._calibrated_gate(sampler, simple_mask, simple_target)
        accepted, result, reasons = gate.evaluate(
            simple_mask,
            simple_target,
            min_coverage=0.0,
            max_epe_band=(0.0, 1000.0),
            max_lcdu_band=(0.0, 1000.0),
        )
        assert accepted is True
        assert reasons == []

    def test_rejects_when_epe_exceeds_spec(
        self,
        sampler: StochasticSampler,
        simple_mask: torch.Tensor,
        simple_target: torch.Tensor,
    ):
        torch.manual_seed(42)
        gate = self._calibrated_gate(sampler, simple_mask, simple_target)
        accepted, result, reasons = gate.evaluate(
            simple_mask,
            simple_target,
            min_coverage=0.0,
            max_epe_band=(0.0, 1e-9),
            max_lcdu_band=None,
        )
        assert accepted is False
        assert any("EPE" in r for r in reasons)

    def test_rejects_when_lcdu_exceeds_spec(
        self,
        sampler: StochasticSampler,
        simple_mask: torch.Tensor,
        simple_target: torch.Tensor,
    ):
        torch.manual_seed(42)
        gate = self._calibrated_gate(sampler, simple_mask, simple_target)
        accepted, result, reasons = gate.evaluate(
            simple_mask,
            simple_target,
            min_coverage=0.0,
            max_epe_band=None,
            max_lcdu_band=(0.0, 1e-9),
        )
        assert accepted is False
        assert any("LCDU" in r for r in reasons)


# ---------------------------------------------------------------------------
# ProcessWindowPlotter
# ---------------------------------------------------------------------------


class TestProcessWindowPlotter:
    def test_compute_window_shape(
        self,
        sampler: StochasticSampler,
        simple_mask: torch.Tensor,
        simple_target: torch.Tensor,
    ):
        torch.manual_seed(42)
        plotter = ProcessWindowPlotter(sampler=sampler, epe_tolerance=5.0)
        window = plotter.compute_window(
            simple_mask,
            simple_target,
            focus_range=(-20.0, 20.0),
            dose_range=(0.95, 1.05),
            n_grid=5,
            n_samples_per_point=3,
        )
        assert window.shape == (5, 5)
        assert window.dtype == torch.bool

    def test_window_area_in_range(
        self,
        sampler: StochasticSampler,
        simple_mask: torch.Tensor,
        simple_target: torch.Tensor,
    ):
        torch.manual_seed(42)
        plotter = ProcessWindowPlotter(sampler=sampler, epe_tolerance=5.0)
        plotter.compute_window(
            simple_mask,
            simple_target,
            focus_range=(-20.0, 20.0),
            dose_range=(0.95, 1.05),
            n_grid=4,
            n_samples_per_point=3,
        )
        area = plotter.window_area()
        assert 0.0 <= area <= 1.0

    def test_window_area_before_compute_raises(self, sampler: StochasticSampler):
        plotter = ProcessWindowPlotter(sampler=sampler)
        with pytest.raises(RuntimeError, match="compute_window"):
            plotter.window_area()


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_sample_calibrate_evaluate_pipeline(
        self, simple_mask: torch.Tensor, simple_target: torch.Tensor
    ):
        torch.manual_seed(42)
        sampler = StochasticSampler(dose_photons_per_nm2=30.0)
        calibrator = ThroughFocusCoverageCalibrator(
            sampler=sampler,
            n_calibration_samples=5,
            n_focus_points=2,
            focus_range_nm=(-10.0, 10.0),
        )
        calibrator.calibrate([simple_mask], [simple_target], alpha=0.1)

        gate = StochasticAcceptanceGate(calibrator=calibrator)
        accepted, result, reasons = gate.evaluate(
            simple_mask, simple_target, min_coverage=0.5
        )

        assert isinstance(accepted, bool)
        assert isinstance(result, CoverageResult)
        assert isinstance(reasons, list)
        assert result.epe_band[0] <= result.epe_band[1]
        assert result.lcdu_band[0] <= result.lcdu_band[1]
