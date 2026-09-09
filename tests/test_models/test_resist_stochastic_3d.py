"""Tests for the 3D resist stochastic model with coverage gate."""

from __future__ import annotations

import pytest
import torch

from openlithohub.models.resist_stochastic_3d import (
    ConformalCoverageGate3D,
    ResistProfile3D,
    SecondaryElectronKernel,
    Stochastic3DBenchmark,
    StochasticDefectModel3D,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def se_kernel() -> SecondaryElectronKernel:
    return SecondaryElectronKernel(correlation_length_nm=2.0, kernel_type="gaussian")


@pytest.fixture
def resist() -> ResistProfile3D:
    return ResistProfile3D(thickness_nm=5.0, pixel_size_nm=1.0)


@pytest.fixture
def line_mask() -> torch.Tensor:
    """32x32 dense line/space pattern with 4px pitch."""
    m = torch.zeros(32, 32)
    for x in range(0, 32, 4):
        m[:, x : x + 2] = 1.0
    return m


@pytest.fixture
def small_mask() -> torch.Tensor:
    """16x16 simple pattern for fast tests."""
    m = torch.zeros(16, 16)
    m[:, 4:8] = 1.0
    m[:, 10:14] = 1.0
    return m


# ---------------------------------------------------------------------------
# SecondaryElectronKernel tests
# ---------------------------------------------------------------------------


class TestSecondaryElectronKernel:
    def test_se_kernel_shape(self, se_kernel: SecondaryElectronKernel) -> None:
        kernel = se_kernel.compute_kernel(5, pixel_size_nm=1.0)
        assert kernel.shape == (5, 5, 5)
        assert kernel.dtype == torch.float32

    def test_se_kernel_normalised(self, se_kernel: SecondaryElectronKernel) -> None:
        kernel = se_kernel.compute_kernel(5, pixel_size_nm=1.0)
        assert kernel.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_se_kernel_correlation(self) -> None:
        kernel = SecondaryElectronKernel(correlation_length_nm=10.0, kernel_type="gaussian")
        k = kernel.compute_kernel(7, pixel_size_nm=1.0)
        # Center should be peak
        center = k[3, 3, 3].item()
        corner = k[0, 0, 0].item()
        assert center > corner

    def test_apply_correlation_output_shape(self, se_kernel: SecondaryElectronKernel) -> None:
        noise = torch.randn(5, 16, 16)
        result = se_kernel.apply_correlation(noise)
        assert result.shape == noise.shape

    def test_exponential_kernel(self) -> None:
        kernel = SecondaryElectronKernel(correlation_length_nm=3.0, kernel_type="exponential")
        k = kernel.compute_kernel(3, pixel_size_nm=1.0)
        assert k.shape == (3, 3, 3)
        assert k.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_invalid_kernel_type_raises(self) -> None:
        with pytest.raises(ValueError, match="kernel_type"):
            SecondaryElectronKernel(kernel_type="invalid")

    def test_invalid_correlation_length_raises(self) -> None:
        with pytest.raises(ValueError, match="correlation_length_nm"):
            SecondaryElectronKernel(correlation_length_nm=-1.0)


# ---------------------------------------------------------------------------
# ResistProfile3D tests
# ---------------------------------------------------------------------------


class TestResistProfile3D:
    def test_resist_profile_3d_shape(
        self,
        resist: ResistProfile3D,
        line_mask: torch.Tensor,
    ) -> None:
        profile = resist.generate_profile(line_mask)
        assert profile.ndim == 3
        assert profile.shape[0] == resist.n_slices
        assert profile.shape[1:] == line_mask.shape

    def test_resist_profile_binary_values(
        self,
        resist: ResistProfile3D,
        line_mask: torch.Tensor,
    ) -> None:
        profile = resist.generate_profile(line_mask)
        unique = torch.unique(profile)
        assert unique.numel() <= 2
        for v in unique.tolist():
            assert v in (0.0, 1.0)

    def test_line_collapse_risk(
        self,
        resist: ResistProfile3D,
        line_mask: torch.Tensor,
    ) -> None:
        profile = resist.generate_profile(line_mask)
        risk = resist.compute_line_collapse_risk(profile)
        assert 0.0 <= risk <= 1.0

    def test_line_collapse_risk_thin_lines_higher(self) -> None:
        """Thinner lines should have higher (or equal) collapse risk."""
        resist = ResistProfile3D(thickness_nm=20.0, pixel_size_nm=1.0)

        thick_mask = torch.zeros(32, 32)
        thick_mask[:, 10:20] = 1.0  # 10px CD

        thin_mask = torch.zeros(32, 32)
        thin_mask[:, 15:17] = 1.0  # 2px CD

        profile_thick = resist.generate_profile(thick_mask)
        profile_thin = resist.generate_profile(thin_mask)

        risk_thick = resist.compute_line_collapse_risk(profile_thick)
        risk_thin = resist.compute_line_collapse_risk(profile_thin)

        assert risk_thin >= risk_thick

    def test_lcdu_3d(self, resist: ResistProfile3D, line_mask: torch.Tensor) -> None:
        profile = resist.generate_profile(line_mask)
        lcdu = resist.compute_lcdu_3d(profile)
        assert lcdu.shape == (resist.n_slices,)
        assert torch.isfinite(lcdu).all()

    def test_vertical_correlation(
        self,
        resist: ResistProfile3D,
        line_mask: torch.Tensor,
    ) -> None:
        profile = resist.generate_profile(line_mask)
        corr = resist.vertical_correlation(profile)
        assert corr.shape == (resist.n_slices - 1,)
        for c in corr.tolist():
            assert -1.0 <= c <= 1.0

    def test_invalid_thickness_raises(self) -> None:
        with pytest.raises(ValueError, match="thickness_nm"):
            ResistProfile3D(thickness_nm=-5.0)


# ---------------------------------------------------------------------------
# StochasticDefectModel3D tests
# ---------------------------------------------------------------------------


class TestStochasticDefectModel3D:
    def test_stochastic_defect_model_runs(
        self,
        se_kernel: SecondaryElectronKernel,
        small_mask: torch.Tensor,
    ) -> None:
        resist = ResistProfile3D(thickness_nm=3.0, pixel_size_nm=1.0)
        model = StochasticDefectModel3D(
            se_kernel=se_kernel,
            resist=resist,
            dose_photons=30.0,
        )
        defect_map = model.simulate_defects(small_mask, n_mc=5, seed=0)
        assert defect_map.ndim == 3
        assert defect_map.shape[0] == resist.n_slices
        assert (defect_map >= 0.0).all()
        assert (defect_map <= 1.0).all()

    def test_defect_map_deterministic(
        self,
        se_kernel: SecondaryElectronKernel,
        small_mask: torch.Tensor,
    ) -> None:
        resist = ResistProfile3D(thickness_nm=3.0, pixel_size_nm=1.0)
        model = StochasticDefectModel3D(se_kernel=se_kernel, resist=resist)
        d1 = model.simulate_defects(small_mask, n_mc=3, seed=42)
        d2 = model.simulate_defects(small_mask, n_mc=3, seed=42)
        assert torch.equal(d1, d2)

    def test_failure_rate_3d(
        self,
        se_kernel: SecondaryElectronKernel,
        small_mask: torch.Tensor,
    ) -> None:
        resist = ResistProfile3D(thickness_nm=3.0, pixel_size_nm=1.0)
        model = StochasticDefectModel3D(se_kernel=se_kernel, resist=resist)
        rates = model.compute_failure_rate_3d(small_mask, n_mc=5, seed=0)
        assert rates.shape == (resist.n_slices,)
        assert (rates >= 0.0).all()
        assert (rates <= 1.0).all()

    def test_through_focus_quantile(
        self,
        se_kernel: SecondaryElectronKernel,
        small_mask: torch.Tensor,
    ) -> None:
        resist = ResistProfile3D(thickness_nm=3.0, pixel_size_nm=1.0)
        model = StochasticDefectModel3D(se_kernel=se_kernel, resist=resist)
        result = model.through_focus_quantile(
            small_mask,
            focus_range=(-10.0, 10.0, 10.0),
            alpha=0.1,
            n_mc=3,
        )
        assert "focus_values" in result
        assert "quantile_defects" in result
        assert "mean_defects" in result
        assert result["quantile_defects"].shape == result["focus_values"].shape
        assert (result["quantile_defects"] >= 0.0).all()


# ---------------------------------------------------------------------------
# ConformalCoverageGate3D tests
# ---------------------------------------------------------------------------


class TestConformalCoverageGate3D:
    def test_conformal_coverage_gate(self) -> None:
        gate = ConformalCoverageGate3D(target_coverage=0.9, alpha=0.1)

        # Create calibration data
        masks = []
        gt_defects = []
        for i in range(10):
            m = torch.zeros(8, 8)
            m[:, 2:6] = 1.0
            masks.append(m)
            gt = torch.full((8, 8), 0.02 + 0.01 * i / 10.0)
            gt_defects.append(gt)

        gate.calibrate(masks, gt_defects)
        assert gate._predictor is not None
        assert gate._gate is not None

        # Evaluate
        test_mask = torch.zeros(8, 8)
        test_mask[:, 2:6] = 1.0
        predicted = torch.full((8, 8), 0.03)
        actual = torch.full((8, 8), 0.025)

        verdict, metrics = gate.evaluate(test_mask, predicted, actual)
        assert verdict.name in ("ACCEPT", "REJECT", "UNCERTAIN")
        assert 0.0 <= metrics.coverage <= 1.0
        assert metrics.mean_bandwidth >= 0.0
        assert isinstance(metrics.n_violations, int)

    def test_coverage_gate_no_calibration(self) -> None:
        gate = ConformalCoverageGate3D(target_coverage=0.9, alpha=0.1)
        mask = torch.zeros(8, 8)
        mask[:, 2:6] = 1.0
        predicted = torch.full((8, 8), 0.05)
        actual = torch.full((8, 8), 0.04)
        verdict, metrics = gate.evaluate(mask, predicted, actual)
        assert verdict.name in ("ACCEPT", "REJECT", "UNCERTAIN")

    def test_invalid_coverage_raises(self) -> None:
        with pytest.raises(ValueError, match="target_coverage"):
            ConformalCoverageGate3D(target_coverage=1.5)

    def test_process_window_3d(self, small_mask: torch.Tensor) -> None:
        gate = ConformalCoverageGate3D(target_coverage=0.9, alpha=0.1)
        result = gate.process_window_3d(
            small_mask,
            focus_range=(-10.0, 10.0, 10.0),
            dose_range=(25.0, 35.0, 10.0),
            n_mc=3,
        )
        assert "defect_grid" in result
        assert "coverage_grid" in result
        assert "focus_values" in result
        assert "dose_values" in result
        n_f = len(result["focus_values"])
        n_d = len(result["dose_values"])
        assert result["defect_grid"].shape == (n_f, n_d)
        assert result["coverage_grid"].shape == (n_f, n_d)


# ---------------------------------------------------------------------------
# Stochastic3DBenchmark tests
# ---------------------------------------------------------------------------


class TestStochastic3DBenchmark:
    def test_benchmark_runs(self) -> None:
        se = SecondaryElectronKernel(correlation_length_nm=2.0)
        resist = ResistProfile3D(thickness_nm=3.0, pixel_size_nm=1.0)
        benchmark = Stochastic3DBenchmark(
            se_kernel=se,
            resist=resist,
            dose_photons=30.0,
        )
        results = benchmark.run(n_seeds=1)
        assert len(results) > 0
        for r in results:
            assert 0.0 <= r.model_3d_defect_rate <= 1.0
            assert 0.0 <= r.model_2d_defect_rate <= 1.0
            assert isinstance(r.defect_rate_diff, float)
            assert 0.0 <= r.model_3d_line_collapse_risk <= 1.0
            assert isinstance(r.monotonicity_check, bool)
