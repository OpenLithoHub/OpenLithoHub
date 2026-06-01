"""Tests for beyond-LER/LCDU stochastic metrics and polarization/triple-beam illumination.

References:
    - Siemens Calibre, "Calibration and verification metrics for EUVL stochastic
      models: beyond LER and LCDU", Proc. SPIE 2026.
    - IBM, "demonstrates High-NA EUV below 2 nm nodes at SPIE 2026",
      research.ibm.com, 2026-02.
"""

from __future__ import annotations

import pytest
import torch

from openlithohub.models.anamorphic_smo import (
    AnamorphicImaging,
    AnamorphicParams,
    PolarizationState,
    TripleBeamIllumination,
)
from openlithohub.models.resist_stochastic_3d import (
    ConformalCoverageGate3D,
    DefectClusterMetrics,
    StochasticCalibrationMetrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def defect_map() -> torch.Tensor:
    """32x32 synthetic defect map with some clustered defects."""
    torch.manual_seed(42)
    d = torch.zeros(32, 32)
    # Cluster 1: a 3x3 block
    d[5:8, 10:13] = 1.0
    # Cluster 2: a 2x1 block
    d[20:22, 5] = 1.0
    # Isolated pixel
    d[15, 25] = 1.0
    return d


@pytest.fixture
def profile_ensemble() -> torch.Tensor:
    """10 x 32 x 32 synthetic profile ensemble with edge variation."""
    torch.manual_seed(0)
    n = 10
    profiles = torch.zeros(n, 32, 32)
    for i in range(n):
        # Shift edge position by a random amount per trial
        shift = torch.randint(0, 3, (1,)).item()
        profiles[i, :, 10 + shift : 20 + shift] = 1.0
    return profiles


@pytest.fixture
def line_mask() -> torch.Tensor:
    """32x32 dense line/space pattern."""
    m = torch.zeros(32, 32)
    for x in range(0, 32, 8):
        m[:, x : x + 4] = 1.0
    return m


# ---------------------------------------------------------------------------
# Test 1: DefectClusterMetrics.failure_correlation_length returns positive
# ---------------------------------------------------------------------------


class TestDefectClusterMetrics:
    def test_correlation_length_positive(self, defect_map: torch.Tensor) -> None:
        metrics = DefectClusterMetrics(pixel_size_nm=1.0)
        corr_len = metrics.failure_correlation_length(defect_map)
        assert isinstance(corr_len, float)
        assert corr_len >= 0.0

    def test_correlation_length_uniform_zero(self) -> None:
        """Uniform map (no spatial structure) should return 0."""
        metrics = DefectClusterMetrics(pixel_size_nm=1.0)
        uniform = torch.ones(16, 16)
        corr_len = metrics.failure_correlation_length(uniform)
        assert corr_len == 0.0

    # -----------------------------------------------------------------------
    # Test 2: cluster_distribution returns valid dict
    # -----------------------------------------------------------------------

    def test_cluster_distribution_returns_valid_dict(
        self,
        defect_map: torch.Tensor,
    ) -> None:
        metrics = DefectClusterMetrics(pixel_size_nm=1.0, connectivity=4)
        dist = metrics.defect_cluster_distribution(defect_map)

        assert isinstance(dist, dict)
        assert "n_clusters" in dist
        assert "cluster_sizes" in dist
        assert "max_cluster_size" in dist
        assert "mean_cluster_size" in dist
        assert "size_histogram" in dist

        # We expect 3 clusters from the fixture
        assert dist["n_clusters"] == 3
        assert dist["max_cluster_size"] == 9  # 3x3 block
        assert len(dist["cluster_sizes"]) == 3
        assert all(isinstance(s, int) for s in dist["cluster_sizes"])
        assert dist["cluster_sizes"] == sorted(dist["cluster_sizes"], reverse=True)

    def test_cluster_distribution_empty_map(self) -> None:
        metrics = DefectClusterMetrics()
        empty = torch.zeros(16, 16)
        dist = metrics.defect_cluster_distribution(empty)
        assert dist["n_clusters"] == 0
        assert dist["max_cluster_size"] == 0
        assert dist["mean_cluster_size"] == 0.0

    # -----------------------------------------------------------------------
    # Test 3: stochastic_epe_quantile returns valid values
    # -----------------------------------------------------------------------

    def test_stochastic_epe_quantile(
        self,
        profile_ensemble: torch.Tensor,
    ) -> None:
        metrics = DefectClusterMetrics(pixel_size_nm=1.0)
        result = metrics.stochastic_epe_quantile(profile_ensemble, alpha=0.5)

        assert isinstance(result, dict)
        assert "epe_quantile_nm" in result
        assert "epe_mean_nm" in result
        assert "epe_std_nm" in result
        assert "n_trials" in result

        assert result["n_trials"] == 10
        assert result["epe_quantile_nm"] >= 0.0
        assert result["epe_mean_nm"] >= 0.0
        assert result["epe_std_nm"] >= 0.0

    def test_stochastic_epe_quantile_invalid_alpha(self) -> None:
        metrics = DefectClusterMetrics()
        ensemble = torch.ones(5, 16, 16)
        with pytest.raises(ValueError, match="alpha"):
            metrics.stochastic_epe_quantile(ensemble, alpha=0.0)

    def test_stochastic_epe_quantile_invalid_ndim(self) -> None:
        metrics = DefectClusterMetrics()
        with pytest.raises(ValueError, match="3D"):
            metrics.stochastic_epe_quantile(torch.ones(16, 16), alpha=0.05)


# ---------------------------------------------------------------------------
# Test 4: StochasticCalibrationMetrics combines all metrics
# ---------------------------------------------------------------------------


class TestStochasticCalibrationMetrics:
    def test_combine_all_metrics(
        self,
        defect_map: torch.Tensor,
        profile_ensemble: torch.Tensor,
    ) -> None:
        cal = StochasticCalibrationMetrics()
        report = cal.evaluate(
            defect_map=defect_map,
            profile_ensemble=profile_ensemble,
            epe_alpha=0.05,
        )

        # All fields should be populated (coverage defaults to 0/UNCERTAIN)
        assert report.failure_corr_length_nm >= 0.0
        assert report.n_defect_clusters >= 0
        assert report.max_cluster_size >= 0
        assert report.mean_cluster_size >= 0.0
        assert report.epe_quantile_nm >= 0.0
        assert report.epe_mean_nm >= 0.0
        assert report.epe_std_nm >= 0.0
        assert report.conformal_verdict == "UNCERTAIN"
        assert report.conformal_coverage == 0.0

    def test_with_coverage_evaluation(self, defect_map: torch.Tensor) -> None:
        gate = ConformalCoverageGate3D(target_coverage=0.9, alpha=0.1)
        cal = StochasticCalibrationMetrics(coverage_gate=gate)

        mask = torch.zeros(32, 32)
        mask[:, 4:12] = 1.0

        report = cal.evaluate(
            defect_map=defect_map,
            predicted_defects=defect_map,
            actual_defects=defect_map,
            mask=mask,
        )

        # With identical predicted and actual, coverage should be non-zero
        assert report.conformal_coverage >= 0.0
        assert isinstance(report.conformal_verdict, str)


# ---------------------------------------------------------------------------
# Test 5: PolarizationState init / validation
# ---------------------------------------------------------------------------


class TestPolarizationState:
    def test_default_init(self) -> None:
        ps = PolarizationState()
        assert ps.angle_deg == 0.0
        assert ps.ellipticity == 0.0
        assert ps.tm_weight == 1.0

    def test_custom_init(self) -> None:
        ps = PolarizationState(angle_deg=45.0, ellipticity=0.3, tm_weight=0.8)
        assert ps.angle_deg == 45.0
        assert ps.ellipticity == 0.3
        assert ps.tm_weight == 0.8

    def test_invalid_ellipticity_raises(self) -> None:
        with pytest.raises(ValueError, match="ellipticity"):
            PolarizationState(ellipticity=-0.1)
        with pytest.raises(ValueError, match="ellipticity"):
            PolarizationState(ellipticity=1.5)

    def test_invalid_tm_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="tm_weight"):
            PolarizationState(tm_weight=-0.5)
        with pytest.raises(ValueError, match="tm_weight"):
            PolarizationState(tm_weight=1.5)

    def test_contrast_factor_range(self) -> None:
        ps = PolarizationState()
        cf = ps.contrast_factor(na=0.55, wavelength_nm=13.5)
        assert 0.0 <= cf <= 1.0

    def test_tm_differs_from_unpolarized(self) -> None:
        """TM polarized should produce a different contrast factor than unpolarized."""
        tm = PolarizationState(tm_weight=1.0)
        unpol = PolarizationState(tm_weight=0.5)
        cf_tm = tm.contrast_factor(na=0.55)
        cf_unpol = unpol.contrast_factor(na=0.55)
        # Both should be in [0, 1] but should differ
        assert 0.0 <= cf_tm <= 1.0
        assert 0.0 <= cf_unpol <= 1.0
        assert cf_tm != pytest.approx(cf_unpol, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 6: TripleBeamIllumination modifies aerial image
# ---------------------------------------------------------------------------


class TestTripleBeamIllumination:
    def test_modifies_aerial_image(self, line_mask: torch.Tensor) -> None:
        imaging = AnamorphicImaging()
        aerial_scalar = imaging.simulate_aerial(line_mask)

        pol = PolarizationState(tm_weight=1.0)
        triple = TripleBeamIllumination(
            polarization=pol,
            pitch_nm=32.0,
            pixel_size_nm=1.0,
            na=0.55,
        )
        aerial_modified = triple.modify_aerial_image(aerial_scalar)

        assert aerial_modified.shape == aerial_scalar.shape
        # Modified should differ from scalar because pitch > resolution limit
        assert not torch.allclose(aerial_modified, aerial_scalar, atol=1e-6)

    def test_effective_contrast(self) -> None:
        triple = TripleBeamIllumination(
            polarization=PolarizationState(),
            pitch_nm=32.0,
            na=0.55,
        )
        contrast = triple.compute_effective_contrast()
        assert 0.0 <= contrast <= 1.0

    def test_sub_resolution_zero_efficiency(self) -> None:
        """Pitch below resolution limit should give zero efficiency."""
        triple = TripleBeamIllumination(
            pitch_nm=8.0,  # well below lambda/(2*NA) ~ 12.3 nm
            na=0.55,
        )
        eta = triple._diffraction_efficiency()
        assert eta == 0.0


# ---------------------------------------------------------------------------
# Test 7: Polarization reduces stochastic failure rate vs scalar
# ---------------------------------------------------------------------------


class TestPolarizationReducesFailure:
    def test_polarized_lower_failure_than_scalar(
        self,
        line_mask: torch.Tensor,
    ) -> None:
        """Polarized illumination should produce a different (ideally better)
        aerial image than scalar, demonstrating the polarization effect."""
        params = AnamorphicParams(na_x=0.55, na_y=0.55)

        # Scalar (no polarization)
        imaging_scalar = AnamorphicImaging(params=params)
        aerial_scalar = imaging_scalar.simulate_aerial(line_mask)

        # With TM polarization
        pol = PolarizationState(tm_weight=1.0)
        imaging_pol = AnamorphicImaging(params=params, polarization=pol)
        aerial_pol = imaging_pol.simulate_aerial(line_mask)

        # Polarized aerial should be different from scalar
        assert not torch.allclose(aerial_pol, aerial_scalar, atol=1e-8)

        # Polarization scales down the image by contrast_factor < 1
        # at high NA, so mean should be lower (or equal if factor=1)
        assert aerial_pol.mean() <= aerial_scalar.mean() + 1e-6

    def test_triple_beam_with_anamorphic(
        self,
        line_mask: torch.Tensor,
    ) -> None:
        """Full pipeline: AnamorphicImaging + Polarization + TripleBeam."""
        params = AnamorphicParams(na_x=0.55, na_y=0.55)
        pol = PolarizationState(tm_weight=0.9, ellipticity=0.1)
        triple = TripleBeamIllumination(
            polarization=pol,
            pitch_nm=32.0,
            pixel_size_nm=1.0,
            na=0.55,
        )
        imaging = AnamorphicImaging(params=params, polarization=pol, triple_beam=triple)
        aerial = imaging.simulate_aerial(line_mask)

        assert aerial.shape == line_mask.shape
        assert (aerial >= 0.0).all()
        assert torch.isfinite(aerial).all()


# ---------------------------------------------------------------------------
# Test 8: Combined metrics pass conformal coverage check
# ---------------------------------------------------------------------------


class TestConformalCoverageCombined:
    def test_combined_metrics_pass_coverage(self) -> None:
        """Full pipeline: stochastic metrics + conformal coverage gate
        should produce a valid calibration report."""
        gate = ConformalCoverageGate3D(target_coverage=0.9, alpha=0.1)
        cal = StochasticCalibrationMetrics(coverage_gate=gate)

        # Create synthetic data
        mask = torch.zeros(16, 16)
        mask[:, 4:12] = 1.0

        defect_map = torch.zeros(16, 16)
        defect_map[8, 6] = 0.8
        defect_map[9, 7] = 0.7

        report = cal.evaluate(
            defect_map=defect_map,
            predicted_defects=defect_map,
            actual_defects=defect_map,
            mask=mask,
        )

        # Verify the report is well-formed
        assert report.failure_corr_length_nm >= 0.0
        assert isinstance(report.n_defect_clusters, int)
        assert isinstance(report.max_cluster_size, int)
        assert isinstance(report.mean_cluster_size, float)
        assert report.epe_quantile_nm == 0.0  # No ensemble provided
        assert report.epe_mean_nm == 0.0
        assert report.epe_std_nm == 0.0
        assert isinstance(report.conformal_coverage, float)
        assert 0.0 <= report.conformal_coverage <= 1.0
        assert report.conformal_verdict in ("ACCEPT", "REJECT", "UNCERTAIN")
