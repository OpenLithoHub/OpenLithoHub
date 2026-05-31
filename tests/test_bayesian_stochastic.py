"""Tests for BayesianStochasticModel: heatmaps, shapes, ranges, monotonicity."""

import torch

from openlithohub.models.bayesian_stochastic import (
    BayesianStochasticModel,
    StochasticUNet,
    generate_synthetic_ground_truth,
)


def _make_line_space_mask(size: int = 64, pitch: int = 8) -> torch.Tensor:
    """Create a line/space mask pattern for testing."""
    mask = torch.zeros(size, size)
    for x in range(0, size, pitch):
        mask[:, x : x + pitch // 2] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Test 1: predict returns correct structure
# ---------------------------------------------------------------------------


class TestPredictStructure:
    def test_poisson_mode_returns_required_fields(self):
        model = BayesianStochasticModel(
            n_mc_samples=8,
            mode="poisson",
            seed=42,
        )
        mask = _make_line_space_mask(32, 8)
        result = model.predict(mask)

        assert result.mask is not None
        assert result.mask.shape == (32, 32)
        assert "failure_prob" in result.metadata
        assert "ler_nm" in result.metadata
        assert "lwr_nm" in result.metadata
        assert result.metadata["failure_prob"].shape == (32, 32)
        assert result.metadata["ler_nm"].shape == (32, 32)
        assert result.metadata["lwr_nm"].shape == (32, 32)

    def test_mc_dropout_mode_returns_required_fields(self):
        model = BayesianStochasticModel(
            n_mc_samples=4,
            mode="mc_dropout",
            seed=42,
        )
        mask = _make_line_space_mask(32, 8)
        result = model.predict(mask)

        assert result.mask is not None
        assert "failure_prob" in result.metadata
        assert "ler_nm" in result.metadata
        assert "lwr_nm" in result.metadata


# ---------------------------------------------------------------------------
# Test 2: heatmap value ranges
# ---------------------------------------------------------------------------


class TestHeatmapRanges:
    def test_failure_prob_in_01(self):
        model = BayesianStochasticModel(n_mc_samples=16, mode="poisson", seed=0)
        result = model.predict(_make_line_space_mask(32, 8))
        fp = result.metadata["failure_prob"]
        assert fp.min() >= -1e-9
        assert fp.max() <= 1.0 + 1e-9

    def test_ler_non_negative(self):
        model = BayesianStochasticModel(n_mc_samples=16, mode="poisson", seed=0)
        result = model.predict(_make_line_space_mask(32, 8))
        assert (result.metadata["ler_nm"] >= -1e-9).all()

    def test_lwr_non_negative(self):
        model = BayesianStochasticModel(n_mc_samples=16, mode="poisson", seed=0)
        result = model.predict(_make_line_space_mask(32, 8))
        assert (result.metadata["lwr_nm"] >= -1e-9).all()


# ---------------------------------------------------------------------------
# Test 3: monotonic response to dose perturbation
# ---------------------------------------------------------------------------


class TestDoseMonotonicity:
    def test_higher_dose_lower_failure(self):
        """Higher dose → more photons → lower failure probability on average."""
        mask = _make_line_space_mask(32, 8)
        model_low = BayesianStochasticModel(
            n_mc_samples=32,
            mode="poisson",
            dose_photons_per_nm2=10.0,
            seed=0,
        )
        model_high = BayesianStochasticModel(
            n_mc_samples=32,
            mode="poisson",
            dose_photons_per_nm2=100.0,
            seed=0,
        )
        fp_low = model_low.predict(mask).metadata["failure_prob"]
        fp_high = model_high.predict(mask).metadata["failure_prob"]
        assert fp_low.mean() > fp_high.mean()


# ---------------------------------------------------------------------------
# Test 4: synthetic ground truth generation
# ---------------------------------------------------------------------------


class TestSyntheticGroundTruth:
    def test_generate_ground_truth_shapes(self):
        mask = _make_line_space_mask(32, 8)
        gt = generate_synthetic_ground_truth(mask, n_mc=16, seed=0)
        assert gt["failure_prob"].shape == (32, 32)
        assert gt["ler_nm"].shape == (32, 32)
        assert gt["lwr_nm"].shape == (32, 32)
        assert gt["aerial"].shape == (32, 32)

    def test_ground_truth_failure_prob_range(self):
        gt = generate_synthetic_ground_truth(
            _make_line_space_mask(32, 8),
            n_mc=16,
            seed=0,
        )
        assert gt["failure_prob"].min() >= 0.0
        assert gt["failure_prob"].max() <= 1.0


# ---------------------------------------------------------------------------
# Test 5: calibration curve
# ---------------------------------------------------------------------------


class TestCalibrationCurve:
    def test_calibration_curve_returns_bins(self):
        model = BayesianStochasticModel(n_mc_samples=8, mode="poisson", seed=0)
        mask = _make_line_space_mask(32, 8)
        cal = model.calibration_curve(mask, n_bins=5, n_ground_truth_mc=32)
        assert len(cal["predicted"]) == 5
        assert len(cal["observed"]) == 5
        for p in cal["predicted"]:
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Test 6: StochasticUNet forward
# ---------------------------------------------------------------------------


class TestStochasticUNet:
    def test_unet_output_shapes(self):
        net = StochasticUNet(base_channels=16, dropout_p=0.1)
        x = torch.randn(1, 1, 32, 32)
        out = net(x)
        assert out["failure_logit"].shape == (1, 1, 32, 32)
        assert out["ler_log"].shape == (1, 1, 32, 32)
        assert out["lwr_log"].shape == (1, 1, 32, 32)


# ---------------------------------------------------------------------------
# Test 7: model registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_model_registered(self):
        from openlithohub.models.registry import registry

        assert "bayesian-stochastic" in registry.list_models()

    def test_instantiate_via_registry(self):
        from openlithohub.models.registry import registry

        model = registry.get("bayesian-stochastic", n_mc_samples=4)
        assert isinstance(model, BayesianStochasticModel)
        assert model.n_mc_samples == 4
