"""Tests for benchmark metrics."""

import pytest
import torch

from openlithohub.benchmark.metrics.epe import compute_epe
from openlithohub.benchmark.metrics.pvband import compute_pvband
from openlithohub.benchmark.metrics.shot_count import estimate_shot_count
from openlithohub.benchmark.metrics.stochastic import compute_stochastic_robustness


def test_epe_identical_masks(sample_design):
    result = compute_epe(sample_design, sample_design, pixel_size_nm=1.0)
    assert result["epe_mean_nm"] == 0.0
    assert result["epe_max_nm"] == 0.0
    assert result["epe_std_nm"] == 0.0


def test_epe_shifted_mask():
    target = torch.zeros(64, 64)
    target[16:48, 16:48] = 1.0

    predicted = torch.zeros(64, 64)
    predicted[18:50, 18:50] = 1.0

    result = compute_epe(predicted, target, pixel_size_nm=1.0)
    assert result["epe_mean_nm"] > 0.0
    assert result["epe_max_nm"] >= result["epe_mean_nm"]


def test_epe_pixel_scaling():
    target = torch.zeros(64, 64)
    target[16:48, 16:48] = 1.0

    predicted = torch.zeros(64, 64)
    predicted[18:50, 18:50] = 1.0

    result_1nm = compute_epe(predicted, target, pixel_size_nm=1.0)
    result_7nm = compute_epe(predicted, target, pixel_size_nm=7.0)

    assert abs(result_7nm["epe_mean_nm"] - result_1nm["epe_mean_nm"] * 7.0) < 1e-4


def test_epe_empty_edges():
    blank = torch.zeros(32, 32)
    result = compute_epe(blank, blank, pixel_size_nm=1.0)
    assert result["epe_mean_nm"] == 0.0


def test_epe_shape_mismatch():
    a = torch.zeros(32, 32)
    b = torch.zeros(64, 64)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_epe(a, b)


def test_epe_returns_expected_keys(sample_design, sample_mask):
    result = compute_epe(sample_design, sample_mask, pixel_size_nm=1.0)
    assert "epe_mean_nm" in result
    assert "epe_max_nm" in result
    assert "epe_std_nm" in result


class TestShotCount:
    def test_mbmw_basic(self):
        mask = torch.zeros(32, 32)
        mask[10:20, 10:20] = 1.0  # 100 foreground pixels
        result = estimate_shot_count(
            mask, writer_type="mbmw", min_shot_size_nm=1.0, pixel_size_nm=1.0
        )
        assert result["shot_count"] == 100
        assert result["estimated_write_time_s"] > 0.0

    def test_mbmw_grid_scaling(self):
        mask = torch.zeros(32, 32)
        mask[10:20, 10:20] = 1.0  # 100 pixels
        # With 5nm grid and 1nm pixel: grid_pitch = 5px, shots_per_pixel = 1/25
        result = estimate_shot_count(
            mask, writer_type="mbmw", min_shot_size_nm=5.0, pixel_size_nm=1.0
        )
        assert result["shot_count"] == 4  # 100 / 25 = 4

    def test_empty_mask(self):
        mask = torch.zeros(32, 32)
        result = estimate_shot_count(mask, writer_type="mbmw")
        assert result["shot_count"] == 0
        assert result["estimated_write_time_s"] == 0.0

    def test_full_mask_mbmw(self):
        mask = torch.ones(16, 16)
        result = estimate_shot_count(
            mask, writer_type="mbmw", min_shot_size_nm=1.0, pixel_size_nm=1.0
        )
        assert result["shot_count"] == 256

    def test_vsb_basic(self):
        mask = torch.zeros(32, 32)
        mask[10:20, 10:20] = 1.0
        result = estimate_shot_count(
            mask, writer_type="vsb", min_shot_size_nm=5.0, pixel_size_nm=1.0
        )
        assert result["shot_count"] >= 1
        assert result["estimated_write_time_s"] > 0.0

    def test_vsb_complex_higher_than_simple(self):
        # A simple square should need fewer shots than a complex pattern
        simple = torch.zeros(64, 64)
        simple[10:54, 10:54] = 1.0

        complex_mask = torch.zeros(64, 64)
        complex_mask[10:54, 10:20] = 1.0
        complex_mask[10:54, 25:35] = 1.0
        complex_mask[10:54, 40:50] = 1.0

        r_simple = estimate_shot_count(simple, writer_type="vsb", pixel_size_nm=1.0)
        r_complex = estimate_shot_count(complex_mask, writer_type="vsb", pixel_size_nm=1.0)
        assert r_complex["shot_count"] >= r_simple["shot_count"]

    def test_invalid_writer_type(self):
        mask = torch.zeros(32, 32)
        with pytest.raises(ValueError, match="writer_type"):
            estimate_shot_count(mask, writer_type="invalid")

    def test_return_keys(self):
        mask = torch.ones(16, 16)
        result = estimate_shot_count(mask)
        assert "shot_count" in result
        assert "estimated_write_time_s" in result


class TestPVBand:
    def test_returns_expected_keys(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        result = compute_pvband(mask)
        assert "pvband_mean_nm" in result
        assert "pvband_max_nm" in result

    def test_uniform_mask_zero_band(self):
        # A full mask with no internal edges won't have a meaningful PV band
        # from internal features, but may have boundary effects
        mask = torch.ones(64, 64)
        result = compute_pvband(mask)
        # The band should be small since the mask is uniform
        assert result["pvband_mean_nm"] >= 0.0

    def test_simple_square_positive_band(self):
        mask = torch.zeros(64, 64)
        mask[16:48, 16:48] = 1.0
        result = compute_pvband(mask, defocus_range_nm=20.0)
        assert result["pvband_mean_nm"] > 0.0
        assert result["pvband_max_nm"] >= result["pvband_mean_nm"]

    def test_increases_with_defocus(self):
        mask = torch.zeros(64, 64)
        mask[16:48, 16:48] = 1.0
        r_small = compute_pvband(mask, defocus_range_nm=5.0)
        r_large = compute_pvband(mask, defocus_range_nm=40.0)
        assert r_large["pvband_mean_nm"] >= r_small["pvband_mean_nm"]

    def test_pixel_scaling(self):
        mask = torch.zeros(64, 64)
        mask[16:48, 16:48] = 1.0
        r1 = compute_pvband(mask, pixel_size_nm=1.0, defocus_range_nm=10.0)
        r2 = compute_pvband(mask, pixel_size_nm=2.0, defocus_range_nm=10.0)
        assert r2["pvband_mean_nm"] >= r1["pvband_mean_nm"]

    def test_empty_mask(self):
        mask = torch.zeros(32, 32)
        result = compute_pvband(mask)
        assert result["pvband_mean_nm"] == 0.0


class TestStochasticRobustness:
    def test_returns_expected_keys(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        result = compute_stochastic_robustness(mask, num_trials=5, seed=42)
        assert "bridge_probability" in result
        assert "break_probability" in result
        assert "ler_mean_nm" in result
        assert "robustness_score" in result

    def test_deterministic_with_seed(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        r1 = compute_stochastic_robustness(mask, num_trials=10, seed=123)
        r2 = compute_stochastic_robustness(mask, num_trials=10, seed=123)
        assert r1 == r2

    def test_large_feature_robust(self):
        mask = torch.zeros(64, 64)
        mask[10:54, 10:54] = 1.0
        result = compute_stochastic_robustness(
            mask, num_trials=10, dose_photons_per_nm2=100.0, seed=42
        )
        assert result["robustness_score"] >= 0.5

    def test_score_bounded(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        result = compute_stochastic_robustness(mask, num_trials=5, seed=42)
        assert 0.0 <= result["robustness_score"] <= 1.0
        assert 0.0 <= result["bridge_probability"] <= 1.0
        assert 0.0 <= result["break_probability"] <= 1.0
        assert result["ler_mean_nm"] >= 0.0
