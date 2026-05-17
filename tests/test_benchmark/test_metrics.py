"""Tests for benchmark metrics."""

import pytest
import torch

from openlithohub.benchmark.metrics.epe import compute_epe
from openlithohub.benchmark.metrics.shot_count import estimate_shot_count


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
