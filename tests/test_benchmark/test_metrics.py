"""Tests for benchmark metrics."""

import torch

from openlithohub.benchmark.metrics.epe import compute_epe


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
    import pytest

    a = torch.zeros(32, 32)
    b = torch.zeros(64, 64)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_epe(a, b)


def test_epe_returns_expected_keys(sample_design, sample_mask):
    result = compute_epe(sample_design, sample_mask, pixel_size_nm=1.0)
    assert "epe_mean_nm" in result
    assert "epe_max_nm" in result
    assert "epe_std_nm" in result
