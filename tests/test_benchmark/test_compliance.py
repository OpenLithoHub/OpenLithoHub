"""Tests for compliance checks."""

import pytest
import torch

from openlithohub.benchmark.compliance.drc import check_drc
from openlithohub.benchmark.compliance.mrc import MRCResult, check_mrc


class TestMRCWidthCheck:
    def test_wide_feature_passes(self):
        mask = torch.zeros(64, 64)
        mask[16:48, 16:48] = 1.0  # 32px wide
        result = check_mrc(mask, min_width_nm=16.0, min_spacing_nm=8.0, pixel_size_nm=1.0)
        assert result.passed is True
        assert result.violation_count == 0

    def test_narrow_feature_fails(self):
        mask = torch.zeros(64, 64)
        mask[30:34, 10:54] = 1.0  # 4px tall strip
        result = check_mrc(mask, min_width_nm=10.0, min_spacing_nm=4.0, pixel_size_nm=1.0)
        assert result.passed is False
        assert result.violation_count > 0

    def test_feature_exactly_at_threshold(self):
        mask = torch.zeros(64, 64)
        mask[20:40, 20:40] = 1.0  # 20px wide
        # radius = floor(20/(2*1)) = 10. Erode 20x20 by 10 -> vanishes. Fails.
        result = check_mrc(mask, min_width_nm=20.0, min_spacing_nm=4.0, pixel_size_nm=1.0)
        assert result.passed is False

    def test_feature_above_threshold(self):
        mask = torch.zeros(64, 64)
        mask[10:54, 10:54] = 1.0  # 44px wide
        # radius = floor(20/(2*1)) = 10. Erode 44x44 by 10 -> 24x24 survives.
        result = check_mrc(mask, min_width_nm=20.0, min_spacing_nm=4.0, pixel_size_nm=1.0)
        assert result.passed is True

    def test_pixel_size_scaling(self):
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0  # 24px wide
        # At 2nm/pixel: physical width = 48nm. min_width=40nm -> should pass.
        # radius = floor(40/(2*2)) = 10. Erode 24x24 by 10 -> 4x4 survives.
        result = check_mrc(mask, min_width_nm=40.0, min_spacing_nm=10.0, pixel_size_nm=2.0)
        assert result.passed is True


class TestMRCSpacingCheck:
    def test_wide_spacing_passes(self):
        mask = torch.zeros(64, 64)
        mask[10:20, 10:20] = 1.0
        mask[10:20, 40:50] = 1.0  # 20px gap
        result = check_mrc(mask, min_width_nm=4.0, min_spacing_nm=10.0, pixel_size_nm=1.0)
        assert result.passed is True

    def test_narrow_spacing_fails(self):
        mask = torch.zeros(64, 64)
        mask[10:30, 10:30] = 1.0
        mask[10:30, 33:53] = 1.0  # 3px gap
        result = check_mrc(mask, min_width_nm=4.0, min_spacing_nm=8.0, pixel_size_nm=1.0)
        assert result.passed is False
        assert result.violation_count > 0


class TestMRCEdgeCases:
    def test_empty_mask_passes(self):
        mask = torch.zeros(64, 64)
        result = check_mrc(mask)
        assert result.passed is True
        assert result.violation_count == 0

    def test_full_mask_passes(self):
        mask = torch.ones(64, 64)
        result = check_mrc(mask, min_width_nm=16.0, min_spacing_nm=16.0)
        assert result.passed is True

    def test_result_type(self, sample_mask):
        result = check_mrc(sample_mask)
        assert isinstance(result, MRCResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.violation_count, int)
        assert isinstance(result.violation_rate, float)
        assert isinstance(result.violations, list)
        assert 0.0 <= result.violation_rate <= 1.0

    def test_violations_have_expected_fields(self):
        mask = torch.zeros(32, 32)
        mask[14:18, 5:27] = 1.0  # narrow strip
        result = check_mrc(mask, min_width_nm=10.0, pixel_size_nm=1.0)
        assert len(result.violations) > 0
        v = result.violations[0]
        assert "type_code" in v
        assert "x_nm" in v
        assert "y_nm" in v
        assert "actual_nm" in v
        assert "required_nm" in v


def test_drc_not_implemented(sample_mask):
    with pytest.raises(NotImplementedError, match="DRC"):
        check_drc(sample_mask)
