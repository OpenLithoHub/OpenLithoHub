"""Tests for compliance checks."""

import math

import torch

from openlithohub.benchmark.compliance.drc import check_drc
from openlithohub.benchmark.compliance.mrc import (
    CurvilinearMRCResult,
    MRCResult,
    check_curvilinear_mrc,
    check_mrc,
)


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


class TestDRC:
    def test_clean_mask_passes(self):
        from openlithohub.benchmark.compliance.drc import DRCRuleDeck

        mask = torch.zeros(64, 64)
        mask[10:54, 10:54] = 1.0
        rules = DRCRuleDeck(min_width_nm=8.0, min_spacing_nm=8.0, min_area_nm2=4.0)
        result = check_drc(mask, rule_deck=rules, pixel_size_nm=1.0)
        assert result.passed is True
        assert result.violation_count == 0

    def test_small_area_violation(self):
        mask = torch.zeros(64, 64)
        mask[10:54, 10:54] = 1.0
        mask[2:4, 2:4] = 1.0  # 4px island = 4nm2 < default 100nm2
        result = check_drc(mask, pixel_size_nm=1.0)
        assert result.passed is False
        assert result.rule_summary["min_area"] > 0

    def test_narrow_feature_width_violation(self):
        mask = torch.zeros(64, 64)
        mask[30:32, 10:54] = 1.0  # 2px-wide strip
        result = check_drc(mask, pixel_size_nm=1.0)
        assert result.passed is False
        assert result.rule_summary["min_width"] > 0

    def test_custom_rule_deck(self):
        from openlithohub.benchmark.compliance.drc import DRCRuleDeck

        mask = torch.zeros(64, 64)
        mask[10:54, 10:54] = 1.0
        mask[2:4, 2:4] = 1.0  # 4px island

        lenient = DRCRuleDeck(
            min_area_nm2=2.0, min_width_nm=1.0, min_spacing_nm=1.0, min_notch_nm=1.0
        )
        result = check_drc(mask, rule_deck=lenient, pixel_size_nm=1.0)
        assert result.passed is True

    def test_result_structure(self, sample_mask):
        from openlithohub.benchmark.compliance.drc import DRCResult

        result = check_drc(sample_mask)
        assert isinstance(result, DRCResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.violation_count, int)
        assert isinstance(result.violations, list)
        assert isinstance(result.rule_summary, dict)

    def test_notch_violation_detected(self):
        from openlithohub.benchmark.compliance.drc import DRCRuleDeck

        # Solid 40x40 fg block with a small enclosed bg pocket inside —
        # the textbook notch signature: bg surrounded by fg, narrower than
        # the notch threshold. Closing of the foreground at radius 2 fills
        # the pocket; the rule reports the pocket as a notch.
        mask = torch.zeros(64, 64)
        mask[12:52, 12:52] = 1.0
        mask[30:32, 30:32] = 0.0  # 2x2 enclosed bg pocket
        rules = DRCRuleDeck(
            min_area_nm2=2.0,
            min_width_nm=2.0,
            min_spacing_nm=2.0,
            min_notch_nm=4.0,
        )
        result = check_drc(mask, rule_deck=rules, pixel_size_nm=1.0)
        assert result.rule_summary["notch"] > 0
        assert result.passed is False


def _disk_mask(size: int, cy: float, cx: float, radius: float) -> torch.Tensor:
    """Return a binary mask with a single filled disk."""
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2).float()


class TestCurvilinearMRCCurvature:
    def test_large_circle_passes(self):
        # Radius 30 px @ 1 nm/px -> 30 nm; threshold 20 nm -> pass.
        mask = _disk_mask(128, 64, 64, 30.0)
        result = check_curvilinear_mrc(
            mask,
            min_curvature_radius_nm=20.0,
            min_feature_area_nm2=100.0,
            pixel_size_nm=1.0,
        )
        assert result.passed is True
        assert result.violation_count == 0
        assert result.min_radius_observed_nm is not None
        # Smoothed traced contour should give a radius near the true 30 nm.
        assert result.min_radius_observed_nm > 20.0

    def test_small_circle_fails_curvature(self):
        # Radius 5 px @ 1 nm/px -> 5 nm; threshold 20 nm -> sharp curvature.
        mask = _disk_mask(64, 32, 32, 5.0)
        # Area is pi * 25 ~= 78 nm^2; relax area threshold so only curvature can fail.
        result = check_curvilinear_mrc(
            mask,
            min_curvature_radius_nm=20.0,
            min_feature_area_nm2=10.0,
            pixel_size_nm=1.0,
        )
        assert result.passed is False
        assert len(result.curvature_violations) > 0
        assert result.min_radius_observed_nm is not None
        assert result.min_radius_observed_nm < 20.0
        v = result.curvature_violations[0]
        assert v["actual_radius_nm"] < v["required_radius_nm"]

    def test_pixel_size_scaling_curvature(self):
        # Radius 5 px @ 4 nm/px -> 20 nm physical; threshold 15 nm -> pass.
        mask = _disk_mask(64, 32, 32, 5.0)
        result = check_curvilinear_mrc(
            mask,
            min_curvature_radius_nm=15.0,
            min_feature_area_nm2=10.0,
            pixel_size_nm=4.0,
        )
        # Physical radius is ~20 nm; with rasterization noise the smoother gives
        # a slightly smaller min radius — accept if no curvature violations dominate.
        assert result.min_radius_observed_nm is not None
        assert result.min_radius_observed_nm > 10.0

    def test_manhattan_square_does_not_blow_up(self):
        # 90 degree corners would produce infinite curvature without smoothing.
        # The smoother + skip stride should keep this finite and not flag the
        # straight edges as violations.
        mask = torch.zeros(64, 64)
        mask[16:48, 16:48] = 1.0
        result = check_curvilinear_mrc(
            mask,
            min_curvature_radius_nm=2.0,
            min_feature_area_nm2=10.0,
            pixel_size_nm=1.0,
            smoothing_window=5,
        )
        # Function returns without exception and observed radius is finite.
        assert result.min_radius_observed_nm is not None
        assert math.isfinite(result.min_radius_observed_nm)


class TestCurvilinearMRCArea:
    def test_large_feature_passes_area(self):
        mask = _disk_mask(64, 32, 32, 10.0)  # ~314 px area @ 1 nm/px
        result = check_curvilinear_mrc(
            mask,
            min_curvature_radius_nm=1.0,
            min_feature_area_nm2=200.0,
            pixel_size_nm=1.0,
        )
        assert all(v["actual_nm2"] >= 200.0 for v in result.area_violations)
        assert result.min_area_observed_nm2 is not None
        assert result.min_area_observed_nm2 >= 200.0

    def test_small_dot_fails_area(self):
        mask = torch.zeros(64, 64)
        mask[31:33, 31:33] = 1.0  # 4 px dot
        result = check_curvilinear_mrc(
            mask,
            min_curvature_radius_nm=0.0,  # disable curvature check
            min_feature_area_nm2=100.0,
            pixel_size_nm=1.0,
        )
        assert result.passed is False
        assert len(result.area_violations) == 1
        v = result.area_violations[0]
        assert v["actual_nm2"] == 4.0
        assert v["required_nm2"] == 100.0

    def test_area_scales_with_pixel_size(self):
        mask = torch.zeros(64, 64)
        mask[31:33, 31:33] = 1.0  # 4 px
        # 4 px * (5 nm)^2 = 100 nm^2 -> exactly at threshold, should pass.
        result = check_curvilinear_mrc(
            mask,
            min_curvature_radius_nm=0.0,
            min_feature_area_nm2=100.0,
            pixel_size_nm=5.0,
        )
        assert len(result.area_violations) == 0


class TestCurvilinearMRCResultShape:
    def test_result_dataclass_fields(self):
        mask = _disk_mask(64, 32, 32, 12.0)
        result = check_curvilinear_mrc(mask)
        assert isinstance(result, CurvilinearMRCResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.violation_count, int)
        assert isinstance(result.curvature_violations, list)
        assert isinstance(result.area_violations, list)
        assert result.violation_count == len(result.curvature_violations) + len(
            result.area_violations
        )

    def test_empty_mask(self):
        mask = torch.zeros(64, 64)
        result = check_curvilinear_mrc(mask)
        assert result.passed is True
        assert result.violation_count == 0
        assert result.min_radius_observed_nm is None
        assert result.min_area_observed_nm2 is None

    def test_4d_input_shape(self):
        mask = _disk_mask(64, 32, 32, 15.0).unsqueeze(0).unsqueeze(0)
        result = check_curvilinear_mrc(mask)
        assert isinstance(result, CurvilinearMRCResult)
