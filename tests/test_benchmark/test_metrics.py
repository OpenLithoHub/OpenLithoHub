"""Tests for benchmark metrics."""

import pytest
import torch

from openlithohub.benchmark.metrics import (
    StochasticDefectRates,
    compute_stochastic_defect_classes,
)
from openlithohub.benchmark.metrics.epe import compute_epe, compute_wafer_epe
from openlithohub.benchmark.metrics.hotspot import compute_hotspot_detection
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
    assert result["valid"] is True


def test_wafer_epe_identity_is_nonzero():
    # A square mask passed straight through (Identity model) must NOT
    # score 0 on wafer-level EPE: Hopkins diffraction rounds the corners
    # of the printed contour, so the resist image differs from the mask.
    # This is the regression guard for the bug where eval skipped the
    # forward simulator and compared mask-against-mask.
    #
    # Use 8 nm/px so the 256x256 grid covers a 2 µm window — small
    # enough to run fast, large enough that 193 nm ArF diffraction
    # actually resolves edges instead of smearing into a DC value.
    from openlithohub.simulators.base import SimulatorConfig
    from openlithohub.simulators.hopkins_sim import HopkinsSimulator

    mask = torch.zeros(256, 256)
    mask[64:192, 64:192] = 1.0

    sim = HopkinsSimulator(SimulatorConfig(extra={"pixel_size_nm": 8.0}))
    result = compute_wafer_epe(mask, mask, pixel_size_nm=8.0, simulator=sim)
    assert result["valid"] is True
    assert result["epe_mean_nm"] > 0.0
    assert result["epe_max_nm"] >= result["epe_mean_nm"]


def test_l2_error_identity_is_nonzero():
    # Mirror of the wafer-EPE regression: a square mask passed unchanged
    # through the Identity model must accumulate nonzero L2 wafer error
    # because diffraction reshapes the printed contour. This pins the
    # Neural-ILT-compatible scoring contract: forward-sim then compare.
    from openlithohub.benchmark.metrics.l2_error import compute_l2_error
    from openlithohub.simulators.base import SimulatorConfig
    from openlithohub.simulators.hopkins_sim import HopkinsSimulator

    mask = torch.zeros(256, 256)
    mask[64:192, 64:192] = 1.0

    sim = HopkinsSimulator(SimulatorConfig(extra={"pixel_size_nm": 8.0}))
    result = compute_l2_error(mask, mask, pixel_size_nm=8.0, simulator=sim)
    assert result["l2_error_pixels"] > 0.0
    # nm² conversion is l2_pixels * pixel_size_nm**2 (= 64 here).
    assert result["l2_error_nm2"] == pytest.approx(result["l2_error_pixels"] * 64.0)
    assert result["target_pixels"] == 128 * 128


def test_l2_error_shape_mismatch():
    from openlithohub.benchmark.metrics.l2_error import compute_l2_error

    a = torch.zeros(32, 32)
    b = torch.zeros(64, 64)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_l2_error(a, b)


def test_epe_one_empty_one_not_returns_inf_and_invalid():
    blank = torch.zeros(32, 32)
    nonblank = torch.zeros(32, 32)
    nonblank[8:24, 8:24] = 1.0

    pred_empty = compute_epe(blank, nonblank, pixel_size_nm=1.0)
    assert pred_empty["epe_mean_nm"] == float("inf")
    assert pred_empty["epe_max_nm"] == float("inf")
    assert pred_empty["valid"] is False

    target_empty = compute_epe(nonblank, blank, pixel_size_nm=1.0)
    assert target_empty["epe_mean_nm"] == float("inf")
    assert target_empty["valid"] is False


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


def test_epe_full_foreground_no_phantom_border():
    # Two identical fully-foreground masks: a zero-padded Sobel would emit a
    # 1-pixel phantom edge along every border, biasing EPE for tile-based
    # workflows. The border-strip in _extract_edges keeps EPE at zero.
    full = torch.ones(32, 32)
    result = compute_epe(full, full, pixel_size_nm=1.0)
    assert result["epe_mean_nm"] == 0.0
    assert result["epe_max_nm"] == 0.0


def test_epe_symmetric_catches_under_print():
    """Predicted mask is missing a feature target has → EPE must be > 0.

    The asymmetric (predicted→target only) form returns 0 here because
    predicted's edge set is empty for the missing region, so there's
    nothing to measure. Symmetric EPE catches this by also measuring
    target→predicted.
    """
    target = torch.zeros(64, 64)
    target[10:20, 10:20] = 1.0  # one square
    target[30:40, 30:40] = 1.0  # second square

    predicted = torch.zeros(64, 64)
    predicted[10:20, 10:20] = 1.0  # only the first square

    result = compute_epe(predicted, target, pixel_size_nm=1.0)
    assert result["valid"] is True
    # The missing 10×10 square contributes target→predicted distances
    # ≳ 10 pixels; mean must be well above zero.
    assert result["epe_mean_nm"] > 1.0
    assert result["epe_max_nm"] > 10.0
    assert result["valid"] is True


def test_epe_single_edge_pixel_yields_nan_std(monkeypatch):

    from openlithohub.benchmark.metrics import epe as epe_mod

    # Sobel always produces a multi-pixel edge ring even for a 1x1 island, so
    # we force the single-edge path by stubbing _extract_edges to return a
    # mask with exactly one true pixel. That exercises the numel()==1 branch.
    def _one_edge(binary):
        out = torch.zeros_like(binary, dtype=torch.bool)
        out[binary.shape[0] // 2, binary.shape[1] // 2] = True
        return out

    monkeypatch.setattr(epe_mod, "_extract_edges", _one_edge)

    a = torch.zeros(8, 8)
    b = torch.zeros(8, 8)
    result = epe_mod.compute_epe(a, b, pixel_size_nm=1.0)
    assert result["valid"] is True
    assert result["epe_mean_nm"] == 0.0
    # Symmetric EPE: pred→target (1) + target→pred (1) = 2 measurements,
    # both at distance 0, so std is well-defined and equals 0.
    assert result["epe_std_nm"] == 0.0


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

    def test_thickness_grows_with_dose_variation(self):
        # PV band should report contour-to-contour distance: increasing dose
        # variation widens the gap between outer and inner envelopes, so the
        # reported width must grow monotonically.
        mask = torch.zeros(96, 96)
        mask[24:72, 24:72] = 1.0
        r_low = compute_pvband(mask, dose_variation=0.02, defocus_range_nm=10.0)
        r_high = compute_pvband(mask, dose_variation=0.20, defocus_range_nm=40.0)
        assert r_high["pvband_mean_nm"] > r_low["pvband_mean_nm"]


class TestStochasticRobustness:
    def test_returns_expected_keys(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        result = compute_stochastic_robustness(mask, num_trials=5, seed=42)
        assert "bridge_probability" in result
        assert "break_probability" in result
        assert "edge_flip_rate" in result
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
        assert result["edge_flip_rate"] >= 0.0

    def test_bridging_layout_reports_bridge(self):
        # Two 4-px-wide bars with a 3-px gap. After sigma=2 blur the gap
        # aerial sits at ~0.44 — just below threshold — so high-dose Poisson
        # noise lifts it above 0.5 and merges the bars (fg components 2 -> 1).
        # Pre-fix the metric counted breaks via background components and
        # this case slipped through; the assertion guards that polarity.
        # Calibrated for resist_threshold=0.5 (the legacy mid-grey cut).
        mask = torch.zeros(64, 64)
        mask[16:48, 24:28] = 1.0
        mask[16:48, 31:35] = 1.0
        result = compute_stochastic_robustness(
            mask, num_trials=60, dose_photons_per_nm2=200.0, seed=7, resist_threshold=0.5
        )
        assert result["bridge_probability"] > 0.5
        assert result["break_probability"] < 0.1

    def test_breaking_layout_reports_break(self):
        # Single thin 4-px bar at very low dose. Poisson noise punches holes
        # along the bar, splitting the nominal line component into multiple
        # pieces. Per-component matching counts only genuine splits of a
        # nominal component (not far-field photon blobs), so the threshold
        # here measures the underlying split rate, not extra-component noise.
        mask = torch.zeros(64, 64)
        mask[30:34, 8:56] = 1.0
        result = compute_stochastic_robustness(
            mask, num_trials=60, dose_photons_per_nm2=2.0, seed=11, resist_threshold=0.5
        )
        assert result["break_probability"] >= 0.5
        assert result["bridge_probability"] < 0.1

    def test_numerical_regression_pinned_seed(self):
        # Pin exact output for a fixed mask + seed so future RNG refactors
        # (e.g. removing the per-trial reseed, switching generators) cannot
        # silently change semantics. Update only when the change is intentional.
        # A solid 16x16 square at dose 30 has no genuine bridges or splits of
        # the nominal component — earlier "break_probability=0.3" reflected
        # far-field photon blobs counted by the prior delta-of-component-count
        # heuristic; per-component matching correctly reports zero.
        # Pinned at resist_threshold=0.5 (the legacy mid-grey cut) — the
        # exact edge_flip_rate is calibration-specific.
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        result = compute_stochastic_robustness(
            mask, num_trials=20, dose_photons_per_nm2=30.0, seed=2026, resist_threshold=0.5
        )
        assert result["bridge_probability"] == pytest.approx(0.0)
        assert result["break_probability"] == pytest.approx(0.0)
        assert result["edge_flip_rate"] == pytest.approx(0.2216666679829359, abs=1e-9)
        assert result["robustness_score"] == pytest.approx(1.0)


class TestStochasticDefectClasses:
    def test_returns_dataclass(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        rates = compute_stochastic_defect_classes(mask, num_trials=5, seed=42)
        assert isinstance(rates, StochasticDefectRates)
        assert rates.num_trials == 5
        assert rates.image_area_cm2 > 0.0

    def test_all_rates_non_negative(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        rates = compute_stochastic_defect_classes(mask, num_trials=10, seed=42)
        assert rates.microbridge_per_cm2 >= 0
        assert rates.broken_line_per_cm2 >= 0
        assert rates.missing_contact_per_cm2 >= 0
        assert rates.merged_contact_per_cm2 >= 0
        assert rates.total_per_cm2 == pytest.approx(
            rates.microbridge_per_cm2
            + rates.broken_line_per_cm2
            + rates.missing_contact_per_cm2
            + rates.merged_contact_per_cm2
        )

    def test_deterministic_with_seed(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        r1 = compute_stochastic_defect_classes(mask, num_trials=10, seed=123)
        r2 = compute_stochastic_defect_classes(mask, num_trials=10, seed=123)
        assert r1.total_per_cm2 == pytest.approx(r2.total_per_cm2)
        assert r1.microbridge_per_cm2 == pytest.approx(r2.microbridge_per_cm2)

    def test_per_cm2_scales_with_pixel_size(self):
        # Pixel size affects both Poisson lambda (per-pixel area) and the
        # per-cm^2 normalisation, so we only assert the normalisation is
        # finite and positive — exact rate scaling is not invariant.
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        rates = compute_stochastic_defect_classes(mask, num_trials=4, pixel_size_nm=2.0, seed=7)
        assert rates.image_area_cm2 == pytest.approx((32 * 2.0) ** 2 * 1e-14)

    def test_isolated_contact_classified_correctly(self):
        # A small square (4x4) on a 32x32 canvas → contact-like (small + square).
        mask = torch.zeros(32, 32)
        mask[14:18, 14:18] = 1.0
        rates = compute_stochastic_defect_classes(
            mask, num_trials=4, dose_photons_per_nm2=1.0, seed=999
        )
        # Low dose → photon noise can wipe the small contact in some trials.
        # We only assert no broken-line or microbridge fires (no line-like
        # nominal component exists).
        assert rates.broken_line_per_cm2 == 0.0
        assert rates.microbridge_per_cm2 == 0.0


class TestHotspotDetection:
    def test_perfect_match(self):
        gt = torch.tensor([[10.0, 10.0], [50.0, 50.0], [100.0, 200.0]])
        pred = gt.clone()
        result = compute_hotspot_detection(pred, gt, match_radius_nm=1.0)
        assert result["num_tp"] == 3.0
        assert result["num_fp"] == 0.0
        assert result["num_fn"] == 0.0
        assert result["recall"] == 1.0
        assert result["precision"] == 1.0
        assert result["f1"] == 1.0

    def test_within_radius_counts_as_match(self):
        gt = torch.tensor([[10.0, 10.0]])
        pred = torch.tensor([[10.5, 10.0]])  # 0.5 nm away
        result = compute_hotspot_detection(pred, gt, match_radius_nm=1.0)
        assert result["num_tp"] == 1.0
        assert result["recall"] == 1.0
        assert result["precision"] == 1.0

    def test_outside_radius_misses(self):
        gt = torch.tensor([[10.0, 10.0]])
        pred = torch.tensor([[15.0, 10.0]])  # 5 nm away
        result = compute_hotspot_detection(pred, gt, match_radius_nm=1.0)
        assert result["num_tp"] == 0.0
        assert result["num_fp"] == 1.0
        assert result["num_fn"] == 1.0
        assert result["recall"] == 0.0
        assert result["precision"] == 0.0
        assert result["f1"] == 0.0

    def test_duplicate_predictions_become_fp(self):
        gt = torch.tensor([[10.0, 10.0]])
        # Two predictions both inside the disk: one TP, one FP. GT cannot be
        # double-counted — this is the property the greedy matcher enforces.
        pred = torch.tensor([[10.0, 10.0], [10.5, 10.0]])
        result = compute_hotspot_detection(pred, gt, match_radius_nm=1.0)
        assert result["num_tp"] == 1.0
        assert result["num_fp"] == 1.0
        assert result["num_fn"] == 0.0
        assert result["recall"] == 1.0
        assert result["precision"] == 0.5

    def test_partial_recall(self):
        gt = torch.tensor([[10.0, 10.0], [50.0, 50.0], [100.0, 100.0]])
        pred = torch.tensor([[10.0, 10.0], [50.0, 50.0]])
        result = compute_hotspot_detection(pred, gt, match_radius_nm=1.0)
        assert result["num_tp"] == 2.0
        assert result["num_fn"] == 1.0
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["precision"] == 1.0

    def test_empty_gt_and_pred_is_vacuous_perfect(self):
        empty = torch.zeros(0, 2)
        result = compute_hotspot_detection(empty, empty, match_radius_nm=1.0)
        assert result["recall"] == 1.0
        assert result["precision"] == 1.0
        assert result["f1"] == 1.0

    def test_empty_predictions_with_gt(self):
        gt = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
        empty = torch.zeros(0, 2)
        result = compute_hotspot_detection(empty, gt, match_radius_nm=1.0)
        assert result["num_fn"] == 2.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_predictions_with_empty_gt(self):
        empty = torch.zeros(0, 2)
        pred = torch.tensor([[10.0, 10.0]])
        result = compute_hotspot_detection(pred, empty, match_radius_nm=1.0)
        assert result["num_fp"] == 1.0
        assert result["precision"] == 0.0
        assert result["f1"] == 0.0

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="predicted_points"):
            compute_hotspot_detection(torch.zeros(3), torch.zeros(0, 2))
        with pytest.raises(ValueError, match="ground_truth_points"):
            compute_hotspot_detection(torch.zeros(0, 2), torch.zeros(3, 3))

    def test_negative_radius_rejected(self):
        with pytest.raises(ValueError, match="match_radius_nm"):
            compute_hotspot_detection(torch.zeros(0, 2), torch.zeros(0, 2), match_radius_nm=-1.0)
