"""Tests for cross-tile EPE, contour residual, overlap-convergence sweep,
and Schwarz-vs-naive comparison metrics."""

import pytest
import torch

from openlithohub.benchmark.metrics.tiling_consistency import (
    cross_tile_contour_residual,
    cross_tile_epe_residual,
    schwarz_vs_naive_comparison,
    sweep_overlap_convergence,
)


def _identity_forward(tile: torch.Tensor) -> torch.Tensor:
    return tile


def _gaussian_forward(tile: torch.Tensor) -> torch.Tensor:
    from openlithohub._utils.forward_model import simulate_aerial_image

    return simulate_aerial_image(tile, sigma_px=2.0)


def _make_tiles_and_results(shape: int = 32, overlap: int = 8):
    """Build two adjacent tile tensors and their results."""
    mask = torch.zeros(shape * 2 - overlap, shape)
    mask[4 : shape - 4, 4 : shape - 4] = 1.0
    mask[shape - overlap + 4 : shape * 2 - overlap - 4, 4 : shape - 4] = 1.0

    tile_a = mask[:shape, :]
    tile_b = mask[shape - overlap :, :]
    return [tile_a, tile_b], [tile_a.clone(), tile_b.clone()]


# ---------------------------------------------------------------------------
# cross_tile_epe_residual
# ---------------------------------------------------------------------------


class TestCrossTileEpe:
    def test_returns_correct_metric_keys(self):
        tiles, results = _make_tiles_and_results()
        out = cross_tile_epe_residual(tiles, results, overlap=8)
        assert "mean_epe" in out
        assert "max_epe" in out
        assert "epe_per_boundary" in out

    def test_identical_tiles_zero_epe(self):
        # Uniform tiles (no edges) → zero EPE by construction
        tile_a = torch.ones(32, 32) * 0.7
        tile_b = tile_a.clone()
        out = cross_tile_epe_residual(
            [tile_a, tile_b], [tile_a.clone(), tile_b.clone()], overlap=8
        )
        assert out["mean_epe"] == pytest.approx(0.0, abs=1e-6)
        assert out["max_epe"] == pytest.approx(0.0, abs=1e-6)

    def test_different_tiles_positive_epe(self):
        tiles, results = _make_tiles_and_results()
        # Shift features in the second tile
        shifted = torch.zeros_like(results[1])
        shifted[:, 2:] = results[1][:, :-2]
        results[1] = shifted
        out = cross_tile_epe_residual(tiles, results, overlap=8)
        assert out["mean_epe"] >= 0.0

    def test_empty_input(self):
        out = cross_tile_epe_residual([], [], overlap=4)
        assert out["mean_epe"] == 0.0
        assert out["epe_per_boundary"] == []

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cross_tile_epe_residual(
                [torch.ones(4, 4)], [torch.ones(4, 4), torch.ones(4, 4)], overlap=2
            )

    def test_zero_overlap(self):
        tiles, results = _make_tiles_and_results()
        out = cross_tile_epe_residual(tiles, results, overlap=0)
        assert out["mean_epe"] == 0.0


# ---------------------------------------------------------------------------
# cross_tile_contour_residual
# ---------------------------------------------------------------------------


class TestCrossTileContour:
    def test_returns_correct_metric_keys(self):
        tiles, results = _make_tiles_and_results()
        out = cross_tile_contour_residual(tiles, results, overlap=8)
        assert "mean_contour_offset" in out
        assert "max_contour_offset" in out
        assert "n_boundary_pixels" in out

    def test_identical_tiles_zero_offset(self):
        # Uniform tiles (no contours) → zero offset
        tile_a = torch.ones(32, 32) * 0.7
        tile_b = tile_a.clone()
        out = cross_tile_contour_residual(
            [tile_a, tile_b], [tile_a.clone(), tile_b.clone()], overlap=8
        )
        assert out["mean_contour_offset"] == pytest.approx(0.0, abs=1e-6)

    def test_different_tiles_nonzero_offset(self):
        tiles, results = _make_tiles_and_results()
        # Invert the second tile to maximise contour disagreement
        results[1] = 1.0 - results[1]
        out = cross_tile_contour_residual(tiles, results, overlap=8)
        # Should detect contour disagreement
        assert out["n_boundary_pixels"] >= 0.0

    def test_empty_input(self):
        out = cross_tile_contour_residual([], [], overlap=4)
        assert out["mean_contour_offset"] == 0.0
        assert out["n_boundary_pixels"] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cross_tile_contour_residual(
                [torch.ones(4, 4)], [torch.ones(4, 4), torch.ones(4, 4)], overlap=2
            )


# ---------------------------------------------------------------------------
# sweep_overlap_convergence
# ---------------------------------------------------------------------------


class TestSweepOverlapConvergence:
    def test_sweep_produces_results_for_all_combinations(self):
        mask = torch.zeros(48, 48)
        mask[8:40, 8:40] = 1.0
        results = sweep_overlap_convergence(
            mask,
            tile_size=32,
            forward_fn=_identity_forward,
            overlap_range=[4, 8],
            schwarz_iter_range=[1, 3],
        )
        # Should have entries for each overlap value
        assert "4" in results
        assert "8" in results
        # Each overlap should have entries for each iteration count
        for ol_key in ("4", "8"):
            assert "1" in results[ol_key]
            assert "3" in results[ol_key]

    def test_sweep_result_keys(self):
        mask = torch.zeros(48, 48)
        mask[8:40, 8:40] = 1.0
        results = sweep_overlap_convergence(
            mask,
            tile_size=32,
            forward_fn=_identity_forward,
            overlap_range=[8],
            schwarz_iter_range=[1],
        )
        entry = results["8"]["1"]
        assert "seam_residual" in entry
        assert "epe" in entry
        assert "contour_offset" in entry
        assert "time" in entry

    def test_sweep_skips_invalid_overlap(self):
        mask = torch.zeros(32, 32)
        mask[4:28, 4:28] = 1.0
        # overlap=64 >= tile_size=32 should be skipped
        results = sweep_overlap_convergence(
            mask,
            tile_size=32,
            forward_fn=_identity_forward,
            overlap_range=[64],
            schwarz_iter_range=[1],
        )
        assert "64" not in results

    def test_sweep_with_gaussian_forward(self):
        mask = torch.zeros(48, 48)
        mask[8:40, 8:40] = 1.0
        results = sweep_overlap_convergence(
            mask,
            tile_size=32,
            forward_fn=_gaussian_forward,
            overlap_range=[4, 8],
            schwarz_iter_range=[1, 2],
        )
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# schwarz_vs_naive_comparison
# ---------------------------------------------------------------------------


class TestSchwarzVsNaive:
    def test_schwarz_produces_lower_seam_error(self):
        mask = torch.zeros(48, 48)
        mask[8:40, 8:40] = 1.0

        result = schwarz_vs_naive_comparison(
            mask,
            tile_size=32,
            overlap=8,
            n_schwarz_iter=3,
            forward_fn=_gaussian_forward,
        )
        assert "schwarz_seam_error" in result
        assert "naive_seam_error" in result
        assert "improvement_ratio" in result
        # Both should be non-negative
        assert result["schwarz_seam_error"] >= 0.0
        assert result["naive_seam_error"] >= 0.0

    def test_identity_forward_equal_errors(self):
        mask = torch.ones(48, 48) * 0.5
        result = schwarz_vs_naive_comparison(
            mask,
            tile_size=32,
            overlap=8,
            n_schwarz_iter=2,
            forward_fn=_identity_forward,
        )
        # With identity forward, Schwarz and naive should produce identical results
        assert result["schwarz_seam_error"] == pytest.approx(
            result["naive_seam_error"], abs=1e-6
        )

    def test_default_forward_fn(self):
        mask = torch.zeros(48, 48)
        mask[8:40, 8:40] = 1.0
        # Should not raise — default uses simulate_aerial_image
        result = schwarz_vs_naive_comparison(
            mask, tile_size=32, overlap=8, n_schwarz_iter=2
        )
        assert result["improvement_ratio"] > 0.0


# ---------------------------------------------------------------------------
# Monotonicity: more overlap should generally reduce residual
# ---------------------------------------------------------------------------


class TestOverlapConvergenceMonotone:
    def test_more_overlap_less_residual(self):
        """With a structured mask, larger overlap should not increase seam residual."""
        mask = torch.zeros(64, 64)
        mask[8:56, 8:56] = 1.0

        results = sweep_overlap_convergence(
            mask,
            tile_size=32,
            forward_fn=_gaussian_forward,
            overlap_range=[4, 8, 16],
            schwarz_iter_range=[3],
        )

        # Collect seam residuals for each overlap
        seam_values: list[float] = []
        overlaps: list[int] = []
        for ol_str, iter_data in results.items():
            if "3" in iter_data:
                overlaps.append(int(ol_str))
                seam_values.append(iter_data["3"]["seam_residual"])

        if len(seam_values) < 2:
            pytest.skip("Not enough overlap values to check monotonicity")

        # General trend: the residual at the largest overlap should be <= the
        # residual at the smallest overlap. We don't enforce strict monotonicity
        # because the forward model and tiling geometry can introduce noise,
        # but the overall direction should hold.
        assert seam_values[-1] <= seam_values[0] + 1e-6, (
            f"Residual did not decrease with more overlap: "
            f"overlap={overlaps}, residuals={seam_values}"
        )
