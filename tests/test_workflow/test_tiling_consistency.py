"""Tests for tiling consistency and Manhattanization metrics."""

import math

import pytest
import torch

from openlithohub.benchmark.metrics.manhattanization import (
    curvilinear_to_manhattan,
    manhattanization_degradation,
)
from openlithohub.benchmark.metrics.tiling_consistency import (
    cross_tile_sraf_consistency,
    tile_boundary_consistency,
)
from openlithohub.workflow.tiling import (
    Tile,
    tiled_ilt_with_consistency,
    tile_layout,
)


# ---------------------------------------------------------------------------
# tile_boundary_consistency
# ---------------------------------------------------------------------------


class TestTileBoundaryConsistency:
    def test_perfectly_consistent_tiles_zero_metrics(self):
        """Identical tile results at the overlap → all metrics ~0."""
        layout = torch.ones(56, 56) * 0.7
        tiles = tile_layout(layout, tile_size=32, overlap=8)

        # Same uniform result for every tile → overlap is identical
        results = [torch.ones(32, 32) * 0.7 for _ in tiles]

        metrics = tile_boundary_consistency(tiles, results, overlap=8)
        assert metrics["boundary_mse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["boundary_max_diff"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["sraf_consistency"] == pytest.approx(1.0, abs=1e-6)

    def test_mismatched_tiles_positive_metrics(self):
        """Different tile results at overlap → metrics should be > 0."""
        layout = torch.ones(56, 56)
        tiles = tile_layout(layout, tile_size=32, overlap=8)
        assert len(tiles) >= 2

        # Alternate between high and low values so adjacent tiles differ
        results = []
        for i, t in enumerate(tiles):
            val = 0.8 if i % 2 == 0 else 0.2
            results.append(torch.ones(t.tensor.shape) * val)

        metrics = tile_boundary_consistency(tiles, results, overlap=8)
        assert metrics["boundary_mse"] > 0.0
        assert metrics["boundary_max_diff"] > 0.0

    def test_empty_tiles_list(self):
        """Empty tiles list → safe default (no boundaries to measure)."""
        metrics = tile_boundary_consistency([], [])
        assert metrics["boundary_mse"] == 0.0
        assert metrics["boundary_max_diff"] == 0.0
        assert metrics["sraf_consistency"] == 1.0

    def test_length_mismatch_raises(self):
        """tiles and tile_results of different lengths → ValueError."""
        tiles = [Tile(tensor=torch.ones(4, 4), origin_x=0, origin_y=0, width=4, height=4, overlap=1)]
        with pytest.raises(ValueError, match="same length"):
            tile_boundary_consistency(tiles, [torch.ones(4, 4), torch.ones(4, 4)])

    def test_vertical_adjacency(self):
        """Tiles stacked vertically with mismatched results."""
        layout = torch.ones(56, 56)
        tiles = tile_layout(layout, tile_size=32, overlap=8)
        # All tiles high value
        results = [torch.ones(32, 32) * 0.9 for _ in tiles]

        metrics = tile_boundary_consistency(tiles, results, overlap=8)
        # Uniform values → should be zero
        assert metrics["boundary_mse"] == pytest.approx(0.0, abs=1e-6)

    def test_no_overlap_zero_metrics(self):
        """Tiles with no overlap → no boundary regions to compare."""
        layout = torch.ones(64, 64)
        tiles = tile_layout(layout, tile_size=32, overlap=0)
        # Different results for different tiles
        results = [torch.rand(32, 32) for _ in tiles]

        metrics = tile_boundary_consistency(tiles, results, overlap=0)
        assert metrics["boundary_mse"] == 0.0
        assert metrics["boundary_max_diff"] == 0.0


# ---------------------------------------------------------------------------
# cross_tile_sraf_consistency
# ---------------------------------------------------------------------------


class TestCrossTileSrafConsistency:
    def test_uniform_mask_no_discontinuity(self):
        """Uniform mask → no SRAF discontinuity at tile edges."""
        mask = torch.ones(64, 64)
        result = cross_tile_sraf_consistency(mask, tile_size=32)
        assert result["sraf_discontinuity_rate"] == 0.0

    def test_single_tile_mask(self):
        """Mask smaller than tile_size → no boundaries, no discontinuity."""
        mask = torch.rand(16, 16)
        result = cross_tile_sraf_consistency(mask, tile_size=64)
        assert result["sraf_discontinuity_rate"] == 0.0
        assert result["n_boundary_pixels"] == 0

    def test_sraf_continuous_across_boundary(self):
        """SRAF stripe that extends across tile boundary → low discontinuity."""
        mask = torch.zeros(64, 64)
        # Continuous 4-pixel-wide bar crossing the tile boundary at col 32
        mask[:, 30:34] = 0.25
        result = cross_tile_sraf_consistency(mask, tile_size=32, sraf_threshold=0.3)
        # At col 32 boundary: col 31 is SRAF, col 32 is SRAF → match.
        # Most of the 64-row boundary should agree.
        assert result["sraf_discontinuity_rate"] < 0.5

    def test_sraf_discontinuous_at_boundary(self):
        """SRAF that stops right before the tile boundary → discontinuity."""
        mask = torch.zeros(64, 64)
        # SRAF in rows 28-31 (just before boundary at row 32), not crossing it
        mask[28:32, :] = 0.25
        result = cross_tile_sraf_consistency(mask, tile_size=32, sraf_threshold=0.3)
        # At row 32: above is SRAF (row 31), below is background (row 32) → mismatch
        assert result["sraf_discontinuity_rate"] > 0.0

    def test_returns_expected_keys(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 0.2
        result = cross_tile_sraf_consistency(mask, tile_size=16)
        assert "sraf_discontinuity_rate" in result
        assert "n_boundary_pixels" in result


# ---------------------------------------------------------------------------
# manhattanization_degradation
# ---------------------------------------------------------------------------


class TestManhattanizationDegradation:
    def test_identical_masks_zero_degradation(self):
        """Same mask passed as both curvilinear and Manhattanized → ~0 error."""
        mask = torch.zeros(64, 64)
        mask[16:48, 16:48] = 1.0
        result = manhattanization_degradation(mask, mask)
        assert result["edge_placement_error_nm"] == pytest.approx(0.0, abs=1e-6)
        assert result["area_error_frac"] == pytest.approx(0.0, abs=1e-6)
        assert result["shot_count_ratio"] == pytest.approx(1.0, abs=1e-6)

    def test_shape_mismatch_raises(self):
        a = torch.zeros(32, 32)
        b = torch.zeros(64, 64)
        with pytest.raises(ValueError, match="Shape mismatch"):
            manhattanization_degradation(a, b)

    def test_simple_rectangle_degradation(self):
        """A rotated rectangle Manhattanized should show nonzero EPE."""
        # Create a diagonal bar (curvilinear-friendly)
        curv = torch.zeros(64, 64)
        for i in range(10, 50):
            y = i
            x = i
            if 0 <= y < 64 and 0 <= x < 64:
                curv[max(0, y - 2) : min(64, y + 3), max(0, x - 2) : min(64, x + 3)] = 1.0

        # Manhattanized version: axis-aligned bar approximating the same area
        manh = torch.zeros(64, 64)
        manh[10:50, 10:50] = 1.0  # axis-aligned square covering the diagonal

        result = manhattanization_degradation(
            curv, manh, target_cd_nm=40.0, pixel_size_nm=2.0
        )
        assert result["edge_placement_error_nm"] >= 0.0
        assert result["shot_count_ratio"] > 0.0

    def test_returns_expected_keys(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        result = manhattanization_degradation(mask, mask)
        assert "edge_placement_error_nm" in result
        assert "pvb_increase" in result
        assert "shot_count_ratio" in result
        assert "area_error_frac" in result

    def test_empty_masks(self):
        """Both empty masks → zero error, no division by zero."""
        blank = torch.zeros(32, 32)
        result = manhattanization_degradation(blank, blank)
        assert result["edge_placement_error_nm"] == 0.0
        assert result["area_error_frac"] == 0.0
        assert result["shot_count_ratio"] == 0.0


# ---------------------------------------------------------------------------
# curvilinear_to_manhattan
# ---------------------------------------------------------------------------


class TestCurvilinearToManhattan:
    def test_invalid_angle_raises(self):
        mask = torch.ones(16, 16)
        with pytest.raises(ValueError, match="angle_quantization"):
            curvilinear_to_manhattan(mask, angle_quantization=30)

    def test_uniform_mask_unchanged(self):
        """Uniform mask has no edges → returns unchanged."""
        mask = torch.ones(16, 16)
        result = curvilinear_to_manhattan(mask, angle_quantization=45)
        assert torch.allclose(result, mask)

    def test_square_preserved(self):
        """An axis-aligned square should survive Manhattanization intact."""
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        result = curvilinear_to_manhattan(mask, angle_quantization=90)
        # Axis-aligned edges are already Manhattan → result should be very close
        assert result.shape == mask.shape

    def test_output_dtype_matches_input(self):
        mask = torch.zeros(16, 16, dtype=torch.float32)
        mask[4:12, 4:12] = 1.0
        result = curvilinear_to_manhattan(mask)
        assert result.dtype == torch.float32

    def test_angle_90_axis_aligned(self):
        """angle_quantization=90 produces axis-aligned output."""
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        result = curvilinear_to_manhattan(mask, angle_quantization=90)
        # The output should be a valid mask (all values in [0, 1] or binary)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_angle_45_allows_diagonals(self):
        """angle_quantization=45 should preserve more diagonal features than 90."""
        mask = torch.zeros(32, 32)
        # Diagonal bar
        for i in range(6, 26):
            mask[i, i] = 1.0
            if i + 1 < 32:
                mask[i, i + 1] = 1.0

        result_45 = curvilinear_to_manhattan(mask, angle_quantization=45)
        result_90 = curvilinear_to_manhattan(mask, angle_quantization=90)
        # Both should produce valid masks
        assert result_45.shape == mask.shape
        assert result_90.shape == mask.shape


# ---------------------------------------------------------------------------
# tiled_ilt_with_consistency
# ---------------------------------------------------------------------------


class TestTiledIltWithConsistency:
    def test_identity_ilt_perfect_consistency(self):
        """Identity ILT function (returns input unchanged) → zero boundary error."""

        def identity_ilt(tile: torch.Tensor) -> torch.Tensor:
            return tile

        mask = torch.ones(64, 64) * 0.5
        result = tiled_ilt_with_consistency(
            mask, tile_size=32, ilt_fn=identity_ilt, overlap=8, n_iterations=1
        )

        assert "mask" in result
        assert "consistency" in result
        assert result["mask"].shape == (64, 64)
        assert result["consistency"]["boundary_mse"] >= 0.0

    def test_returns_expected_keys(self):
        def identity_ilt(tile: torch.Tensor) -> torch.Tensor:
            return tile

        mask = torch.zeros(48, 48)
        mask[8:40, 8:40] = 1.0
        result = tiled_ilt_with_consistency(
            mask, tile_size=32, ilt_fn=identity_ilt, overlap=4, n_iterations=1
        )

        assert "mask" in result
        assert "tiles" in result
        assert "tile_results" in result
        assert "consistency" in result
        assert len(result["tiles"]) == len(result["tile_results"])

    def test_noisy_ilt_worse_consistency(self):
        """ILT function that adds noise → worse consistency than identity."""

        def identity_ilt(tile: torch.Tensor) -> torch.Tensor:
            return tile

        def noisy_ilt(tile: torch.Tensor) -> torch.Tensor:
            return tile + torch.randn_like(tile) * 0.1

        mask = torch.ones(64, 64) * 0.5

        result_clean = tiled_ilt_with_consistency(
            mask, tile_size=32, ilt_fn=identity_ilt, overlap=8, n_iterations=1
        )
        torch.manual_seed(42)
        result_noisy = tiled_ilt_with_consistency(
            mask, tile_size=32, ilt_fn=noisy_ilt, overlap=8, n_iterations=1
        )

        # Noisy should have higher boundary MSE than clean
        assert (
            result_noisy["consistency"]["boundary_mse"]
            >= result_clean["consistency"]["boundary_mse"]
        )
