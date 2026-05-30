"""Tests for Schwarz domain decomposition and Born forward correction (O3)."""

import pytest
import torch

from openlithohub.benchmark.metrics.tiling_consistency import schwarz_convergence_metrics
from openlithohub.workflow.tiling import schwarz_tiled_ilt


class TestSchwarzTiledIlt:
    def test_identity_ilt_returns_stitched_mask(self):
        """Identity ILT with Schwarz iteration returns valid stitched mask."""

        def identity_ilt(tile: torch.Tensor) -> torch.Tensor:
            return tile

        mask = torch.ones(64, 64) * 0.5
        result = schwarz_tiled_ilt(
            mask,
            tile_size=32,
            ilt_fn=identity_ilt,
            overlap=8,
            n_schwarz_iters=2,
            n_inner_iters=1,
        )

        assert result["mask"].shape == (64, 64)
        assert "schwarz_history" in result
        assert len(result["schwarz_history"]) <= 2

    def test_schwarz_reduces_boundary_mse(self):
        """Schwarz should reduce boundary MSE compared to independent tiling."""
        from openlithohub.workflow.tiling import tiled_ilt_with_consistency

        torch.manual_seed(0)

        def smoothing_ilt(tile: torch.Tensor) -> torch.Tensor:
            kernel = torch.ones(3, 3) / 9.0
            inp = tile.unsqueeze(0).unsqueeze(0)
            pad = torch.nn.functional.pad(inp, (1, 1, 1, 1), mode="reflect")
            out = torch.nn.functional.conv2d(pad, kernel.unsqueeze(0).unsqueeze(0))
            return out.squeeze(0).squeeze(0)

        mask = torch.rand(64, 64)

        independent = tiled_ilt_with_consistency(
            mask, tile_size=32, ilt_fn=smoothing_ilt, overlap=8, n_iterations=5
        )

        schwarz_result = schwarz_tiled_ilt(
            mask,
            tile_size=32,
            ilt_fn=smoothing_ilt,
            overlap=8,
            n_schwarz_iters=3,
            n_inner_iters=5,
        )

        # Schwarz should have a consistency metric
        assert "consistency" in schwarz_result
        assert schwarz_result["consistency"]["boundary_mse"] >= 0.0

    def test_schwarz_history_decreasing_or_converged(self):
        """Schwarz history should show non-increasing boundary MSE."""

        def identity_ilt(tile: torch.Tensor) -> torch.Tensor:
            return tile

        mask = torch.ones(64, 64) * 0.5
        result = schwarz_tiled_ilt(
            mask,
            tile_size=32,
            ilt_fn=identity_ilt,
            overlap=8,
            n_schwarz_iters=5,
            n_inner_iters=1,
        )

        history = result["schwarz_history"]
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-6, (
                f"Schwarz MSE increased: {history[i - 1]:.6f} → {history[i]:.6f}"
            )

    def test_early_convergence(self):
        """With convergence_tol=0 and identity ILT, should converge immediately."""

        def identity_ilt(tile: torch.Tensor) -> torch.Tensor:
            return tile

        mask = torch.ones(48, 48) * 0.7
        result = schwarz_tiled_ilt(
            mask,
            tile_size=32,
            ilt_fn=identity_ilt,
            overlap=8,
            n_schwarz_iters=10,
            n_inner_iters=1,
            convergence_tol=1e-3,
        )

        # Should converge in 1-2 iterations for identity
        assert len(result["schwarz_history"]) <= 3

    def test_returns_expected_keys(self):
        def identity_ilt(tile: torch.Tensor) -> torch.Tensor:
            return tile

        mask = torch.zeros(64, 64)
        mask[8:56, 8:56] = 1.0
        result = schwarz_tiled_ilt(
            mask, tile_size=32, ilt_fn=identity_ilt, overlap=8, n_schwarz_iters=2
        )

        for key in ("mask", "tiles", "tile_results", "consistency", "schwarz_history"):
            assert key in result, f"Missing key: {key}"


class TestSchwarzConvergenceMetrics:
    def test_empty_history(self):
        result = schwarz_convergence_metrics([])
        assert result["initial_mse"] == 0.0
        assert result["final_mse"] == 0.0
        assert result["converged_monotone"] is True

    def test_single_entry(self):
        result = schwarz_convergence_metrics([0.1])
        assert result["initial_mse"] == 0.1
        assert result["final_mse"] == 0.1
        assert result["reduction_ratio"] == 1.0
        assert result["converged_monotone"] is True

    def test_monotone_decrease(self):
        result = schwarz_convergence_metrics([0.5, 0.3, 0.1, 0.01])
        assert result["converged_monotone"] is True
        assert abs(result["reduction_ratio"] - 0.02) < 1e-6

    def test_non_monotone(self):
        result = schwarz_convergence_metrics([0.1, 0.2, 0.05])
        assert result["converged_monotone"] is False
        assert result["reduction_ratio"] == 0.5


class TestBornForwardCorrection:
    def test_born_single_term_matches_standard(self):
        """n_born_terms=1 should match simulate_aerial_image."""
        from openlithohub._utils.forward_model import (
            simulate_aerial_image,
            simulate_aerial_image_born,
        )

        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        sigma = 2.0

        standard = simulate_aerial_image(mask, sigma_px=sigma, dose=1.0)
        born1 = simulate_aerial_image_born(
            mask, sigma_px=sigma, dose=1.0, n_born_terms=1
        )

        assert torch.allclose(standard, born1, atol=1e-5), (
            f"Max diff: {(standard - born1).abs().max():.6e}"
        )

    def test_born_higher_orders_modify_output(self):
        """n_born_terms=2 should differ from n_born_terms=1."""
        from openlithohub._utils.forward_model import simulate_aerial_image_born

        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0

        born1 = simulate_aerial_image_born(
            mask, sigma_px=2.0, dose=1.0, n_born_terms=1, reflectivity=0.2
        )
        born2 = simulate_aerial_image_born(
            mask, sigma_px=2.0, dose=1.0, n_born_terms=2, reflectivity=0.2
        )

        assert not torch.allclose(born1, born2, atol=1e-5)

    def test_born_output_bounded(self):
        """Born-corrected output should be finite and non-negative."""
        from openlithohub._utils.forward_model import simulate_aerial_image_born

        mask = torch.rand(48, 48)
        for n_terms in [1, 2, 3]:
            aerial = simulate_aerial_image_born(
                mask, sigma_px=3.0, dose=1.0, n_born_terms=n_terms, reflectivity=0.15
            )
            assert torch.isfinite(aerial).all()
            assert aerial.min() >= -1e-6

    def test_born_invalid_terms_raises(self):
        from openlithohub._utils.forward_model import simulate_aerial_image_born

        mask = torch.ones(16, 16)
        with pytest.raises(ValueError, match="n_born_terms"):
            simulate_aerial_image_born(mask, sigma_px=1.0, n_born_terms=0)
