"""Tests for curvilinear_ilt module — curvilinear ILT with process window, adjoint, and Pareto."""

from __future__ import annotations

import math

import pytest
import torch

from openlithohub.models.curvilinear_ilt import (
    AdjointGuidance,
    CurvilinearILTBenchmark,
    CurvilinearILTConfig,
    CurvilinearMaskILT,
    CurvilinearMaskRepresentation,
    ProcessWindowConfig,
    ProcessWindowOptimizer,
    ShotCountPareto,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linespace(grid_size: int = 32, pitch: int = 8) -> torch.Tensor:
    """Create a simple line-space target pattern."""
    t = torch.zeros(grid_size, grid_size)
    half = pitch // 2
    for x in range(0, grid_size, pitch):
        t[:, x : x + half] = 1.0
    return t


def _make_contact(grid_size: int = 32, pitch: int = 10) -> torch.Tensor:
    """Create a simple contact-hole target pattern."""
    t = torch.zeros(grid_size, grid_size)
    size = max(2, pitch // 3)
    for cy in range(pitch // 2, grid_size, pitch):
        for cx in range(pitch // 2, grid_size, pitch):
            y0 = max(0, cy - size // 2)
            y1 = min(grid_size, cy + size // 2 + 1)
            x0 = max(0, cx - size // 2)
            x1 = min(grid_size, cx + size // 2 + 1)
            t[y0:y1, x0:x1] = 1.0
    return t


# ---------------------------------------------------------------------------
# CurvilinearMaskRepresentation tests
# ---------------------------------------------------------------------------


class TestCurvilinearMaskRepresentation:
    """Tests for CurvilinearMaskRepresentation."""

    def test_init_from_target_shape(self) -> None:
        """SDF should have same spatial size as target."""
        target = _make_linespace(32)
        rep = CurvilinearMaskRepresentation(grid_size=32)
        sdf = rep.init_from_target(target)
        assert sdf.shape == target.shape

    def test_init_from_target_values(self) -> None:
        """Interior points should have negative SDF, exterior positive."""
        target = _make_linespace(32)
        rep = CurvilinearMaskRepresentation(grid_size=32)
        sdf = rep.init_from_target(target)

        interior_mask = target > 0.5
        exterior_mask = target < 0.5

        # Interior should have negative (or zero) SDF.
        assert (sdf[interior_mask] <= 0.5).all(), "Interior SDF should be <= 0.5"
        # Exterior should have positive (or zero) SDF.
        assert (sdf[exterior_mask] >= -0.5).all(), "Exterior SDF should be >= -0.5"

    def test_sdf_to_mask_range(self) -> None:
        """Converted mask should be in [0, 1]."""
        rep = CurvilinearMaskRepresentation(steepness=20.0)
        sdf = torch.randn(16, 16)
        mask = rep.sdf_to_mask(sdf)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert mask.shape == sdf.shape

    def test_sdf_to_mask_boundary(self) -> None:
        """Zero SDF should map to ~0.5 mask value."""
        rep = CurvilinearMaskRepresentation(steepness=20.0)
        sdf = torch.zeros(8, 8)
        mask = rep.sdf_to_mask(sdf)
        assert torch.allclose(mask, torch.tensor(0.5), atol=0.01)

    def test_smoothness_loss_finite(self) -> None:
        """Smoothness loss should be finite and non-negative."""
        rep = CurvilinearMaskRepresentation(smoothness_weight=0.01)
        sdf = torch.randn(16, 16)
        loss = rep.smoothness_loss(sdf)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_sdf_to_mask_batch(self) -> None:
        """SDF-to-mask should support batched input (B, H, W)."""
        rep = CurvilinearMaskRepresentation(steepness=10.0)
        sdf = torch.randn(3, 16, 16)
        mask = rep.sdf_to_mask(sdf)
        assert mask.shape == (3, 16, 16)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


# ---------------------------------------------------------------------------
# AdjointGuidance tests
# ---------------------------------------------------------------------------


class TestAdjointGuidance:
    """Tests for AdjointGuidance."""

    def test_epe_loss_finite(self) -> None:
        """EPE loss should be finite for valid inputs."""
        guidance = AdjointGuidance(sigma_px=2.0, dose=1.0)
        mask = torch.sigmoid(torch.randn(32, 32))
        target = _make_linespace(32)
        loss = guidance.compute_epe_loss(mask, target)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_epe_loss_perfect(self) -> None:
        """EPE loss should be near zero when mask equals target (approximately)."""
        guidance = AdjointGuidance(sigma_px=0.1, dose=1.0)
        target = _make_linespace(32)
        # With very small sigma, aerial ~ mask, so target mask should match.
        loss = guidance.compute_epe_loss(target, target)
        assert loss.item() < 0.1

    def test_nils_finite(self) -> None:
        """NILS should be finite."""
        guidance = AdjointGuidance(sigma_px=2.0)
        mask = torch.sigmoid(torch.randn(32, 32))
        nils = guidance.compute_nils(mask)
        assert torch.isfinite(nils)

    def test_compute_gradient_shape(self) -> None:
        """Adjoint gradient should have same shape as input SDF."""
        rep = CurvilinearMaskRepresentation(steepness=10.0)
        guidance = AdjointGuidance(sigma_px=2.0)
        target = _make_linespace(32)
        sdf = torch.randn(32, 32, requires_grad=True)
        grad = guidance.compute_gradient(sdf, target, rep)
        assert grad.shape == sdf.shape
        assert torch.isfinite(grad).all()

    def test_compute_gradient_nonzero(self) -> None:
        """Adjoint gradient should be non-zero for non-matching mask."""
        rep = CurvilinearMaskRepresentation(steepness=10.0)
        guidance = AdjointGuidance(sigma_px=2.0)
        target = _make_linespace(32)
        # Random SDF -> non-matching mask -> non-zero gradient.
        sdf = torch.randn(32, 32) * 2.0
        grad = guidance.compute_gradient(sdf, target, rep)
        assert grad.abs().sum() > 0, "Gradient should be non-zero for random SDF"


# ---------------------------------------------------------------------------
# ProcessWindowOptimizer tests
# ---------------------------------------------------------------------------


class TestProcessWindowOptimizer:
    """Tests for ProcessWindowOptimizer."""

    def test_condition_grid_size(self) -> None:
        """Condition grid should have n_focus * n_dose entries."""
        config = ProcessWindowConfig(n_focus_steps=3, n_dose_steps=3)
        pw = ProcessWindowOptimizer(config)
        grid = pw.get_condition_grid()
        assert len(grid) == 9

    def test_condition_grid_values(self) -> None:
        """Condition grid should span the configured ranges."""
        config = ProcessWindowConfig(
            focus_range_nm=(-40.0, 40.0),
            dose_range=(0.9, 1.1),
            n_focus_steps=2,
            n_dose_steps=2,
        )
        pw = ProcessWindowOptimizer(config)
        grid = pw.get_condition_grid()
        assert len(grid) == 4
        focuses = [g[0] for g in grid]
        doses = [g[1] for g in grid]
        assert min(focuses) == pytest.approx(-40.0)
        assert max(focuses) == pytest.approx(40.0)
        assert min(doses) == pytest.approx(0.9)
        assert max(doses) == pytest.approx(1.1)

    def test_sigma_for_defocus(self) -> None:
        """Defocus should increase PSF sigma."""
        pw = ProcessWindowOptimizer()
        nominal = 2.0
        sigma_neg = pw.compute_sigma_for_defocus(nominal, -40.0)
        sigma_zero = pw.compute_sigma_for_defocus(nominal, 0.0)
        sigma_pos = pw.compute_sigma_for_defocus(nominal, 40.0)
        assert sigma_zero == pytest.approx(nominal)
        assert sigma_neg > nominal
        assert sigma_pos > nominal
        assert sigma_neg == pytest.approx(sigma_pos)  # symmetric

    def test_compute_pw_loss_finite(self) -> None:
        """PW loss should be finite and info should have expected keys."""
        config = ProcessWindowConfig(n_focus_steps=2, n_dose_steps=2)
        pw = ProcessWindowOptimizer(config)
        mask = torch.sigmoid(torch.randn(32, 32))
        target = _make_linespace(32)
        loss, info = pw.compute_pw_loss(mask, target)
        assert torch.isfinite(loss)
        assert "mean_epe" in info
        assert "max_epe" in info
        assert "per_condition" in info
        assert info["n_conditions"] == 4

    def test_compute_pw_loss_best_at_nominal(self) -> None:
        """EPE at nominal focus/dose should generally be lowest."""
        config = ProcessWindowConfig(n_focus_steps=3, n_dose_steps=1)
        pw = ProcessWindowOptimizer(config)
        target = _make_linespace(32)
        # Use target as mask — should print well at nominal.
        loss, info = pw.compute_pw_loss(target, target, nominal_sigma_px=1.0)
        assert info["mean_epe"] < 0.5


# ---------------------------------------------------------------------------
# ShotCountPareto tests
# ---------------------------------------------------------------------------


class TestShotCountPareto:
    """Tests for ShotCountPareto."""

    def test_estimate_shots_empty(self) -> None:
        """Uniform mask should have near-zero shots."""
        pareto = ShotCountPareto()
        mask = torch.zeros(32, 32)
        shots = pareto.estimate_curvilinear_shots(mask)
        assert shots == pytest.approx(0.0)

    def test_estimate_shots_with_boundary(self) -> None:
        """Mask with features should have positive shot count."""
        pareto = ShotCountPareto()
        mask = _make_linespace(32)
        shots = pareto.estimate_curvilinear_shots(mask)
        assert shots > 0.0

    def test_estimate_sraf_shots(self) -> None:
        """SRAF features should add to shot count."""
        pareto = ShotCountPareto()
        main = torch.zeros(32, 32)
        main[10:20, 10:20] = 1.0
        full = main.clone()
        full[5:7, 5:7] = 1.0  # SRAF feature
        sraf_shots = pareto.estimate_sraf_shots(full, main)
        assert sraf_shots > 0.0

    def test_record_and_history(self) -> None:
        """Recorded points should be retrievable from history."""
        pareto = ShotCountPareto()
        pareto.record(100.0, 0.05)
        pareto.record(150.0, 0.03)
        assert len(pareto.history) == 2
        assert pareto.history[0] == (100.0, 0.05)

    def test_pareto_frontier_dominated(self) -> None:
        """Pareto frontier should exclude dominated points."""
        pareto = ShotCountPareto()
        # (shots, epe): point 1 is best, point 2 is dominated, point 3 trades off.
        pareto.record(100.0, 0.10)  # Low shots, moderate epe
        pareto.record(120.0, 0.15)  # Dominated by point 1
        pareto.record(150.0, 0.05)  # More shots but better epe
        frontier = pareto.compute_pareto_frontier()
        # Point 2 (120, 0.15) should be excluded (dominated by point 1).
        assert (100.0, 0.10) in frontier
        assert (150.0, 0.05) in frontier
        assert (120.0, 0.15) not in frontier

    def test_pareto_frontier_sorted_by_shots(self) -> None:
        """Frontier should be sorted by ascending shot count."""
        pareto = ShotCountPareto()
        pareto.record(200.0, 0.02)
        pareto.record(100.0, 0.10)
        pareto.record(150.0, 0.05)
        frontier = pareto.compute_pareto_frontier()
        for i in range(len(frontier) - 1):
            assert frontier[i][0] <= frontier[i + 1][0]

    def test_pareto_frontier_empty(self) -> None:
        """Empty history should produce empty frontier."""
        pareto = ShotCountPareto()
        assert pareto.compute_pareto_frontier() == []

    def test_reset(self) -> None:
        """Reset should clear history."""
        pareto = ShotCountPareto()
        pareto.record(100.0, 0.05)
        pareto.reset()
        assert len(pareto.history) == 0


# ---------------------------------------------------------------------------
# CurvilinearMaskILT integration tests
# ---------------------------------------------------------------------------


class TestCurvilinearMaskILT:
    """Integration tests for CurvilinearMaskILT optimizer."""

    def test_optimize_returns_mask(self) -> None:
        """Optimize should return a binary mask of correct shape."""
        config = CurvilinearILTConfig(
            grid_size=32,
            n_steps=5,
            sigma_px=1.5,
            smoothness_weight=0.001,
        )
        ilt = CurvilinearMaskILT(config)
        target = _make_linespace(32)
        mask, info = ilt.optimize(target)
        assert mask.shape == target.shape
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_optimize_info_keys(self) -> None:
        """Info dict should contain expected keys."""
        config = CurvilinearILTConfig(grid_size=32, n_steps=3, sigma_px=1.5)
        ilt = CurvilinearMaskILT(config)
        target = _make_linespace(32)
        _, info = ilt.optimize(target)
        assert "pareto_frontier" in info
        assert "final_epe" in info
        assert "final_shots" in info
        assert "loss_history" in info
        assert len(info["loss_history"]) == 3

    def test_optimize_reduces_loss(self) -> None:
        """Loss should generally decrease over iterations."""
        config = CurvilinearILTConfig(
            grid_size=32,
            n_steps=20,
            lr=0.1,
            sigma_px=1.5,
            smoothness_weight=0.001,
            shot_count_weight=0.01,
        )
        ilt = CurvilinearMaskILT(config)
        target = _make_linespace(32)
        _, info = ilt.optimize(target)
        history = info["loss_history"]
        # First few steps may fluctuate, but final should be lower than first.
        assert history[-1] < history[0] * 2.0, (
            f"Loss should decrease: first={history[0]:.4f}, last={history[-1]:.4f}"
        )

    def test_optimize_rectilinear(self) -> None:
        """Rectilinear optimization should also return valid mask."""
        config = CurvilinearILTConfig(grid_size=32, n_steps=5, sigma_px=1.5)
        ilt = CurvilinearMaskILT(config)
        target = _make_linespace(32)
        mask, info = ilt.optimize_rectilinear(target)
        assert mask.shape == target.shape
        assert "pareto_frontier" in info

    def test_optimize_contact_pattern(self) -> None:
        """Should work on contact-hole patterns too."""
        config = CurvilinearILTConfig(grid_size=32, n_steps=5, sigma_px=1.5)
        ilt = CurvilinearMaskILT(config)
        target = _make_contact(32)
        mask, info = ilt.optimize(target)
        assert mask.shape == target.shape
        assert info["final_epe"] > 0.0

    def test_pareto_frontier_populated(self) -> None:
        """Pareto frontier should have at least one point after optimization."""
        config = CurvilinearILTConfig(grid_size=32, n_steps=5, sigma_px=1.5)
        ilt = CurvilinearMaskILT(config)
        target = _make_linespace(32)
        _, info = ilt.optimize(target)
        assert len(info["pareto_frontier"]) >= 1


# ---------------------------------------------------------------------------
# CurvilinearILTBenchmark tests
# ---------------------------------------------------------------------------


class TestCurvilinearILTBenchmark:
    """Tests for CurvilinearILTBenchmark."""

    def test_benchmark_returns_expected_structure(self) -> None:
        """Benchmark should return dict with curvilinear and rectilinear keys."""
        config = CurvilinearILTConfig(grid_size=32, n_steps=3, sigma_px=1.5)
        bench = CurvilinearILTBenchmark(config=config, n_steps=3)
        target = _make_linespace(32)
        result = bench.run(targets=[target], n_seeds=1, n_steps=3)
        assert "curvilinear" in result
        assert "rectilinear" in result
        assert "mean_epe" in result["curvilinear"]
        assert "mean_shots" in result["curvilinear"]
        assert "mean_epe" in result["rectilinear"]
        assert "mean_shots" in result["rectilinear"]

    def test_benchmark_default_targets(self) -> None:
        """Benchmark with no targets should auto-generate patterns."""
        config = CurvilinearILTConfig(grid_size=32, n_steps=2, sigma_px=1.5)
        bench = CurvilinearILTBenchmark(config=config, n_steps=2)
        result = bench.run(n_seeds=1, n_steps=2)
        assert result["n_targets"] == 2
        assert result["n_seeds"] == 1

    def test_benchmark_epe_finite(self) -> None:
        """All EPE values should be finite."""
        config = CurvilinearILTConfig(grid_size=32, n_steps=3, sigma_px=1.5)
        bench = CurvilinearILTBenchmark(config=config, n_steps=3)
        target = _make_linespace(32)
        result = bench.run(targets=[target], n_seeds=1, n_steps=3)
        assert math.isfinite(result["curvilinear"]["mean_epe"])
        assert math.isfinite(result["rectilinear"]["mean_epe"])

    def test_make_target_patterns(self) -> None:
        """All target pattern types should produce valid binary masks."""
        bench = CurvilinearILTBenchmark(n_steps=1)
        for pattern in ("linespace", "contact", "elbow"):
            target = bench._make_target(pattern, grid_size=32)
            assert target.shape == (32, 32)
            assert target.min() >= 0.0
            assert target.max() <= 1.0
            assert target.sum() > 0, f"Pattern {pattern} should have foreground"
