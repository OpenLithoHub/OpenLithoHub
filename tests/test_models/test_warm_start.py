"""Tests for openlithohub.models.warm_start."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.resist_model import apply_differentiable_resist
from openlithohub.models.warm_start import (
    CandidateScorer,
    GANOPCWarmStart,
    NeuralILTWarmStart,
    WarmStartProvider,
    warm_start_ilt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def target() -> torch.Tensor:
    """64x64 binary target with a centered square feature."""
    t = torch.zeros(64, 64)
    t[16:48, 16:48] = 1.0
    return t


@pytest.fixture()
def target_small() -> torch.Tensor:
    """32x32 binary target for faster ILT convergence tests."""
    t = torch.zeros(32, 32)
    t[8:24, 8:24] = 1.0
    return t


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestGANOPCWarmStart:
    def test_generate_initial_mask_shape(self, target: torch.Tensor) -> None:
        provider = GANOPCWarmStart()
        mask = provider.generate_initial_mask(target)
        assert mask.shape == target.shape

    def test_generate_candidates_shape(self, target: torch.Tensor) -> None:
        provider = GANOPCWarmStart()
        candidates = provider.generate_candidates(target, n_candidates=3)
        assert len(candidates) == 3
        for c in candidates:
            assert c.shape == target.shape

    def test_forward_shape(self, target: torch.Tensor) -> None:
        provider = GANOPCWarmStart()
        mask = provider(target)
        assert mask.shape == target.shape

    def test_satisfies_protocol(self) -> None:
        provider = GANOPCWarmStart()
        assert isinstance(provider, WarmStartProvider)

    def test_3d_input(self) -> None:
        provider = GANOPCWarmStart()
        target_3d = torch.zeros(1, 32, 32)
        target_3d[0, 8:24, 8:24] = 1.0
        mask = provider.generate_initial_mask(target_3d)
        assert mask.shape == target_3d.shape


class TestNeuralILTWarmStart:
    def test_generate_initial_mask_shape(self, target: torch.Tensor) -> None:
        provider = NeuralILTWarmStart()
        mask = provider.generate_initial_mask(target)
        assert mask.shape == target.shape

    def test_generate_candidates_shape(self, target: torch.Tensor) -> None:
        provider = NeuralILTWarmStart()
        candidates = provider.generate_candidates(target, n_candidates=3)
        assert len(candidates) == 3
        for c in candidates:
            assert c.shape == target.shape

    def test_forward_shape(self, target: torch.Tensor) -> None:
        provider = NeuralILTWarmStart()
        mask = provider(target)
        assert mask.shape == target.shape

    def test_satisfies_protocol(self) -> None:
        provider = NeuralILTWarmStart()
        assert isinstance(provider, WarmStartProvider)


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------


class TestCandidateScorer:
    def test_scores_are_finite(self, target: torch.Tensor) -> None:
        scorer = CandidateScorer()
        mask = torch.rand_like(target)
        s = scorer.score(mask, target)
        assert isinstance(s, float)
        assert not torch.isnan(torch.tensor(s))

    def test_score_target_is_low(self, target: torch.Tensor) -> None:
        """Scoring the target itself should be relatively low (good)."""
        scorer = CandidateScorer()
        noise = torch.rand_like(target)
        score_target = scorer.score(target, target)
        score_noise = scorer.score(noise, target)
        assert score_target < score_noise

    def test_rank_candidates_ordered(self, target: torch.Tensor) -> None:
        scorer = CandidateScorer()
        candidates = [
            target,
            torch.rand_like(target),
            torch.zeros_like(target),
        ]
        ranked = scorer.rank_candidates(candidates, target)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores)

    def test_custom_forward_fn(self, target: torch.Tensor) -> None:
        call_count = 0

        def counting_forward(mask: torch.Tensor) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            return mask

        scorer = CandidateScorer(forward_fn=counting_forward)
        scorer.score(target, target)
        assert call_count == 1


# ---------------------------------------------------------------------------
# Diversity test
# ---------------------------------------------------------------------------


class TestDiversity:
    def test_warm_start_candidates_diverse(self, target: torch.Tensor) -> None:
        """Different candidates (from dropout) should not be identical."""
        provider = GANOPCWarmStart()
        candidates = provider.generate_candidates(target, n_candidates=5)
        for i in range(1, len(candidates)):
            diff = (candidates[i] - candidates[0]).abs().sum()
            # At least one pixel differs between any two candidates
            assert diff > 0, f"candidates 0 and {i} are identical"

    def test_neural_ilt_candidates_diverse(self, target: torch.Tensor) -> None:
        provider = NeuralILTWarmStart()
        candidates = provider.generate_candidates(target, n_candidates=5)
        for i in range(1, len(candidates)):
            diff = (candidates[i] - candidates[0]).abs().sum()
            assert diff > 0, f"candidates 0 and {i} are identical"


# ---------------------------------------------------------------------------
# Warm-start vs cold-start convergence test
# ---------------------------------------------------------------------------


def _simple_ilt_loop(
    initial_mask: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 20,
    lr: float = 0.5,
    sigma_px: float = 2.0,
) -> tuple[torch.Tensor, float]:
    """Minimal ILT gradient descent returning (final_mask, final_loss)."""
    mask_logit = (initial_mask.clone().clamp(1e-6, 1 - 1e-6))
    mask_logit = torch.log(mask_logit / (1.0 - mask_logit))
    mask_logit = mask_logit.detach().requires_grad_(True)

    optimizer = torch.optim.Adam([mask_logit], lr=lr)
    final_loss = float("inf")

    for _ in range(iterations):
        optimizer.zero_grad()
        mask_continuous = torch.sigmoid(mask_logit)
        aerial = simulate_aerial_image(mask_continuous, sigma_px=sigma_px)
        resist = apply_differentiable_resist(aerial, threshold=0.5, steepness=50.0)
        loss = nn.functional.mse_loss(resist, target)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    with torch.no_grad():
        final_mask = (torch.sigmoid(mask_logit) > 0.5).float()
    return final_mask, final_loss


class TestWarmStartVsColdStart:
    def test_warm_start_converges_better(self, target_small: torch.Tensor) -> None:
        """Warm-started ILT should reach a lower loss than cold-start (identity)."""
        provider = NeuralILTWarmStart(residual_weight=0.7)
        warm_mask = provider.generate_initial_mask(target_small)

        # Cold start: use the target directly (identity initialization)
        _, cold_loss = _simple_ilt_loop(target_small, target_small, iterations=15)

        # Warm start: use the provider's initial mask
        _, warm_loss = _simple_ilt_loop(warm_mask, target_small, iterations=15)

        assert warm_loss <= cold_loss, (
            f"warm-start loss {warm_loss:.4f} should be <= cold-start loss {cold_loss:.4f}"
        )

    def test_warm_start_ilt_function(self, target_small: torch.Tensor) -> None:
        """Integration test: warm_start_ilt returns valid results."""
        provider = NeuralILTWarmStart(residual_weight=0.7)

        def simple_refiner(mask: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            refined, _ = _simple_ilt_loop(mask, tgt, iterations=5)
            return refined

        result = warm_start_ilt(
            target_small,
            provider,
            ilt_refiner=simple_refiner,
            n_candidates=3,
            top_k=2,
        )

        assert "best_mask" in result
        assert "all_results" in result
        assert "warm_start_mask" in result
        assert "cold_start_mask" in result
        assert "warm_start_score" in result
        assert "cold_start_score" in result

        assert result["best_mask"].shape == target_small.shape
        assert len(result["all_results"]) == 2  # top_k=2
        for mask, score in result["all_results"]:
            assert mask.shape == target_small.shape
            assert isinstance(score, float)

    def test_warm_start_ilt_without_refiner(self, target_small: torch.Tensor) -> None:
        """warm_start_ilt works even without a refiner."""
        provider = GANOPCWarmStart()
        result = warm_start_ilt(
            target_small,
            provider,
            ilt_refiner=None,
            n_candidates=3,
        )
        assert result["best_mask"].shape == target_small.shape
        assert len(result["all_results"]) == 3
