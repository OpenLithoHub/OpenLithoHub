"""Tests for openlithohub.models.grpo_warm_start."""

from __future__ import annotations

import pytest
import torch

from openlithohub.models.grpo_warm_start import (
    GRPOConfig,
    GRPOWarmStart,
    StyleConditioning,
)
from openlithohub.models.posterior_warm_start import PosteriorWarmStart
from openlithohub.models.warm_start import CandidateScorer
from openlithohub.workflow.layer_purpose import LayerPurpose

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def target() -> torch.Tensor:
    t = torch.zeros(32, 32)
    t[8:24, 8:24] = 1.0
    return t


@pytest.fixture()
def scorer() -> CandidateScorer:
    return CandidateScorer()


@pytest.fixture()
def layer_purpose() -> LayerPurpose:
    return LayerPurpose(layer=1, datatype=0, purpose="drawing")


# ---------------------------------------------------------------------------
# GRPOConfig
# ---------------------------------------------------------------------------


class TestGRPOConfig:
    def test_defaults(self) -> None:
        cfg = GRPOConfig()
        assert cfg.group_size == 4
        assert cfg.clip_ratio == 0.2
        assert cfg.entropy_coeff == 0.01
        assert cfg.ilt_guidance_weight == 0.5
        assert cfg.n_candidates == 16
        assert cfg.style_aware is True

    def test_custom(self) -> None:
        cfg = GRPOConfig(group_size=8, clip_ratio=0.1, entropy_coeff=0.05)
        assert cfg.group_size == 8
        assert cfg.clip_ratio == 0.1
        assert cfg.entropy_coeff == 0.05


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_wraps_posterior(self) -> None:
        vae = PosteriorWarmStart(latent_dim=8, base_channels=16)
        grpo = GRPOWarmStart(vae=vae)
        assert grpo.vae is vae

    def test_default_vae_created(self) -> None:
        grpo = GRPOWarmStart()
        assert isinstance(grpo.vae, PosteriorWarmStart)

    def test_custom_config(self) -> None:
        cfg = GRPOConfig(group_size=2)
        grpo = GRPOWarmStart(config=cfg)
        assert grpo.config.group_size == 2


# ---------------------------------------------------------------------------
# generate_group
# ---------------------------------------------------------------------------


class TestGenerateGroup:
    def test_correct_count(self, target: torch.Tensor) -> None:
        grpo = GRPOWarmStart(config=GRPOConfig(group_size=4))
        group = grpo.generate_group(target)
        assert len(group) == 4

    def test_correct_shape(self, target: torch.Tensor) -> None:
        grpo = GRPOWarmStart()
        group = grpo.generate_group(target)
        for c in group:
            assert c.shape == target.shape

    def test_with_layer_purpose(self, target: torch.Tensor, layer_purpose: LayerPurpose) -> None:
        grpo = GRPOWarmStart()
        group = grpo.generate_group(target, layer_purpose=layer_purpose)
        assert len(group) == grpo.config.group_size


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------


class TestComputeReward:
    def test_returns_per_candidate(self, target: torch.Tensor, scorer: CandidateScorer) -> None:
        grpo = GRPOWarmStart()
        candidates = [torch.rand_like(target) for _ in range(4)]
        rewards = grpo.compute_reward(candidates, target, scorer)
        assert rewards.shape == (4,)

    def test_negated_scores(self, target: torch.Tensor) -> None:
        grpo = GRPOWarmStart()

        def _mock_scorer(mask: torch.Tensor, tgt: torch.Tensor) -> float:
            return 1.0

        scorer = CandidateScorer()
        scorer.score = _mock_scorer  # type: ignore[assignment]
        candidates = [torch.rand_like(target) for _ in range(3)]
        rewards = grpo.compute_reward(candidates, target, scorer)
        assert torch.allclose(rewards, torch.tensor([-1.0, -1.0, -1.0]))


# ---------------------------------------------------------------------------
# Group-relative advantages
# ---------------------------------------------------------------------------


class TestGroupRelativeAdvantages:
    def test_zero_mean_unit_std(self) -> None:
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mean = rewards.mean()
        std = rewards.std()
        advantages = (rewards - mean) / std
        assert abs(advantages.mean().item()) < 1e-6
        assert abs(advantages.std().item() - 1.0) < 1e-4

    def test_constant_rewards_handled(self) -> None:
        rewards = torch.tensor([5.0, 5.0, 5.0])
        std = rewards.std()
        assert std < 1e-8


# ---------------------------------------------------------------------------
# GRPO step
# ---------------------------------------------------------------------------


class TestGRPOStep:
    def test_parameters_updated(self, target: torch.Tensor, scorer: CandidateScorer) -> None:
        torch.manual_seed(0)
        grpo = GRPOWarmStart(config=GRPOConfig(group_size=2))
        params_before = [p.clone() for p in grpo.vae.parameters()]
        grpo.grpo_step(target, scorer)
        any_changed = any(
            not torch.equal(p, pb)
            for p, pb in zip(grpo.vae.parameters(), params_before, strict=True)
        )
        assert any_changed, "GRPO step did not update any parameters"

    def test_returns_metrics(self, target: torch.Tensor, scorer: CandidateScorer) -> None:
        grpo = GRPOWarmStart()
        info = grpo.grpo_step(target, scorer)
        assert "loss" in info
        assert "policy_loss" in info
        assert "ilt_loss" in info
        assert "mean_reward" in info
        assert all(torch.isfinite(torch.tensor(v)) for v in info.values())

    def test_gradient_flows(self, target: torch.Tensor, scorer: CandidateScorer) -> None:
        grpo = GRPOWarmStart(config=GRPOConfig(group_size=2))
        grads = []
        grpo.vae.zero_grad()
        grpo.grpo_step(target, scorer)
        for p in grpo.vae.parameters():
            if p.grad is not None:
                grads.append(p.grad.norm().item())
        assert len(grads) > 0, "No gradients computed"


# ---------------------------------------------------------------------------
# Clipped surrogate
# ---------------------------------------------------------------------------


class TestClippedSurrogate:
    def test_clip_bounds(self) -> None:
        cfg = GRPOConfig(clip_ratio=0.2)
        ratio = torch.tensor([0.5, 1.0, 1.5, 2.0])
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
        assert clipped.min() >= 0.8
        assert clipped.max() <= 1.2


# ---------------------------------------------------------------------------
# Style conditioning
# ---------------------------------------------------------------------------


class TestStyleConditioning:
    def test_output_shape(self, layer_purpose: LayerPurpose) -> None:
        sc = StyleConditioning(embed_dim=16)
        vec = sc(layer_purpose)
        assert vec.shape == (16,)

    def test_different_purposes_differ(self) -> None:
        torch.manual_seed(0)
        sc = StyleConditioning(embed_dim=16)
        lp1 = LayerPurpose(layer=1, datatype=0, purpose="drawing")
        lp2 = LayerPurpose(layer=2, datatype=2, purpose="pin")
        v1 = sc(lp1)
        v2 = sc(lp2)
        assert not torch.allclose(v1, v2)

    def test_unknown_purpose(self) -> None:
        sc = StyleConditioning(embed_dim=8)
        lp = LayerPurpose(layer=1, datatype=0, purpose=None)
        vec = sc(lp)
        assert vec.shape == (8,)

    def test_grpo_style_conditioning(self, layer_purpose: LayerPurpose) -> None:
        grpo = GRPOWarmStart(style_embed_dim=16)
        vec = grpo.style_aware_conditioning(layer_purpose)
        assert vec.shape == (16,)


# ---------------------------------------------------------------------------
# ILT guidance loss
# ---------------------------------------------------------------------------


class TestILTCuidanceLoss:
    def test_computed(self, target: torch.Tensor) -> None:
        grpo = GRPOWarmStart()
        candidates = [torch.rand_like(target) for _ in range(3)]
        loss = grpo.compute_ilt_guidance_loss(candidates, target)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_zero_for_identical(self, target: torch.Tensor) -> None:
        grpo = GRPOWarmStart()
        loss = grpo.compute_ilt_guidance_loss([target], target)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# train_loop
# ---------------------------------------------------------------------------


class TestTrainLoop:
    def test_runs_on_toy_data(self, target: torch.Tensor, scorer: CandidateScorer) -> None:
        torch.manual_seed(0)
        cfg = GRPOConfig(group_size=2)
        grpo = GRPOWarmStart(config=cfg)
        targets = [target]
        history = grpo.train_loop(targets, scorer, n_epochs=2)
        assert len(history) == 2
        for step in history:
            assert "loss" in step


# ---------------------------------------------------------------------------
# evaluate_escape
# ---------------------------------------------------------------------------


class TestEvaluateEscape:
    def test_returns_comparison(self, target: torch.Tensor) -> None:
        torch.manual_seed(0)
        grpo = GRPOWarmStart(config=GRPOConfig(group_size=2))
        result = grpo.evaluate_escape(target, n_attempts=4)
        assert "grpo_mean" in result
        assert "baseline_mean" in result
        assert "improvement" in result
        assert len(result["grpo_scores"]) == 4
        assert len(result["baseline_scores"]) == 4

    def test_with_custom_baseline(self, target: torch.Tensor) -> None:
        grpo = GRPOWarmStart()
        baseline = PosteriorWarmStart(latent_dim=8, base_channels=16)
        result = grpo.evaluate_escape(target, n_attempts=2, baseline_model=baseline)
        assert isinstance(result["grpo_mean"], float)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_grpo_output_compatible_with_scorer(
        self, target: torch.Tensor, scorer: CandidateScorer
    ) -> None:
        grpo = GRPOWarmStart()
        candidates = grpo.generate_group(target)
        for c in candidates:
            score = scorer.score(c, target)
            assert isinstance(score, float)
            assert score >= 0

    def test_deterministic_with_seed(self, target: torch.Tensor) -> None:
        torch.manual_seed(42)
        grpo1 = GRPOWarmStart()
        masks1 = grpo1.generate_group(target)

        torch.manual_seed(42)
        grpo2 = GRPOWarmStart()
        masks2 = grpo2.generate_group(target)

        for m1, m2 in zip(masks1, masks2, strict=True):
            assert torch.allclose(m1, m2)
