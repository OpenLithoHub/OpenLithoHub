"""Tests for diffusion_mask module — latent diffusion mask synthesis."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from openlithohub.models.diffusion_mask import (
    DiffusionMaskBenchmark,
    DiffusionMaskConfig,
    DiffusionMaskSynthesis,
    LithoGuidance,
    MaskDiffusionUNet,
    MaskLatentDecoder,
    MaskLatentEncoder,
    _reparameterize,
    _sinusoidal_embedding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_mask(batch: int = 2, size: int = 32) -> torch.Tensor:
    return (torch.rand(batch, 1, size, size) > 0.5).float()


# ---------------------------------------------------------------------------
# 1. test_mask_latent_encoder_shape
# ---------------------------------------------------------------------------


def test_mask_latent_encoder_shape() -> None:
    enc = MaskLatentEncoder(latent_channels=16, hidden_channels=32, n_down_layers=2)
    x = _random_mask(batch=4, size=32)
    mean, logvar = enc(x)
    assert mean.shape == (4, 16, 8, 8)
    assert logvar.shape == (4, 16, 8, 8)


# ---------------------------------------------------------------------------
# 2. test_mask_latent_decoder_shape
# ---------------------------------------------------------------------------


def test_mask_latent_decoder_shape() -> None:
    dec = MaskLatentDecoder(latent_channels=16, hidden_channels=32, n_up_layers=2)
    z = torch.randn(4, 16, 8, 8)
    out = dec(z)
    assert out.shape == (4, 1, 32, 32)
    assert out.min() >= 0.0 and out.max() <= 1.0


# ---------------------------------------------------------------------------
# 3. test_autoencoder_roundtrip
# ---------------------------------------------------------------------------


def test_autoencoder_roundtrip() -> None:
    enc = MaskLatentEncoder(latent_channels=8, hidden_channels=16, n_down_layers=2)
    dec = MaskLatentDecoder(latent_channels=8, hidden_channels=16, n_up_layers=2)
    masks = _random_mask(batch=4, size=32)
    mean, logvar = enc(masks)
    z = _reparameterize(mean, logvar)
    recon = dec(z)
    assert recon.shape == masks.shape
    assert torch.isfinite(recon).all()


# ---------------------------------------------------------------------------
# 4. test_litho_guidance_gradient
# ---------------------------------------------------------------------------


def test_litho_guidance_gradient() -> None:
    dec = MaskLatentDecoder(latent_channels=8, hidden_channels=16, n_up_layers=2)
    guidance = LithoGuidance(dec, sigma_px=1.5, dose=1.0)
    z = torch.randn(2, 8, 8, 8, requires_grad=True)
    target = _random_mask(batch=1, size=32)[0, 0]
    grad = guidance.compute_guidance(z, target, guidance_scale=2.0)
    assert grad.shape == z.shape
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, "Guidance gradient should be nonzero"


# ---------------------------------------------------------------------------
# 5. test_mask_diffusion_unet_forward
# ---------------------------------------------------------------------------


def test_mask_diffusion_unet_forward() -> None:
    unet = MaskDiffusionUNet(latent_channels=8, context_channels=8, base_channels=(8, 16, 24, 32))
    z = torch.randn(2, 8, 8, 8)
    t = torch.randint(0, 1000, (2,))
    context = torch.randn(2, 8, 8, 8)
    out = unet(z, t, context)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 6. test_mask_diffusion_sampling
# ---------------------------------------------------------------------------


def test_mask_diffusion_sampling() -> None:
    cfg = DiffusionMaskConfig(
        latent_channels=4,
        hidden_channels=8,
        spatial_stride=2,
        n_diffusion_steps=10,
        n_sampling_steps=3,
        guidance_scale=0.0,
    )
    model = DiffusionMaskSynthesis(config=cfg)
    target = _random_mask(batch=1, size=16)[0, 0]
    result = model.synthesize(target, n_candidates=2, n_steps=3, guidance_scale=0.0)
    assert "candidates" in result
    assert len(result["candidates"]) == 2
    assert "scores" in result
    assert len(result["scores"]) == 2
    assert "best_idx" in result
    assert "best_mask" in result


# ---------------------------------------------------------------------------
# 7. test_diffusion_mask_synthesize_candidates
# ---------------------------------------------------------------------------


def test_diffusion_mask_synthesize_candidates() -> None:
    cfg = DiffusionMaskConfig(
        latent_channels=4,
        hidden_channels=8,
        spatial_stride=2,
        n_diffusion_steps=10,
        n_sampling_steps=3,
        guidance_scale=0.0,
    )
    model = DiffusionMaskSynthesis(config=cfg)
    target = _random_mask(batch=1, size=16)[0, 0]
    result = model.synthesize(target, n_candidates=8, n_steps=3)
    assert len(result["candidates"]) == 8
    for c in result["candidates"]:
        assert c.ndim == 2
        assert c.shape == (16, 16)
    assert result["best_idx"] in range(8)


# ---------------------------------------------------------------------------
# 8. test_manufacturable_filtering
# ---------------------------------------------------------------------------


def test_manufacturable_filtering() -> None:
    big_mask = torch.ones(32, 32)
    tiny_mask = torch.zeros(32, 32)
    tiny_mask[10, 10] = 1.0
    results = DiffusionMaskSynthesis._filter_manufacturable(
        [big_mask, tiny_mask], min_feature_px=3,
    )
    assert results[0] is True
    assert results[1] is False


# ---------------------------------------------------------------------------
# 9. test_candidate_scoring
# ---------------------------------------------------------------------------


def test_candidate_scoring() -> None:
    target = torch.zeros(32, 32)
    good_mask = torch.zeros(32, 32)
    bad_mask = torch.ones(32, 32)
    scores = DiffusionMaskSynthesis._score_candidates(
        [good_mask, bad_mask], target, sigma_px=1.0,
    )
    assert len(scores) == 2
    assert scores[0] < scores[1], "Good mask should score lower (better)"


# ---------------------------------------------------------------------------
# 10. test_compare_vs_grpo
# ---------------------------------------------------------------------------


def test_compare_vs_grpo() -> None:
    cfg = DiffusionMaskConfig(
        latent_channels=4,
        hidden_channels=8,
        spatial_stride=2,
        n_diffusion_steps=10,
        n_sampling_steps=2,
        guidance_scale=0.0,
    )
    model = DiffusionMaskSynthesis(config=cfg)

    grpo = MagicMock()
    grpo.generate_group.return_value = [_random_mask(batch=1, size=16)[0, 0] for _ in range(4)]

    target = _random_mask(batch=1, size=16)[0, 0]
    result = model.compare_vs_grpo(target, grpo, n_seeds=1)

    assert "diffusion" in result
    assert "grpo" in result
    assert "mean_score" in result["diffusion"]
    assert "mean_score" in result["grpo"]
    assert "mean_diversity" in result["diffusion"]
    assert "total_mrc_violations" in result["diffusion"]
    assert "total_mrc_violations" in result["grpo"]
    assert result["n_seeds"] == 1


# ---------------------------------------------------------------------------
# 11. test_benchmark_runs
# ---------------------------------------------------------------------------


def test_benchmark_runs() -> None:
    cfg = DiffusionMaskConfig(
        latent_channels=4,
        hidden_channels=8,
        spatial_stride=2,
        n_diffusion_steps=10,
        n_sampling_steps=2,
        guidance_scale=0.0,
    )
    diffusion = DiffusionMaskSynthesis(config=cfg)
    grpo = MagicMock()
    grpo.generate_group.return_value = [_random_mask(batch=1, size=16)[0, 0] for _ in range(4)]

    bench = DiffusionMaskBenchmark(diffusion, grpo, cfg)
    targets = [_random_mask(batch=1, size=16)[0, 0], _random_mask(batch=1, size=16)[0, 0]]
    result = bench.run(targets, n_candidates=4, n_seeds=1)

    assert "per_target" in result
    assert len(result["per_target"]) == 2
    assert "aggregate" in result
    assert result["aggregate"]["n_targets"] == 2
    assert "diffusion_mean_score" in result["aggregate"]
    assert "grpo_mean_score" in result["aggregate"]


# ---------------------------------------------------------------------------
# 12. test_with_decision_gate
# ---------------------------------------------------------------------------


def test_with_decision_gate() -> None:
    cfg = DiffusionMaskConfig(
        latent_channels=4,
        hidden_channels=8,
        spatial_stride=2,
        n_diffusion_steps=10,
        n_sampling_steps=2,
        guidance_scale=0.0,
    )
    model = DiffusionMaskSynthesis(config=cfg)
    target = _random_mask(batch=1, size=16)[0, 0]
    result = model.synthesize(target, n_candidates=4, n_steps=2)

    try:
        from diff_surrogate.decision import AcceptRejectGate, MultiCandidateDecision, DecisionVerdict

        gate = AcceptRejectGate(min_coverage=0.8)
        mcd = MultiCandidateDecision()

        masks = result["candidates"]
        scores = torch.tensor(result["scores"])
        lower = scores - 0.1
        upper = scores + 0.1
        candidates_t = torch.stack([m.flatten() for m in masks])

        verdict, reasons = gate.evaluate(scores, lower, upper, coverage=0.95)
        assert verdict in (DecisionVerdict.ACCEPT, DecisionVerdict.REJECT, DecisionVerdict.UNCERTAIN)
        assert "mean_bandwidth" in reasons

        best_idx, mcd_scores, verdicts = mcd.select(candidates_t, scores, lower, upper, maximize=False)
        assert 0 <= best_idx < len(masks)
        assert len(verdicts) == len(masks)
    except ImportError:
        pytest.skip("diff_surrogate not installed")


# ---------------------------------------------------------------------------
# Additional helper tests
# ---------------------------------------------------------------------------


def test_sinusoidal_embedding_shape() -> None:
    t = torch.tensor([0, 100, 500])
    emb = _sinusoidal_embedding(t, 64)
    assert emb.shape == (3, 64)


def test_reparameterize_shape() -> None:
    mean = torch.randn(4, 8, 4, 4)
    logvar = torch.randn(4, 8, 4, 4)
    z = _reparameterize(mean, logvar)
    assert z.shape == mean.shape
    assert torch.isfinite(z).all()
