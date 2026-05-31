"""Tests for generative SRAF insertion via RL."""

from __future__ import annotations

import torch

from openlithohub.models.generative_sraf import (
    LithographicImprovementScorer,
    SRAFConfig,
    SRAFGenerator,
    SRAFPipeline,
    SRAFRLTrainer,
)


def _toy_mask(size: int = 64) -> torch.Tensor:
    t = torch.zeros(size, size)
    t[16:48, 16:48] = 1.0
    return t


def _toy_target(size: int = 64) -> torch.Tensor:
    t = torch.zeros(size, size)
    t[18:46, 18:46] = 1.0
    return t


def test_sraf_generator_forward_shape() -> None:
    torch.manual_seed(0)
    gen = SRAFGenerator(SRAFConfig(mask_size=64, max_sraf_count=20))
    mask = _toy_mask()
    placements = gen(mask)
    assert placements.shape == (1, 20, 5)


def test_sraf_generator_batch_shape() -> None:
    torch.manual_seed(0)
    gen = SRAFGenerator(SRAFConfig(mask_size=64, max_sraf_count=10))
    masks = torch.stack([_toy_mask(), _toy_mask()])
    placements = gen(masks)
    assert placements.shape == (2, 10, 5)


def test_sraf_placements_valid_coordinates() -> None:
    torch.manual_seed(0)
    gen = SRAFGenerator(SRAFConfig(mask_size=64, max_sraf_count=15))
    mask = _toy_mask()
    placements = gen(mask)
    x = placements[0, :, 0]
    y = placements[0, :, 1]
    conf = placements[0, :, 4]
    assert (x >= 0).all() and (x <= 1).all()
    assert (y >= 0).all() and (y <= 1).all()
    assert (conf >= 0).all() and (conf <= 1).all()


def test_render_srafs_produces_features() -> None:
    torch.manual_seed(0)
    gen = SRAFGenerator(SRAFConfig(mask_size=64, max_sraf_count=20))
    mask = _toy_mask()
    placements = gen(mask)
    rendered = gen.render_srafs(mask, placements)
    assert rendered.shape == mask.shape
    assert rendered.max() >= 0.5


def test_scorer_returns_scalar() -> None:
    torch.manual_seed(42)
    scorer = LithographicImprovementScorer()
    mask = _toy_mask()
    target = _toy_target()
    score = scorer.score(mask, mask, target)
    assert isinstance(score, float)


def test_trainer_train_step_runs() -> None:
    torch.manual_seed(0)
    config = SRAFConfig(mask_size=64, max_sraf_count=10)
    trainer = SRAFRLTrainer(config=config)
    mask = _toy_mask().unsqueeze(0)
    target = _toy_target().unsqueeze(0)
    scorer = LithographicImprovementScorer()
    result = trainer.train_step(mask, target, scorer)
    assert "loss" in result
    assert "improvement" in result
    assert isinstance(result["loss"], float)


def test_pipeline_insert_srafs_returns_mask() -> None:
    torch.manual_seed(0)
    pipeline = SRAFPipeline(config=SRAFConfig(mask_size=64))
    mask = _toy_mask()
    target = _toy_target()
    result = pipeline.insert_srafs(mask, target, n_candidates=2)
    assert result.shape == mask.shape


def test_pipeline_evaluate_returns_metrics() -> None:
    torch.manual_seed(0)
    pipeline = SRAFPipeline(config=SRAFConfig(mask_size=64))
    mask = _toy_mask()
    target = _toy_target()
    metrics = pipeline.evaluate(mask, mask, target)
    assert "improvement" in metrics
    assert "improved" in metrics
    assert isinstance(metrics["improvement"], float)
    assert isinstance(metrics["improved"], bool)


def test_pipeline_benchmark_runs() -> None:
    torch.manual_seed(0)
    pipeline = SRAFPipeline(config=SRAFConfig(mask_size=64))
    masks = [_toy_mask(), _toy_mask()]
    targets = [_toy_target(), _toy_target()]
    stats = pipeline.benchmark(masks, targets)
    assert "avg_improvement" in stats
    assert "improvement_rate" in stats
    assert stats["n_samples"] == 2.0


def test_deterministic_with_seed() -> None:
    def _run() -> torch.Tensor:
        torch.manual_seed(123)
        gen = SRAFGenerator(SRAFConfig(mask_size=64, max_sraf_count=10))
        return gen(_toy_mask())

    r1 = _run()
    r2 = _run()
    assert torch.allclose(r1, r2)
