"""Tests for mask topology simplification with MBMW post-processing."""

from __future__ import annotations

import torch

from openlithohub.workflow.topology_simplification import (
    CurvilinearSimplifier,
    MBMWPostProcessor,
    SimplificationConfig,
    TopologyAnalyzer,
    TopologyPipeline,
)


def _complex_mask(size: int = 64, seed: int = 42) -> torch.Tensor:
    """Generate a complex curvilinear-like mask with multiple features."""
    torch.manual_seed(seed)
    m = torch.zeros(size, size)
    m[8:24, 8:24] = 1.0
    m[30:50, 30:50] = 1.0
    m[4:10, 40:55] = 1.0
    noise = torch.randn(size, size) * 0.05
    m = (m + noise).clamp(0.0, 1.0)
    return m


class TestTopologyAnalyzer:
    def test_count_edges_returns_positive_int(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        count = TopologyAnalyzer.count_edges(mask)
        assert isinstance(count, int)
        assert count > 0

    def test_complexity_score_in_unit_interval(self):
        mask = _complex_mask()
        score = TopologyAnalyzer.complexity_score(mask)
        assert score.ndim == 0
        assert 0.0 <= float(score.item()) <= 1.0

    def test_empty_mask_zero_complexity(self):
        mask = torch.zeros(16, 16)
        score = TopologyAnalyzer.complexity_score(mask)
        assert float(score.item()) == 0.0

    def test_count_connected_components(self):
        mask = torch.zeros(32, 32)
        mask[2:8, 2:8] = 1.0
        mask[20:28, 20:28] = 1.0
        n = TopologyAnalyzer.count_connected_components(mask)
        assert n == 2

    def test_mbmw_compatible(self):
        simple = torch.zeros(16, 16)
        simple[4:12, 4:12] = 1.0
        assert TopologyAnalyzer.mbmw_compatible(simple, target_edges=1000)


class TestCurvilinearSimplifier:
    def test_reduces_complexity(self):
        mask = _complex_mask()
        config = SimplificationConfig(min_feature_size=2)
        simplifier = CurvilinearSimplifier(smoothing_sigma=2.0)
        simplified = simplifier(mask, config)
        orig_edges = TopologyAnalyzer.count_edges(mask)
        simp_edges = TopologyAnalyzer.count_edges(simplified)
        assert simp_edges <= orig_edges

    def test_preserves_main_features(self):
        mask = torch.zeros(64, 64)
        mask[16:48, 16:48] = 1.0
        config = SimplificationConfig(min_feature_size=2)
        simplifier = CurvilinearSimplifier(smoothing_sigma=1.0)
        simplified = simplifier(mask, config)
        assert (simplified > 0.5).sum().item() > 100

    def test_simplification_loss_nonnegative(self):
        original = _complex_mask()
        config = SimplificationConfig()
        simplifier = CurvilinearSimplifier()
        simplified = simplifier(original, config)
        loss = simplifier.simplification_loss(original, simplified, config)
        assert loss.item() >= 0.0


class TestMBMWPostProcessor:
    def test_snap_to_grid_produces_grid_aligned_mask(self):
        mask = torch.tensor([[0.3, 0.7], [1.2, -0.1]])
        snapped = MBMWPostProcessor.snap_to_grid(mask, grid_size=1.0)
        assert torch.allclose(snapped, snapped.round().float(), atol=1e-6)

    def test_remove_small_features_removes_small_components(self):
        mask = torch.zeros(32, 32)
        mask[4:20, 4:20] = 1.0
        mask[25, 25] = 1.0
        mask[26, 26] = 1.0
        cleaned = MBMWPostProcessor.remove_small_features(mask, min_size=3)
        assert cleaned[25, 25].item() == 0.0
        assert cleaned[26, 26].item() == 0.0
        assert cleaned[10, 10].item() > 0.0

    def test_ensure_mrc_compliance(self):
        mask = torch.zeros(32, 32)
        mask[10:20, 10:20] = 1.0
        compliant = MBMWPostProcessor.ensure_mrc_compliance(mask, min_space=3, min_width=3)
        assert compliant.shape == mask.shape
        assert compliant.min() >= 0.0
        assert compliant.max() <= 1.0


class TestTopologyPipeline:
    def test_simplify_returns_mask_and_metrics(self):
        mask = _complex_mask()
        config = SimplificationConfig()
        pipeline = TopologyPipeline()
        result = pipeline.simplify(mask, config)
        assert result.mask.shape == mask.shape
        assert isinstance(result.original_edges, int)
        assert isinstance(result.simplified_edges, int)
        assert isinstance(result.complexity_reduction, float)
        assert isinstance(result.mbmw_compatible, bool)

    def test_compare_returns_quality_metrics(self):
        original = _complex_mask()
        pipeline = TopologyPipeline()
        config = SimplificationConfig()
        result = pipeline.simplify(original, config)
        comparison = pipeline.compare(original, result.mask)
        assert comparison.edge_placement_error >= 0.0
        assert comparison.original_edge_count > 0
        assert 0.0 <= comparison.area_preservation <= 1.0

    def test_benchmark_sweeps_configs(self):
        mask = _complex_mask()
        pipeline = TopologyPipeline()
        configs = [
            SimplificationConfig(min_feature_size=1),
            SimplificationConfig(min_feature_size=5),
        ]
        results = pipeline.benchmark([mask], None, configs)
        assert len(results) == 1
        assert len(results[0]) == 2

    def test_deterministic_with_seed(self):
        torch.manual_seed(123)
        mask = _complex_mask(seed=7)
        config = SimplificationConfig()
        pipeline1 = TopologyPipeline()
        r1 = pipeline1.simplify(mask, config)

        torch.manual_seed(123)
        pipeline2 = TopologyPipeline()
        r2 = pipeline2.simplify(mask, config)

        assert torch.allclose(r1.mask, r2.mask, atol=1e-6)
        assert r1.original_edges == r2.original_edges
        assert r1.simplified_edges == r2.simplified_edges
