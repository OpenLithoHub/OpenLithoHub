"""Tests for GPU tile-batch benchmark and ICCAD13 evaluation (O9.2)."""

from __future__ import annotations

import pytest
import torch

from openlithohub.workflow.gpu_tiling_benchmark import (
    GPUTileBatchProcessor,
    ICCAD13Benchmark,
    TileBatchConfig,
    TileBatchResult,
    TilingResidualRegression,
)

# ---------------------------------------------------------------------------
# TileBatchConfig
# ---------------------------------------------------------------------------


class TestTileBatchConfig:
    def test_defaults_reasonable(self):
        cfg = TileBatchConfig()
        assert cfg.tile_size == 64
        assert cfg.overlap == 8
        assert cfg.n_tiles_x == 4
        assert cfg.n_tiles_y == 4
        assert cfg.batch_size == 4
        assert cfg.n_schwarz_iterations == 10
        assert cfg.device == "cpu"
        assert cfg.overlap < cfg.tile_size
        assert cfg.batch_size >= 1
        assert cfg.n_schwarz_iterations >= 1


# ---------------------------------------------------------------------------
# GPUTileBatchProcessor
# ---------------------------------------------------------------------------


class TestGPUTileBatchProcessor:
    def test_runs_on_cpu_with_synthetic_layout(self):
        cfg = TileBatchConfig(
            tile_size=32,
            overlap=4,
            n_tiles_x=2,
            n_tiles_y=2,
            n_schwarz_iterations=2,
        )
        proc = GPUTileBatchProcessor(seed=42)
        result = proc.process_tiles(config=cfg)

        assert isinstance(result, TileBatchResult)
        assert result.total_tiles >= 1
        assert result.total_time_ms > 0.0
        assert result.boundary_residual >= 0.0
        assert result.device == "cpu"

    def test_tile_batch_result_has_all_fields(self):
        cfg = TileBatchConfig(
            tile_size=32,
            overlap=4,
            n_schwarz_iterations=1,
        )
        proc = GPUTileBatchProcessor(seed=0)
        result = proc.process_tiles(config=cfg)

        expected_fields = [
            "tile_config", "total_tiles", "forward_time_ms",
            "schwarz_time_ms", "total_time_ms", "peak_memory_mb",
            "boundary_residual", "convergence_achieved", "device",
        ]
        for field in expected_fields:
            assert hasattr(result, field), f"Missing field: {field}"

    def test_benchmark_scalability_runs_multiple_configs(self):
        configs = [
            TileBatchConfig(
                tile_size=32, overlap=4, n_tiles_x=2, n_tiles_y=2, n_schwarz_iterations=1
            ),
            TileBatchConfig(
                tile_size=32, overlap=4, n_tiles_x=3, n_tiles_y=3, n_schwarz_iterations=1
            ),
        ]
        proc = GPUTileBatchProcessor(seed=0)
        results = proc.benchmark_scalability(configs)

        assert len(results) == 2
        for r in results:
            assert isinstance(r, TileBatchResult)
            assert r.total_tiles >= 1

    def test_deterministic_with_seed(self):
        cfg = TileBatchConfig(
            tile_size=32,
            overlap=4,
            n_schwarz_iterations=2,
        )
        r1 = GPUTileBatchProcessor(seed=7).process_tiles(config=cfg)
        r2 = GPUTileBatchProcessor(seed=7).process_tiles(config=cfg)
        assert r1.boundary_residual == pytest.approx(r2.boundary_residual, rel=1e-6)

    def test_with_explicit_mask_layout(self):
        torch.manual_seed(0)
        mask = (torch.rand(64, 64) > 0.5).float()
        cfg = TileBatchConfig(tile_size=32, overlap=4, n_schwarz_iterations=2)
        proc = GPUTileBatchProcessor(seed=0)
        result = proc.process_tiles(mask_layout=mask, config=cfg)
        assert result.total_tiles >= 1


# ---------------------------------------------------------------------------
# ICCAD13Benchmark
# ---------------------------------------------------------------------------


class TestICCAD13Benchmark:
    def test_evaluate_design_returns_expected_metrics(self):
        torch.manual_seed(0)
        mask = (torch.rand(64, 64) > 0.5).float()
        target = mask.clone()

        bench = ICCAD13Benchmark(seed=0)
        result = bench.evaluate_design(mask, target, metrics=["EPE", "PVB", "MRC"])

        assert "EPE" in result
        assert "PVB" in result
        assert "MRC" in result
        assert result["PVB"] == pytest.approx(0.0, abs=1e-6)

    def test_generate_synthetic_case(self):
        bench = ICCAD13Benchmark(seed=42)
        mask, target = bench.generate_synthetic_case(size=128, complexity="medium")

        assert mask.shape == (128, 128)
        assert target.shape == (128, 128)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_evaluate_suite(self):
        bench = ICCAD13Benchmark(seed=0)
        designs = [(torch.rand(32, 32) > 0.5).float() for _ in range(3)]
        targets = [d.clone() for d in designs]

        results = bench.evaluate_suite(designs, targets, metrics=["EPE", "PVB"])
        assert len(results) == 3
        for r in results:
            assert "EPE" in r
            assert "PVB" in r

    def test_compare_with_public(self):
        bench = ICCAD13Benchmark(seed=0)
        ours = {"EPE": 0.12, "PVB": 0.05}
        published = {"EPE": 0.15, "PVB": 0.04}

        comparison = bench.compare_with_public(ours, published)
        assert "EPE" in comparison
        assert comparison["EPE"]["delta"] == pytest.approx(-0.03)
        assert comparison["EPE"]["ratio"] == pytest.approx(0.8)

    def test_generate_synthetic_suite(self):
        bench = ICCAD13Benchmark(seed=0)
        cases = bench.generate_synthetic_suite(
            sizes=[64, 128],
            complexities=["low", "high"],
        )
        assert len(cases) == 4
        for mask, target, label in cases:
            assert mask.shape[0] == mask.shape[1]
            assert target.shape == mask.shape
            assert isinstance(label, str)


# ---------------------------------------------------------------------------
# TilingResidualRegression
# ---------------------------------------------------------------------------


class TestTilingResidualRegression:
    def test_sweep_tile_count_produces_data(self):
        reg = TilingResidualRegression(seed=42)
        results = reg.sweep_tile_count(
            tile_counts=[2, 4, 8],
            overlap=4,
            n_iterations=2,
        )

        assert len(results) == 3
        for n_tiles, residual in results:
            assert isinstance(n_tiles, int)
            assert residual >= 0.0 or torch.isnan(torch.tensor(residual))

    def test_sweep_batch_size_produces_data(self):
        reg = TilingResidualRegression(seed=42)
        results = reg.sweep_batch_size(
            batch_sizes=[1, 2, 4],
            tile_size=32,
            overlap=4,
            n_iterations=2,
        )

        assert len(results) == 3
        for bs, residual in results:
            assert isinstance(bs, int)
            assert residual >= 0.0

    def test_residual_decreases_with_more_iterations(self):
        torch.manual_seed(42)
        mask = (torch.rand(128, 128) > 0.5).float()

        proc_lo = GPUTileBatchProcessor(seed=42)
        result_lo = proc_lo.process_tiles(
            mask_layout=mask,
            config=TileBatchConfig(
                tile_size=32, overlap=4, n_schwarz_iterations=1,
            ),
        )

        proc_hi = GPUTileBatchProcessor(seed=42)
        result_hi = proc_hi.process_tiles(
            mask_layout=mask,
            config=TileBatchConfig(
                tile_size=32, overlap=4, n_schwarz_iterations=10,
            ),
        )

        assert result_hi.boundary_residual <= result_lo.boundary_residual + 1e-4

    def test_plot_data_returns_both_sweeps(self):
        reg = TilingResidualRegression(seed=42)
        reg.sweep_tile_count(tile_counts=[2, 4], overlap=4, n_iterations=1)
        reg.sweep_batch_size(batch_sizes=[1, 2], tile_size=32, overlap=4, n_iterations=1)

        data = reg.plot_data()
        assert "tile_count" in data
        assert "batch_size" in data
        assert len(data["tile_count"]) == 2
        assert len(data["batch_size"]) == 2
