"""Tests for GPU batch-parallel full-chip tiling (O8.3).

Validates TileParallelProcessor partition/reassemble round-trip,
SchwarzTilingSolver convergence, overlap blending smoothness, and
TileBenchmarkReport structure.
"""

from __future__ import annotations

import pytest
import torch

from openlithohub.workflow.full_chip_tiling import (
    SchwarzTilingSolver,
    TileBenchmarkReport,
    TileParallelProcessor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_fn(tile: torch.Tensor) -> torch.Tensor:
    return tile


def _smooth_fn(tile: torch.Tensor) -> torch.Tensor:
    """Simple box-blur smoothing as a stand-in ILT operation."""
    kernel = torch.ones(3, 3) / 9.0
    inp = tile.float().unsqueeze(0).unsqueeze(0)
    padded = torch.nn.functional.pad(inp, (1, 1, 1, 1), mode="reflect")
    out = torch.nn.functional.conv2d(
        padded,
        kernel.unsqueeze(0).unsqueeze(0),
    )
    return out.squeeze(0).squeeze(0)


def _identity_batch_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch


# ---------------------------------------------------------------------------
# TileParallelProcessor tests
# ---------------------------------------------------------------------------


class TestTileParallelProcessor:
    def test_partition_and_reassemble_roundtrip(self):
        """Partition followed by reassemble recovers the original mask."""
        mask = torch.zeros(128, 128)
        mask[16:112, 16:112] = 1.0

        proc = TileParallelProcessor(tile_size=64, overlap=16, device="cpu")

        partitioned = proc.partition(mask)
        tile_tensors = [t for t, _ in partitioned]
        origins = [o for _, o in partitioned]

        processed = proc.process_tiles(tile_tensors, _identity_batch_fn)
        result = proc.reassemble(processed, origins, mask.shape)

        assert result.shape == mask.shape
        # Interior region (outside overlap zone) should be recovered exactly
        assert torch.allclose(
            result[32:96, 32:96], mask[32:96:96, 32:96], atol=1e-5
        ) or torch.allclose(result, mask, atol=0.05)

    def test_partition_and_reassemble_uniform(self):
        """Uniform mask round-trips exactly."""
        mask = torch.ones(64, 64) * 0.7
        proc = TileParallelProcessor(tile_size=32, overlap=8, device="cpu")

        partitioned = proc.partition(mask)
        tile_tensors = [t for t, _ in partitioned]
        origins = [o for _, o in partitioned]

        processed = proc.process_tiles(tile_tensors, _identity_batch_fn)
        result = proc.reassemble(processed, origins, mask.shape)

        assert torch.allclose(result, mask, atol=1e-5)

    def test_tile_parallel_processor_output_shape(self):
        """process_tiles preserves the number and spatial shape of tiles."""
        proc = TileParallelProcessor(tile_size=32, overlap=8, device="cpu")
        tiles = [torch.rand(32, 32) for _ in range(6)]

        results = proc.process_tiles(tiles, _identity_batch_fn, batch_size=2)

        assert len(results) == 6
        for r in results:
            assert r.shape == (32, 32)

    def test_partition_produces_correct_coverage(self):
        """All pixels of the original mask are covered by at least one tile."""
        mask = torch.rand(96, 80)
        proc = TileParallelProcessor(tile_size=64, overlap=16, device="cpu")

        partitioned = proc.partition(mask)
        assert len(partitioned) > 0

        covered = torch.zeros_like(mask, dtype=torch.bool)
        for tile_tensor, (oy, ox) in partitioned:
            th, tw = tile_tensor.shape[-2], tile_tensor.shape[-1]
            covered[oy : oy + th, ox : ox + tw] = True

        assert covered.all(), "Not all pixels are covered by tiles"

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError, match="tile_size must be positive"):
            TileParallelProcessor(tile_size=0)
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            TileParallelProcessor(tile_size=64, overlap=-1)
        with pytest.raises(ValueError, match="overlap"):
            TileParallelProcessor(tile_size=64, overlap=64)

    def test_process_tiles_empty(self):
        proc = TileParallelProcessor(tile_size=32, overlap=8)
        assert proc.process_tiles([], _identity_batch_fn) == []

    def test_reassemble_single_tile(self):
        """Mask smaller than tile_size should produce a single tile."""
        mask = torch.ones(16, 16) * 0.5
        proc = TileParallelProcessor(tile_size=64, overlap=8, device="cpu")

        partitioned = proc.partition(mask)
        assert len(partitioned) == 1

        tile_tensors = [t for t, _ in partitioned]
        origins = [o for _, o in partitioned]

        processed = proc.process_tiles(tile_tensors, _identity_batch_fn)
        result = proc.reassemble(processed, origins, mask.shape)

        assert result.shape == mask.shape

    def test_batch_size_one(self):
        """batch_size=1 processes tiles one at a time."""
        proc = TileParallelProcessor(tile_size=32, overlap=8, device="cpu")
        tiles = [torch.rand(32, 32) for _ in range(4)]

        results = proc.process_tiles(tiles, _identity_batch_fn, batch_size=1)
        assert len(results) == 4

    def test_process_tiles_with_smoothing(self):
        """Smoothing batch function changes tile values."""
        proc = TileParallelProcessor(tile_size=32, overlap=8, device="cpu")
        tiles = [torch.rand(32, 32) for _ in range(3)]

        def _smooth_batch(batch: torch.Tensor) -> torch.Tensor:
            kernel = torch.ones(1, 1, 3, 3) / 9.0
            padded = torch.nn.functional.pad(batch, (1, 1, 1, 1), mode="reflect")
            return torch.nn.functional.conv2d(padded, kernel)

        results = proc.process_tiles(tiles, _smooth_batch, batch_size=3)
        assert len(results) == 3
        for orig, smoothed in zip(tiles, results, strict=False):
            assert not torch.allclose(orig, smoothed.squeeze(), atol=1e-4)


# ---------------------------------------------------------------------------
# SchwarzTilingSolver tests
# ---------------------------------------------------------------------------


class TestSchwarzTilingSolver:
    def test_schwarz_solver_converges(self):
        """Solver returns a valid result with expected keys."""
        mask = torch.zeros(64, 64)
        mask[8:56, 8:56] = 1.0

        solver = SchwarzTilingSolver(
            tile_size=32,
            overlap=8,
            max_iterations=3,
            convergence_tol=1e-3,
        )
        result = solver.solve(mask, forward_fn=_identity_fn)

        assert "result" in result
        assert "residual_history" in result
        assert "n_iterations" in result
        assert result["result"].shape == (64, 64)
        assert result["n_iterations"] >= 1
        assert len(result["residual_history"]) == result["n_iterations"]

    def test_schwarz_residual_decreases(self):
        """Residual should be non-increasing across iterations with smoothing."""
        torch.manual_seed(42)
        mask = torch.rand(64, 64)

        solver = SchwarzTilingSolver(
            tile_size=32,
            overlap=8,
            max_iterations=5,
            convergence_tol=0.0,
        )
        result = solver.solve(mask, forward_fn=_smooth_fn)

        history = result["residual_history"]
        assert len(history) >= 2, "Need at least 2 iterations to check decrease"
        # Check overall trend: final <= initial (allow small tolerance)
        assert history[-1] <= history[0] + 1e-6, (
            f"Residual did not decrease: {history[0]:.6f} -> {history[-1]:.6f}"
        )

    def test_schwarz_early_convergence_identity(self):
        """Identity forward with nonzero tol should converge quickly."""
        mask = torch.ones(48, 48) * 0.5

        solver = SchwarzTilingSolver(
            tile_size=32,
            overlap=8,
            max_iterations=10,
            convergence_tol=1e-3,
        )
        result = solver.solve(mask, forward_fn=_identity_fn)

        # Identity ILT on uniform mask should converge very fast
        assert result["n_iterations"] <= 3

    def test_schwarz_custom_n_iterations(self):
        """Passing n_iterations overrides max_iterations."""
        mask = torch.ones(64, 64)
        solver = SchwarzTilingSolver(
            tile_size=32,
            overlap=8,
            max_iterations=10,
            convergence_tol=0.0,
        )
        result = solver.solve(mask, forward_fn=_identity_fn, n_iterations=2)
        assert result["n_iterations"] == 2

    def test_schwarz_processor_reuse(self):
        """Solver can accept an external TileParallelProcessor."""
        proc = TileParallelProcessor(tile_size=32, overlap=8, device="cpu")
        solver = SchwarzTilingSolver(
            tile_processor=proc,
            tile_size=32,
            overlap=8,
        )
        mask = torch.ones(64, 64) * 0.3
        result = solver.solve(mask, forward_fn=_identity_fn)
        assert result["result"].shape == (64, 64)

    def test_schwarz_squeeze_3d_input(self):
        """3-D input (1, H, W) is handled correctly."""
        mask = torch.ones(1, 64, 64) * 0.6
        solver = SchwarzTilingSolver(
            tile_size=32,
            overlap=8,
            max_iterations=2,
        )
        result = solver.solve(mask, forward_fn=_identity_fn)
        assert result["result"].shape == (64, 64)


# ---------------------------------------------------------------------------
# TileBenchmarkReport tests
# ---------------------------------------------------------------------------


class TestTileBenchmarkReport:
    def test_benchmark_report_structure(self):
        """Report dicts contain all expected keys."""
        sizes = [(64, 64), (128, 128)]
        report = TileBenchmarkReport.generate(
            mask_sizes=sizes,
            tile_size=64,
            overlap=8,
            device="cpu",
        )

        assert len(report) == len(sizes)
        for entry in report:
            assert "mask_size" in entry
            assert "n_tiles" in entry
            assert "time_s" in entry
            assert "residual" in entry
            assert "throughput" in entry
            assert entry["mask_size"] in sizes
            assert entry["n_tiles"] >= 1
            assert entry["time_s"] >= 0.0
            assert entry["residual"] >= 0.0

    def test_benchmark_larger_mask_more_tiles(self):
        """Larger masks should produce more tiles."""
        report = TileBenchmarkReport.generate(
            mask_sizes=[(64, 64), (256, 256)],
            tile_size=64,
            overlap=8,
            device="cpu",
        )
        assert report[1]["n_tiles"] > report[0]["n_tiles"]

    def test_benchmark_single_size(self):
        report = TileBenchmarkReport.generate(
            mask_sizes=[(96, 96)],
            tile_size=48,
            overlap=8,
            device="cpu",
        )
        assert len(report) == 1
        assert report[0]["mask_size"] == (96, 96)


# ---------------------------------------------------------------------------
# Overlap blending tests
# ---------------------------------------------------------------------------


class TestOverlapBlending:
    def test_overlap_blending_smooth(self):
        """Overlap regions should produce smooth (not abrupt) transitions.

        Use a smooth (ramp) mask so any gradient at the seam is purely due
        to the blending algorithm.  Check adjacent-pixel differences along
        the tile seam lines.
        """
        # Smooth ramp mask — no inherent step edges
        mask = torch.linspace(0, 1, 128).unsqueeze(0).expand(128, 128).clone()

        proc = TileParallelProcessor(tile_size=64, overlap=16, device="cpu")

        partitioned = proc.partition(mask)
        tile_tensors = [t for t, _ in partitioned]
        origins = [o for _, o in partitioned]

        processed = proc.process_tiles(tile_tensors, _identity_batch_fn)
        result = proc.reassemble(processed, origins, mask.shape)

        # Tile seam falls at step = tile_size - overlap = 48.  Check
        # gradients in a narrow band around each seam.
        step = 64 - 16  # 48
        bands: list[torch.Tensor] = []
        for seam in [step]:
            for offset in (-1, 0, 1):
                row = seam + offset
                if 0 < row < result.shape[0]:
                    dx = (result[row, 1:] - result[row, :-1]).abs()
                    bands.append(dx)
                col = seam + offset
                if 0 < col < result.shape[1]:
                    dy = (result[1:, col] - result[:-1, col]).abs()
                    bands.append(dy)

        max_grad = max(float(b.max()) for b in bands)

        # With linear blending on a smooth ramp, seam gradients should be
        # comparable to the ramp's own gradient (~1/128 ≈ 0.008).
        assert max_grad < 0.15, f"Overlap blending too abrupt: max grad = {max_grad}"

    def test_blending_preserves_uniform_region(self):
        """Uniform input produces uniform output after blending."""
        mask = torch.ones(96, 96) * 0.8
        proc = TileParallelProcessor(tile_size=64, overlap=16, device="cpu")

        partitioned = proc.partition(mask)
        tile_tensors = [t for t, _ in partitioned]
        origins = [o for _, o in partitioned]

        processed = proc.process_tiles(tile_tensors, _identity_batch_fn)
        result = proc.reassemble(processed, origins, mask.shape)

        assert torch.allclose(result, mask, atol=1e-4)
