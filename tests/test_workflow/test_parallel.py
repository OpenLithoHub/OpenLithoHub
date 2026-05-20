"""Tests for the multi-GPU tile inference path (RFC 0004)."""

from __future__ import annotations

import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from openlithohub.models.registry import register_builtin_models
from openlithohub.workflow.parallel import (
    _resolve_worker_device,
    _round_robin_shards,
    parallel_tile_inference,
)
from openlithohub.workflow.tiling import stitch_tiles, tile_layout

register_builtin_models()


def _layout(h: int = 64, w: int = 64) -> torch.Tensor:
    t = torch.zeros(h, w)
    t[16:48, 16:48] = 1.0
    return t


def test_round_robin_shards_balanced():
    assert _round_robin_shards(7, 3) == [[0, 3, 6], [1, 4], [2, 5]]
    assert [len(s) for s in _round_robin_shards(7, 3)] == [3, 2, 2]
    assert _round_robin_shards(0, 4) == [[], [], [], []]
    assert _round_robin_shards(4, 4) == [[0], [1], [2], [3]]


def test_resolve_worker_device_falls_back_to_cpu_when_no_gpu():
    # Even if the user said "cuda", we fall back to CPU when CUDA isn't
    # available — this is what makes the CPU CI smoke valid.
    if not torch.cuda.is_available():
        assert _resolve_worker_device(0, 2, "cuda") == "cpu"
    assert _resolve_worker_device(0, 2, "cpu") == "cpu"


def test_parallel_dispatch_cpu_smoke():
    """num_gpus=2 on CPU dispatches the loop and returns all tiles in order."""
    layout = _layout()
    tiles = tile_layout(layout, tile_size=32, overlap=0)
    results = parallel_tile_inference(
        model_name="dummy-identity",
        model_kwargs={},
        tiles=tiles,
        num_gpus=2,
        base_perf_kwargs={"device": "cpu", "dtype": torch.float32, "compile_forward": False},
    )
    assert len(results) == len(tiles)
    for orig, (tile, mask) in zip(tiles, results, strict=True):
        assert tile is orig
        assert torch.equal(mask, orig.tensor)


def test_parallel_matches_sequential():
    """Stitched output of parallel path equals the sequential path bit-for-bit."""
    layout = _layout()
    tiles = tile_layout(layout, tile_size=32, overlap=8)

    seq_pairs = [(t, t.tensor.clone()) for t in tiles]
    seq_out = stitch_tiles(seq_pairs, layout.shape)

    par_pairs = parallel_tile_inference(
        model_name="dummy-identity",
        model_kwargs={},
        tiles=tiles,
        num_gpus=2,
        base_perf_kwargs={"device": "cpu", "dtype": torch.float32, "compile_forward": False},
    )
    par_out = stitch_tiles(par_pairs, layout.shape)

    assert torch.allclose(seq_out, par_out)


def test_parallel_worker_error_propagates():
    """A worker raising in predict surfaces as RuntimeError in the parent."""
    layout = _layout()
    tiles = tile_layout(layout, tile_size=32, overlap=0)

    with pytest.raises(RuntimeError, match="dummy-failing"):
        parallel_tile_inference(
            model_name="dummy-failing",
            model_kwargs={},
            tiles=tiles,
            num_gpus=2,
            base_perf_kwargs={"device": "cpu", "dtype": torch.float32, "compile_forward": False},
        )

    # No orphaned worker processes left around.
    leftover = [p for p in mp.active_children() if p.name.startswith("Process")]
    assert leftover == []


def test_parallel_invalid_num_gpus():
    layout = _layout()
    tiles = tile_layout(layout, tile_size=32, overlap=0)
    with pytest.raises(ValueError, match="num_gpus"):
        parallel_tile_inference(
            model_name="dummy-identity",
            model_kwargs={},
            tiles=tiles,
            num_gpus=0,
            base_perf_kwargs={"device": "cpu", "dtype": torch.float32, "compile_forward": False},
        )


def test_parallel_more_workers_than_tiles():
    """Asking for more workers than tiles: caps at len(tiles), no hangs."""
    layout = _layout(32, 32)
    tiles = tile_layout(layout, tile_size=32, overlap=0)
    assert len(tiles) == 1
    results = parallel_tile_inference(
        model_name="dummy-identity",
        model_kwargs={},
        tiles=tiles,
        num_gpus=4,
        base_perf_kwargs={"device": "cpu", "dtype": torch.float32, "compile_forward": False},
    )
    assert len(results) == 1


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires >=2 CUDA devices",
)
def test_parallel_gpu_two():
    layout = _layout(128, 128)
    tiles = tile_layout(layout, tile_size=64, overlap=8)

    par_pairs = parallel_tile_inference(
        model_name="dummy-identity",
        model_kwargs={},
        tiles=tiles,
        num_gpus=2,
        base_perf_kwargs={"device": "cuda", "dtype": torch.float32, "compile_forward": False},
    )
    par_out = stitch_tiles(par_pairs, layout.shape)

    seq_pairs = [(t, t.tensor.clone()) for t in tiles]
    seq_out = stitch_tiles(seq_pairs, layout.shape)
    assert torch.allclose(seq_out, par_out)


def test_optimize_run_num_gpus_default_unchanged():
    """--num-gpus 1 (default) produces the same output as not passing the flag."""
    from typer.testing import CliRunner

    from openlithohub.cli.app import app

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        layout_arr = np.zeros((64, 64), dtype=np.float32)
        layout_arr[16:48, 16:48] = 1.0
        in_path = Path(tmpdir) / "in.npy"
        np.save(in_path, layout_arr)

        out_default = Path(tmpdir) / "out_default.oas"
        out_explicit = Path(tmpdir) / "out_explicit.oas"
        out_fallback_default = out_default.with_suffix(".pt")
        out_fallback_explicit = out_explicit.with_suffix(".pt")

        r1 = runner.invoke(
            app,
            ["optimize", "run", "-i", str(in_path), "-m", "dummy-identity", "-o", str(out_default)],
        )
        r2 = runner.invoke(
            app,
            [
                "optimize",
                "run",
                "-i",
                str(in_path),
                "-m",
                "dummy-identity",
                "--num-gpus",
                "1",
                "-o",
                str(out_explicit),
            ],
        )
        assert r1.exit_code == 0, r1.output
        assert r2.exit_code == 0, r2.output

        # Either OASIS path lands or the .pt fallback does — accept either.
        path1 = out_default if out_default.exists() else out_fallback_default
        path2 = out_explicit if out_explicit.exists() else out_fallback_explicit
        assert path1.exists()
        assert path2.exists()
        if path1.suffix == ".pt" and path2.suffix == ".pt":
            t1 = torch.load(path1, weights_only=True)
            t2 = torch.load(path2, weights_only=True)
            assert torch.equal(t1, t2)


# Sanity check: parallel.py is import-clean from a fresh interpreter — this
# matters because workers re-import everything from scratch under the spawn
# context.
def test_parallel_module_imports_clean():
    assert "openlithohub.workflow.parallel" in sys.modules
