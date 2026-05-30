"""Tests for multi-process shared-weight inference."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from openlithohub.inference.multiproc import (
    CompiledCache,
    SharedStateDictServer,
    multiproc_predict,
)


# -- tiny model for testing ------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# -- SharedStateDictServer tests -------------------------------------------

class TestSharedStateDictServer:
    def test_shared_memory_matches_original(self) -> None:
        model = _TinyModel()
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        server = SharedStateDictServer(model, prefix="test_match")
        shared_sd = server.state_dict_for_worker()

        for key in original_sd:
            assert torch.allclose(original_sd[key], shared_sd[key]), (
                f"Mismatch for {key}"
            )
        server.cleanup()

    def test_cleanup_removes_shared_memory(self) -> None:
        model = _TinyModel()
        server = SharedStateDictServer(model, prefix="test_cleanup")
        shm_names = [shm.name for shm in server._shms]
        server.cleanup()
        # After cleanup, shared memory blocks should be gone
        import multiprocessing.shared_memory as sm
        for name in shm_names:
            with pytest.raises(FileNotFoundError):
                sm.SharedMemory(name=name)


# -- CompiledCache tests ---------------------------------------------------

class TestCompiledCache:
    def test_cache_hit_on_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = CompiledCache(cache_dir=tmp)
            model1 = _TinyModel()
            compiled1 = cache.get_or_compile(model1)
            meta_path = Path(tmp) / cache._model_hash(model1) / "meta.json"
            assert meta_path.exists(), "Cache entry should exist after first compile"

            # Second call with same weights should be a cache hit
            model2 = _TinyModel()
            model2.load_state_dict(model1.state_dict())
            compiled2 = cache.get_or_compile(model2)
            assert cache._model_hash(model1) == cache._model_hash(model2)

    def test_cache_miss_for_different_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = CompiledCache(cache_dir=tmp)
            model_a = _TinyModel()
            cache.get_or_compile(model_a)

            model_b = _TinyModel()
            nn.init.xavier_uniform_(model_b.fc.weight)
            cache.get_or_compile(model_b)

            # Two different cache entries
            assert cache._model_hash(model_a) != cache._model_hash(model_b)
            entries = list(Path(tmp).iterdir())
            assert len(entries) == 2

    def test_clear_removes_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = CompiledCache(cache_dir=tmp)
            cache.get_or_compile(_TinyModel())
            assert Path(tmp).exists()
            cache.clear()
            assert not Path(tmp).exists()


# -- multiproc_predict tests -----------------------------------------------

class TestMultiprocPredict:
    def test_single_worker_matches_serial(self) -> None:
        """Single-worker output must match serial execution."""
        torch.manual_seed(0)
        model = _TinyModel()
        inputs = [torch.randn(1, 4) for _ in range(4)]

        # Serial reference
        model.eval()
        with torch.no_grad():
            serial_out = [model(t) for t in inputs]

        # Single worker (pass model instance directly)
        mp_out = multiproc_predict(model, inputs, n_workers=1)

        assert len(mp_out) == len(serial_out)
        for s, m in zip(serial_out, mp_out):
            assert torch.allclose(s, m, atol=1e-5), (
                f"Single-worker output diverged: max diff={torch.max(torch.abs(s - m))}"
            )

    def test_peak_memory_constant_with_workers(self) -> None:
        """Peak RSS should not grow linearly with worker count (shared weights)."""
        pytest.importorskip("psutil")
        import psutil

        model = _TinyModel()
        inputs = [torch.randn(1, 4) for _ in range(8)]

        proc = psutil.Process()
        _, rss_2 = multiproc_predict(model, inputs, n_workers=2), proc.memory_info().rss
        _, rss_4 = multiproc_predict(model, inputs, n_workers=4), proc.memory_info().rss

        # RSS growth should be < 50% when doubling workers (shared weights)
        growth = (rss_4 - rss_2) / max(rss_2, 1)
        assert growth < 0.5, (
            f"Memory grew {growth:.0%} when going from 2 to 4 workers "
            f"(expected < 50% with shared weights)"
        )

    def test_output_order_preserved(self) -> None:
        """Outputs must be in the same order as inputs regardless of worker count."""
        torch.manual_seed(42)
        model = _TinyModel()
        inputs = [torch.randn(1, 4) for _ in range(6)]

        out_1 = multiproc_predict(model, inputs, n_workers=1)
        out_3 = multiproc_predict(model, inputs, n_workers=3)

        assert len(out_1) == len(out_3) == 6
        for i, (a, b) in enumerate(zip(out_1, out_3)):
            assert torch.allclose(a, b, atol=1e-5), f"Output {i} mismatch"
