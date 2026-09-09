"""Tile source / sink tests (RFC 0008, prompt §2/§3)."""

import numpy as np
import pytest
import torch

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.sinks import MemmapTileSink, MetricOnlyTileSink, TensorTileSink
from openlithohub.streaming.sources import (
    MemmapTensorTileSource,
    SpatialLayoutIndex,
    TensorTileSource,
    VectorLayoutTileSource,
)


class TestTensorTileSource:
    def test_window_reads(self):
        t = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        src = TensorTileSource(t, pixel_size_nm=2.0)
        assert src.shape == (10, 10)
        assert src.pixel_size_nm == 2.0
        win = src.read_window(BoundingBox(2, 3, 5, 7))
        assert torch.equal(win, t[3:7, 2:5])

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="2-D"):
            TensorTileSource(torch.zeros(2, 2, 2))


class TestMemmapSource:
    def test_reads_from_npy_file(self, tmp_path):
        full = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        path = tmp_path / "layout.npy"
        np.save(path, full)
        src = MemmapTensorTileSource(path)
        assert src.shape == (64, 64)
        win = src.read_window(BoundingBox(16, 16, 48, 48))
        assert torch.equal(win, torch.from_numpy(full[16:48, 16:48]))

    def test_raw_memmap_with_explicit_meta(self, tmp_path):
        full = np.zeros((32, 32), dtype=np.float16)
        raw = tmp_path / "layout.raw"
        np.memmap(raw, dtype=np.float16, mode="w+", shape=(32, 32))[...] = full
        src = MemmapTensorTileSource(raw, dtype=np.float16, shape=(32, 32))
        assert src.read_window(BoundingBox(0, 0, 4, 4)).shape == (4, 4)

    def test_npy_header_mismatch_rejected(self, tmp_path):
        # a raw (non-.npy) file without dtype/shape metadata cannot be opened
        raw = tmp_path / "layout.raw"
        raw.write_bytes(b"\x00" * (8 * 8 * 4))
        with pytest.raises(Exception):  # noqa: B017 — np.memmap raises TypeError/ValueError
            MemmapTensorTileSource(raw)


class TestVectorSource:
    def _index(self):
        boxes = [(10, 10, 20, 20), (30, 30, 40, 40), (0, 50, 100, 60)]
        return SpatialLayoutIndex(boxes, (64, 100), bucket=32)

    def test_window_query_only_intersecting(self):
        src = VectorLayoutTileSource(self._index(), (64, 100))
        win = src.read_window(BoundingBox(25, 25, 30, 30))
        assert win.shape == (5, 5)
        assert float(win.max()) == 0.0  # nothing intersects

    def test_window_partial_overlap(self):
        src = VectorLayoutTileSource(self._index(), (64, 100))
        win = src.read_window(BoundingBox(15, 15, 25, 25))
        # object (10,10)-(20,20) intersects the window in global
        # (15,15)-(20,20) → window-local (0,0)-(5,5)
        assert float(win[0:5, 0:5].max()) == 1.0
        assert float(win[5:, 5:].max()) == 0.0

    def test_empty_index_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            SpatialLayoutIndex([], (64, 64))


class TestTensorTileSink:
    def test_identity_assembly(self):
        full = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        sink = TensorTileSink((8, 8))
        sink.write_core("t0", BoundingBox(0, 0, 4, 4), full[:4, :4])
        sink.write_core("t1", BoundingBox(4, 0, 8, 4), full[:4, 4:])
        sink.write_core("t2", BoundingBox(0, 4, 4, 8), full[4:, :4])
        sink.write_core("t3", BoundingBox(4, 4, 8, 8), full[4:, 4:])
        assert torch.equal(sink.finalize(), full)

    def test_duplicate_write_rejected(self):
        sink = TensorTileSink((8, 8))
        box = BoundingBox(0, 0, 4, 4)
        sink.write_core("t0", box, torch.zeros(4, 4))
        with pytest.raises(ValueError, match="duplicate"):
            sink.write_core("t0", box, torch.zeros(4, 4))

    def test_shape_mismatch_rejected(self):
        sink = TensorTileSink((8, 8))
        with pytest.raises(ValueError, match="core tensor shape"):
            sink.write_core("t0", BoundingBox(0, 0, 4, 4), torch.zeros(2, 2))


class TestMemmapSink:
    def test_out_of_core_assembly(self, tmp_path):
        full = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        sink = MemmapTileSink((8, 8), tmp_path / "out.npy")
        for i in range(2):
            for j in range(2):
                y, x = i * 4, j * 4
                sink.write_core(
                    f"t{i}{j}",
                    BoundingBox(x, y, x + 4, y + 4),
                    full[y : y + 4, x : x + 4],
                )
        out = sink.finalize()
        assert np.array_equal(np.asarray(out), full.numpy())


class TestMetricOnlySink:
    def test_never_materialises_raster(self):
        sink = MetricOnlyTileSink((64, 64))
        assert not hasattr(sink, "output")  # no raster attribute at all
        for i in range(4):
            sink.write_core(
                f"t{i}",
                BoundingBox(0, i * 16, 64, (i + 1) * 16),
                torch.ones(16, 64),
                metadata={"worst_epe_nm": float(i), "tile": i},
            )
        result = sink.finalize()
        assert result["n_tiles"] == 4
        assert sink.covered_pixels == 64 * 64
        assert result["metrics"]["max_worst_epe_nm"] == 3.0
        assert not hasattr(sink, "output") or not isinstance(
            getattr(sink, "output", None), torch.Tensor
        )
