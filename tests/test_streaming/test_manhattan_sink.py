"""Unit tests for the PR-C sink additions.

* ``MemmapTileSink(npy=True)`` writes a self-describing ``.npy`` artifact.
* ``StreamingManhattanTileSink`` converts trusted binary cores into
  Manhattan rectangles exactly once (Gate C2) with y-flipped, pixel-sized
  DBU coordinates, and rejects anything it cannot represent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

db = pytest.importorskip("klayout.db")  # noqa: E402 — gate before project imports

from openlithohub.streaming.geometry import BoundingBox  # noqa: E402
from openlithohub.streaming.sinks import (  # noqa: E402
    MemmapTileSink,
    StreamingManhattanTileSink,
    _dbu_for_pixel,
)


def _rasterize_boxes(
    ly: db.Layout, cell: db.Cell, layer_index: int, shape: tuple[int, int]
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    h = shape[0]
    for sh in cell.shapes(layer_index).each():
        b = sh.box
        x0, x1 = int(b.left), int(b.right)
        y0, y1 = h - int(b.top), h - int(b.bottom)
        out[y0:y1, x0:x1] = 1.0
    return out


class TestMemmapTileSinkNpy:
    def test_npy_artifact_is_self_describing(self, tmp_path: Path) -> None:
        path = tmp_path / "artifact.npy"
        shape = (16, 24)
        sink = MemmapTileSink(shape, path, npy=True)
        core = torch.ones((8, 12))
        sink.write_core("a", BoundingBox(0, 0, 12, 8), core)
        sink.write_core("b", BoundingBox(12, 8, 24, 16), core)
        sink.finalize()
        loaded = np.load(path, mmap_mode="r")
        assert loaded.shape == shape
        assert loaded.dtype == np.float32
        arr = np.asarray(loaded)
        assert arr[:8, :12].all() and arr[8:16, 12:].all()
        assert arr[8:16, :12].sum() == 0

    def test_raw_mode_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "raw.bin"
        sink = MemmapTileSink((4, 4), path)
        sink.write_core("a", BoundingBox(0, 0, 4, 4), torch.ones((4, 4)))
        sink.finalize()
        # A raw raster has no npy magic.
        assert path.read_bytes()[:6] != b"\x93NUMPY"


class TestDbuForPixel:
    @pytest.mark.parametrize(
        "pixel_nm", [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 50.0]
    )
    def test_pixel_edges_lay_on_integer_dbu(self, pixel_nm: float) -> None:
        ppd, dbu_um = _dbu_for_pixel(pixel_nm)
        assert ppd >= 1
        # pixel edge index * pixel size in dbu must be an integer
        edge_dbu = pixel_nm / (dbu_um * 1000.0)
        assert abs(edge_dbu - round(edge_dbu)) < 1e-9
        assert 1e-5 - 1e-12 <= dbu_um <= 1e-2 + 1e-12

    def test_invalid_pixel_rejected(self) -> None:
        with pytest.raises(ValueError):
            _dbu_for_pixel(0.0)


class TestStreamingManhattanTileSink:
    def test_cores_round_trip_to_identical_occupancy(self, tmp_path: Path) -> None:
        shape = (20, 22)
        path = tmp_path / "out.oas"
        sink = StreamingManhattanTileSink(shape, path, pixel_size_nm=8.0)
        expected = np.zeros(shape, dtype=np.float32)
        # Two tiles with patterns that exercise vertical run merging and
        # per-row fragmentation.
        core_a = np.array(
            [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        core_b = np.array([[1, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        sink.write_core("a", BoundingBox(3, 5, 7, 9), torch.from_numpy(core_a))
        sink.write_core("b", BoundingBox(10, 2, 12, 6), torch.from_numpy(core_b))
        expected[5:9, 3:7] = core_a
        expected[2:6, 10:12] = core_b
        sink.finalize()

        ly = db.Layout()
        ly.read(str(path))
        top = ly.top_cells()[0]
        back = _rasterize_boxes(ly, top, ly.layer(1, 0), shape)
        assert np.array_equal(back, expected)

    def test_duplicate_core_write_rejected(self, tmp_path: Path) -> None:
        sink = StreamingManhattanTileSink((8, 8), tmp_path / "o.oas")
        core = torch.ones((2, 2))
        sink.write_core("t", BoundingBox(0, 0, 2, 2), core)
        with pytest.raises(ValueError, match="duplicate"):
            sink.write_core("t", BoundingBox(0, 0, 2, 2), core)

    def test_non_binary_core_rejected(self, tmp_path: Path) -> None:
        sink = StreamingManhattanTileSink((8, 8), tmp_path / "o.oas")
        with pytest.raises(ValueError, match="binarised"):
            sink.write_core("t", BoundingBox(0, 0, 2, 2), torch.tensor([[0.5, 0.0], [0.0, 0.0]]))

    def test_shape_mismatch_rejected(self, tmp_path: Path) -> None:
        sink = StreamingManhattanTileSink((8, 8), tmp_path / "o.oas")
        with pytest.raises(ValueError, match="bbox"):
            sink.write_core("t", BoundingBox(0, 0, 3, 3), torch.ones((2, 2)))

    def test_certified_commit_semantics(self, tmp_path: Path) -> None:
        path = tmp_path / "o.oas"
        sink = StreamingManhattanTileSink((6, 6), path, pixel_size_nm=8.0)
        sink.record_certified_core("one", BoundingBox(0, 0, 2, 2), exact_fill=1.0, metadata={})
        sink.record_certified_core("zero", BoundingBox(2, 0, 4, 2), exact_fill=0.0, metadata={})
        from openlithohub.streaming.sinks import CertifiedCommitNotRepresentableError

        with pytest.raises(CertifiedCommitNotRepresentableError):
            sink.record_certified_core(
                "none", BoundingBox(4, 0, 6, 2), exact_fill=None, metadata={}
            )
        sink.finalize()
        ly = db.Layout()
        ly.read(str(path))
        back = _rasterize_boxes(ly, ly.top_cells()[0], ly.layer(1, 0), (6, 6))
        expected = np.zeros((6, 6), dtype=np.float32)
        expected[0:2, 0:2] = 1.0
        assert np.array_equal(back, expected)

    def test_write_does_not_materialise_full_raster(self, tmp_path: Path) -> None:
        """Structural check: the sink object holds geometry, not a raster."""
        sink = StreamingManhattanTileSink((4096, 4096), tmp_path / "big.oas")
        assert not hasattr(sink, "output")
        assert not hasattr(sink, "_map")
        sink.write_core("t", BoundingBox(0, 0, 8, 8), torch.ones((8, 8)))
        assert sink.n_rectangles == 1
