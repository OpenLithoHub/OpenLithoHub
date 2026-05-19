"""Tests for workflow layer."""

import pytest
import torch

from openlithohub.workflow.contour.manhattan import extract_manhattan_contour
from openlithohub.workflow.tiling import Tile, stitch_tiles, tile_layout

pytest.importorskip("scipy")

from openlithohub.workflow.contour.curvilinear import BSplineCurve, export_oasis_mbw, fit_bspline
from openlithohub.workflow.export import export_oasis
from openlithohub.workflow.parsing import parse_layout


class TestParseLayout:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_layout("/nonexistent/path.oas")

    def test_unsupported_format_raises(self):
        import tempfile

        with (
            tempfile.NamedTemporaryFile(suffix=".txt") as f,
            pytest.raises(ValueError, match="Unsupported"),
        ):
            parse_layout(f.name)

    def test_requires_klayout(self, tmp_path, monkeypatch):
        import sys

        fake_file = tmp_path / "test.oas"
        fake_file.write_bytes(b"fake")
        monkeypatch.setitem(sys.modules, "klayout", None)
        monkeypatch.setitem(sys.modules, "klayout.db", None)
        with pytest.raises(ImportError, match="klayout"):
            parse_layout(str(fake_file))


class TestBSplineFitting:
    def test_fit_square_mask(self):
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        curves = fit_bspline(mask, tolerance_nm=1.0)
        assert len(curves) >= 1
        for c in curves:
            assert isinstance(c, BSplineCurve)
            assert c.control_points.ndim == 2
            assert c.control_points.shape[1] == 2
            assert c.degree == 3

    def test_fit_ordered_points(self):
        t = torch.linspace(0, 2 * 3.14159, 50)
        points = torch.stack([torch.cos(t) * 10 + 16, torch.sin(t) * 10 + 16], dim=1)
        curves = fit_bspline(points, tolerance_nm=0.5)
        assert len(curves) >= 1

    def test_empty_mask_returns_empty(self):
        mask = torch.zeros(16, 16)
        curves = fit_bspline(mask, tolerance_nm=1.0)
        assert curves == []

    def test_export_creates_file(self, tmp_path):
        pytest.importorskip("klayout.db")
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        curves = fit_bspline(mask, tolerance_nm=1.0)
        assert len(curves) > 0

        out = tmp_path / "output.oas"
        export_oasis_mbw(curves, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_oasis_round_trip(self, tmp_path):
        """Re-read the exported OASIS via klayout and verify polygon count + bbox.

        Guards against silent export corruption: a writer that produces a
        well-formed but semantically empty file would pass the size check above
        but fail here.
        """
        db = pytest.importorskip("klayout.db")

        pixel_size_nm = 1.0
        samples_per_curve = 64
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        curves = fit_bspline(mask, tolerance_nm=1.0)
        assert len(curves) > 0

        out = tmp_path / "round_trip.oas"
        export_oasis_mbw(
            curves,
            str(out),
            samples_per_curve=samples_per_curve,
            pixel_size_nm=pixel_size_nm,
            layer=1,
            datatype=0,
            cell_name="TOP",
        )

        layout = db.Layout()
        layout.read(str(out))

        cells = list(layout.each_cell())
        assert len(cells) == 1
        top = cells[0]
        assert top.name == "TOP"

        layer_idx = layout.layer(1, 0)
        polygons = list(top.shapes(layer_idx).each())
        assert len(polygons) == len(curves)
        for shape in polygons:
            assert shape.is_polygon()
            assert shape.polygon.num_points() == samples_per_curve

        bbox = top.bbox()
        assert not bbox.empty()
        # Mask foreground sits in pixels [8, 24) -> nm window roughly [8, 24).
        # Allow generous slack: B-spline smoothing pulls in slightly, and
        # klayout bbox is reported in dbu units (= pixel_size_nm / 1000 nm).
        dbu_per_nm = 1.0 / layout.dbu
        assert bbox.left >= 0
        assert bbox.bottom >= 0
        assert bbox.right <= int(round(32 * dbu_per_nm))
        assert bbox.top <= int(round(32 * dbu_per_nm))
        # Bounding box should at least span half of the foreground region.
        min_span_dbu = int(round(8 * dbu_per_nm))
        assert (bbox.right - bbox.left) >= min_span_dbu
        assert (bbox.top - bbox.bottom) >= min_span_dbu


class TestExportOASIS:
    def test_curvilinear_export(self, tmp_path):
        pytest.importorskip("klayout.db")
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        out = tmp_path / "out_curvi.oas"
        export_oasis(mask, out, mode="curvilinear", pixel_size_nm=1.0)
        assert out.exists()

    def test_invalid_mode_raises(self, tmp_path):
        mask = torch.zeros(16, 16)
        mask[4:12, 4:12] = 1.0
        with pytest.raises(ValueError, match="mode"):
            export_oasis(mask, tmp_path / "out.oas", mode="invalid")


class TestTileLayout:
    def test_basic_tiling(self):
        layout = torch.ones(128, 128)
        tiles = tile_layout(layout, tile_size=64, overlap=0)
        assert len(tiles) == 4  # 2x2 grid
        assert all(t.tensor.shape == (64, 64) for t in tiles)

    def test_tile_origins(self):
        layout = torch.ones(128, 128)
        tiles = tile_layout(layout, tile_size=64, overlap=0)
        origins = {(t.origin_x, t.origin_y) for t in tiles}
        assert origins == {(0, 0), (64, 0), (0, 64), (64, 64)}

    def test_with_overlap(self):
        layout = torch.ones(128, 128)
        tiles = tile_layout(layout, tile_size=64, overlap=16)
        # step = 64 - 16 = 48. Positions: 0, 48, 96.  3x3 = 9 tiles.
        assert len(tiles) == 9

    def test_boundary_padding(self):
        layout = torch.ones(100, 100)
        tiles = tile_layout(layout, tile_size=64, overlap=0)
        # Positions: 0, 64. 2x2 = 4 tiles.
        assert len(tiles) == 4
        # Boundary tiles are anchored to the layout edge so the full tile is
        # filled with real layout (no zero-pad artefacts at the chip edge).
        last = tiles[-1]
        assert last.tensor.shape == (64, 64)
        assert last.width == 64
        assert last.height == 64
        # The boundary tile's origin is pulled back to fit inside the layout.
        assert last.origin_x == 100 - 64
        assert last.origin_y == 100 - 64

    def test_boundary_zero_pads_when_layout_smaller_than_tile(self):
        # When the layout is smaller than tile_size in some axis, anchoring
        # is impossible and we must fall back to zero padding.
        layout = torch.ones(48, 48)
        tiles = tile_layout(layout, tile_size=64, overlap=0)
        assert len(tiles) == 1
        t = tiles[0]
        assert t.tensor.shape == (64, 64)
        assert t.width == 48
        assert t.height == 48
        assert t.origin_x == 0
        assert t.origin_y == 0

    def test_tile_data_correctness(self):
        layout = torch.zeros(128, 128)
        layout[0:64, 0:64] = 1.0
        tiles = tile_layout(layout, tile_size=64, overlap=0)
        # First tile (origin 0,0) should be all 1s
        first = [t for t in tiles if t.origin_x == 0 and t.origin_y == 0][0]
        assert first.tensor.sum() == 64 * 64

    def test_invalid_overlap(self):
        layout = torch.ones(64, 64)
        with pytest.raises(ValueError):
            tile_layout(layout, tile_size=64, overlap=64)

    def test_invalid_tile_size(self):
        layout = torch.ones(64, 64)
        with pytest.raises(ValueError):
            tile_layout(layout, tile_size=0)

    def test_tile_is_dataclass(self):
        layout = torch.ones(64, 64)
        tiles = tile_layout(layout, tile_size=64, overlap=0)
        assert len(tiles) == 1
        t = tiles[0]
        assert isinstance(t, Tile)
        assert t.overlap == 0

    def test_no_duplicate_origins_when_step_underflows_layout(self):
        # 120x120 layout with tile=64 step=56 puts the last sliding-window
        # position past the edge; anchoring snaps it back onto the previous
        # tile's origin. Without dedup this emits 9 tiles for 4 unique
        # origins, doubling forward-model work.
        layout = torch.ones(120, 120)
        tiles = tile_layout(layout, tile_size=64, overlap=8)
        origins = [(t.origin_x, t.origin_y) for t in tiles]
        assert len(origins) == len(set(origins))
        assert set(origins) == {(0, 0), (56, 0), (0, 56), (56, 56)}


class TestStitchTiles:
    def test_basic_stitch_no_overlap(self):
        layout = torch.ones(128, 128)
        tiles = tile_layout(layout, tile_size=64, overlap=0)
        pairs = [(t, t.tensor) for t in tiles]
        result = stitch_tiles(pairs, (128, 128))
        assert result.shape == (128, 128)
        assert torch.allclose(result, torch.ones(128, 128), atol=1e-5)

    def test_stitch_with_overlap(self):
        layout = torch.ones(64, 64) * 0.7
        tiles = tile_layout(layout, tile_size=32, overlap=8)
        pairs = [(t, t.tensor) for t in tiles]
        result = stitch_tiles(pairs, (64, 64))
        assert result.shape == (64, 64)
        assert (result > 0.0).all()

    def test_stitch_preserves_pattern(self):
        layout = torch.zeros(64, 64)
        layout[16:48, 16:48] = 1.0
        tiles = tile_layout(layout, tile_size=64, overlap=0)
        pairs = [(t, t.tensor) for t in tiles]
        result = stitch_tiles(pairs, (64, 64))
        assert torch.allclose(result, layout, atol=1e-5)


class TestManhattanContour:
    def test_square_contour(self):
        mask = torch.zeros(16, 16)
        mask[4:12, 4:12] = 1.0
        contours = extract_manhattan_contour(mask, pixel_size_nm=1.0)
        assert len(contours) == 1
        # A square should have exactly 4 vertices after simplification
        assert len(contours[0]) == 4

    def test_rectangle_vertices(self):
        mask = torch.zeros(16, 16)
        mask[4:12, 4:12] = 1.0
        contours = extract_manhattan_contour(mask, pixel_size_nm=1.0)
        vertices = contours[0]
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        # Boundary of [4:12, 4:12] is at pixel corners: x in {4, 12}, y in {4, 12}
        assert min(xs) == 4.0
        assert max(xs) == 12.0
        assert min(ys) == 4.0
        assert max(ys) == 12.0

    def test_two_components(self):
        mask = torch.zeros(32, 32)
        mask[2:8, 2:8] = 1.0
        mask[20:26, 20:26] = 1.0
        contours = extract_manhattan_contour(mask, pixel_size_nm=1.0)
        assert len(contours) == 2

    def test_l_shape(self):
        mask = torch.zeros(16, 16)
        mask[2:10, 2:5] = 1.0  # vertical bar
        mask[7:10, 2:10] = 1.0  # horizontal bar
        contours = extract_manhattan_contour(mask, pixel_size_nm=1.0)
        assert len(contours) == 1
        # L-shape has 6 vertices
        assert len(contours[0]) == 6

    def test_pixel_size_scaling(self):
        mask = torch.zeros(8, 8)
        mask[2:6, 2:6] = 1.0
        contours = extract_manhattan_contour(mask, pixel_size_nm=5.0)
        vertices = contours[0]
        xs = [v[0] for v in vertices]
        # Pixel coords [2,6] * 5nm = [10, 30]
        assert min(xs) == 10.0
        assert max(xs) == 30.0

    def test_empty_mask(self):
        mask = torch.zeros(16, 16)
        contours = extract_manhattan_contour(mask, pixel_size_nm=1.0)
        assert contours == []

    def test_full_mask(self):
        mask = torch.ones(8, 8)
        contours = extract_manhattan_contour(mask, pixel_size_nm=1.0)
        assert len(contours) == 1
        # Full mask boundary is the image boundary: 4 corners
        assert len(contours[0]) == 4
