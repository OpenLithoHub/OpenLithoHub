"""Tests for workflow layer."""

import pytest
import torch

from openlithohub.workflow.contour.manhattan import extract_manhattan_contour
from openlithohub.workflow.tiling import Tile, stitch_tiles, tile_layout

pytest.importorskip("scipy")

from openlithohub.workflow.contour.curvilinear import BSplineCurve, export_oasis_mbw, fit_bspline
from openlithohub.workflow.export import export_gds, export_oasis
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

    def test_warns_on_skipped_short_loop(self):
        """A 3-pixel feature trace produces too few points for a cubic
        periodic spline. Earlier behaviour silently dropped it; small
        SRAFs vanished from the OASIS export with no signal. Verify a
        ``UserWarning`` now fires so the caller can detect missing
        geometry. Suppressing it via ``warn_on_skip=False`` should keep
        the existing silent path available for callers that intentionally
        feed mixed-size loops.
        """
        import pytest as _pytest

        mask = torch.zeros(16, 16)
        mask[2, 2] = 1.0  # single pixel → tiny loop
        with _pytest.warns(UserWarning, match="(skipping|< 5)"):
            curves = fit_bspline(mask, tolerance_nm=1.0)
        assert curves == []

        # Opt-out: no warning, still empty result.
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            curves2 = fit_bspline(mask, tolerance_nm=1.0, warn_on_skip=False)
        assert curves2 == []

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
            # KLayout dedupes coincident vertices when integerising to DB
            # units. At 1 nm pixel pitch and 1 nm DBU, neighbouring samples
            # on a small B-spline can collapse to the same DB coord — the
            # round-trip count is therefore an upper bound, not equality.
            assert 3 <= shape.polygon.num_points() <= samples_per_curve

        bbox = top.bbox()
        assert not bbox.empty()
        # Mask foreground sits in pixels [8, 24) -> nm window roughly [8, 24).
        # KLayout bboxes are reported in DB integer units. With the default
        # dbu=0.001 µm/DBU, 1 DBU = 1 nm, so the values below are in nm.
        nm_per_dbu = layout.dbu * 1000.0
        dbu_per_nm = 1.0 / nm_per_dbu
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


class TestExportGDS:
    def test_curvilinear_gds_round_trip(self, tmp_path):
        # GDSII has no native curve primitive — export_gds samples B-splines
        # to polygons and writes via klayout (which dispatches on extension).
        # Round-trip verifies the file is a real GDS readable by KLayout.
        db = pytest.importorskip("klayout.db")
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        out = tmp_path / "out_curvi.gds"
        export_gds(mask, out, mode="curvilinear", pixel_size_nm=1.0)
        assert out.exists()
        assert out.stat().st_size > 0

        layout = db.Layout()
        layout.read(str(out))
        assert layout.cells() >= 1
        # At least one polygon was emitted on the default (1, 0) layer.
        cell = layout.top_cell()
        layer_idx = layout.layer(1, 0)
        assert sum(1 for _ in cell.shapes(layer_idx).each()) >= 1

    def test_manhattan_gds(self, tmp_path):
        pytest.importorskip("klayout.db")
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        out = tmp_path / "out_man.gds"
        export_gds(mask, out, mode="manhattan", pixel_size_nm=1.0)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_invalid_mode_raises(self, tmp_path):
        mask = torch.zeros(16, 16)
        mask[4:12, 4:12] = 1.0
        with pytest.raises(ValueError, match="mode"):
            export_gds(mask, tmp_path / "out.gds", mode="invalid")

    def test_samples_per_curve_increases_vertex_count(self, tmp_path):
        # Higher samples_per_curve must yield more polygon vertices —
        # this is the user-facing knob for fidelity vs file size.
        db = pytest.importorskip("klayout.db")
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0

        out_lo = tmp_path / "lo.gds"
        out_hi = tmp_path / "hi.gds"
        export_gds(mask, out_lo, mode="curvilinear", pixel_size_nm=1.0, samples_per_curve=16)
        export_gds(mask, out_hi, mode="curvilinear", pixel_size_nm=1.0, samples_per_curve=128)

        def _vertex_count(path):
            layout = db.Layout()
            layout.read(str(path))
            cell = layout.top_cell()
            layer_idx = layout.layer(1, 0)
            n = 0
            for shape in cell.shapes(layer_idx).each():
                if shape.is_polygon():
                    n += shape.polygon.num_points()
            return n

        assert _vertex_count(out_hi) > _vertex_count(out_lo)


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

    def test_layout_smaller_than_tile_emits_one_tile(self):
        # Issue #13: with overlap < layout < tile_size, the sliding window
        # used to step `tile_size - overlap` and emit redundant zero-padded
        # tiles for each step it took before exiting the layout — wasted
        # forward-model work for zero coverage gain.
        layout = torch.ones(2000, 2000)
        tiles = tile_layout(layout, tile_size=2048, overlap=128)
        assert len(tiles) == 1
        # Single tile should still cover full layout
        assert tiles[0].width == 2000
        assert tiles[0].height == 2000

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

    def test_diagonal_touch_junction(self):
        """Two foreground regions meeting at a single corner (X-junction).

        Each region should produce its own closed polygon — the tracer must
        not jump from one region's boundary onto the other at the shared
        vertex, which would leave both polygons unclosed.
        """
        mask = torch.zeros(8, 8)
        mask[2, 2] = 1.0
        mask[3, 3] = 1.0
        contours = extract_manhattan_contour(mask, pixel_size_nm=1.0)
        # Each unit pixel produces one closed 4-vertex square.
        assert len(contours) == 2
        for poly in contours:
            assert len(poly) == 4
            # Closed: first and last edges connect; in a unit square the
            # extents are exactly 1.0 wide and 1.0 tall.
            xs = [v[0] for v in poly]
            ys = [v[1] for v in poly]
            assert max(xs) - min(xs) == 1.0
            assert max(ys) - min(ys) == 1.0


class TestExportMinAreaFilter:
    """``export_oasis_mbw(min_area_nm2=...)`` drops sub-resolution islands.

    Default 0.0 keeps every shape (Hackathon-safe); a positive value drops
    polygons below the threshold and logs the count. Guards the fab-ready
    export path against MRC-rejecting micro-SRAFs without changing
    Hackathon scoring.
    """

    def _two_squares_mask(self) -> torch.Tensor:
        # Big square (16x16 = 256 px area) + tiny isolated square (2x2 = 4 px)
        # Spaced apart so the contour tracer treats them as separate loops.
        mask = torch.zeros(64, 64)
        mask[8:24, 8:24] = 1.0
        mask[40:42, 40:42] = 1.0
        return mask

    def test_default_zero_keeps_all_shapes(self, tmp_path):
        db = pytest.importorskip("klayout.db")
        mask = self._two_squares_mask()
        curves = fit_bspline(mask, tolerance_nm=1.0, warn_on_skip=False)
        # The 2x2 island may be skipped by fit_bspline (n<5 after dedup) —
        # that path is unrelated to the area filter. We need at least one
        # curve plus one we can construct synthetically below.
        assert len(curves) >= 1

        out = tmp_path / "default.oas"
        export_oasis_mbw(curves, str(out), pixel_size_nm=1.0)
        layout = db.Layout()
        layout.read(str(out))
        polys = list(next(layout.each_cell()).shapes(layout.layer(1, 0)).each())
        assert len(polys) == len(curves)

    def test_positive_threshold_drops_small_polygon(self, tmp_path, caplog):
        """Synthesise two B-spline curves with known areas and verify the
        filter drops only the small one.
        """
        db = pytest.importorskip("klayout.db")
        # Big circle: radius 10 nm → area ~314 nm^2
        big = self._make_circle_curve(radius_nm=10.0, center=(20.0, 20.0))
        # Small circle: radius 1 nm → area ~3.14 nm^2
        small = self._make_circle_curve(radius_nm=1.0, center=(50.0, 50.0))

        out = tmp_path / "filtered.oas"
        with caplog.at_level("INFO", logger="openlithohub.workflow.contour.curvilinear"):
            export_oasis_mbw([big, small], str(out), pixel_size_nm=1.0, min_area_nm2=50.0)
        layout = db.Layout()
        layout.read(str(out))
        polys = list(next(layout.each_cell()).shapes(layout.layer(1, 0)).each())
        assert len(polys) == 1, "small SRAF below 50 nm^2 should be filtered"
        assert any("filtered 1 shape" in rec.getMessage() for rec in caplog.records)

    def test_threshold_above_all_drops_everything(self, tmp_path):
        db = pytest.importorskip("klayout.db")
        big = self._make_circle_curve(radius_nm=10.0, center=(20.0, 20.0))
        out = tmp_path / "all_dropped.oas"
        export_oasis_mbw([big], str(out), pixel_size_nm=1.0, min_area_nm2=1e9)
        layout = db.Layout()
        layout.read(str(out))
        polys = list(next(layout.each_cell()).shapes(layout.layer(1, 0)).each())
        assert polys == []

    def test_negative_threshold_raises(self, tmp_path):
        big = self._make_circle_curve(radius_nm=10.0, center=(20.0, 20.0))
        with pytest.raises(ValueError, match="min_area_nm2 must be >= 0"):
            export_oasis_mbw([big], str(tmp_path / "x.oas"), pixel_size_nm=1.0, min_area_nm2=-1.0)

    @staticmethod
    def _make_circle_curve(*, radius_nm: float, center: tuple[float, float]) -> BSplineCurve:
        """Build a closed periodic cubic B-spline that traces a circle.

        Uses scipy.interpolate.splprep on a sampled circle so the output
        round-trips through splev → polygon → shoelace area cleanly.
        """
        from scipy.interpolate import splprep

        n = 32
        thetas = torch.linspace(0, 2 * 3.14159265, n + 1)[:-1]
        xs = (center[0] + radius_nm * torch.cos(thetas)).numpy()
        ys = (center[1] + radius_nm * torch.sin(thetas)).numpy()
        tck, _ = splprep([xs, ys], s=0.0, per=True, k=3)
        ctrl_x = torch.tensor(tck[1][0], dtype=torch.float32)
        ctrl_y = torch.tensor(tck[1][1], dtype=torch.float32)
        ctrl = torch.stack([ctrl_x, ctrl_y], dim=1)
        knots = torch.tensor(tck[0], dtype=torch.float32)
        return BSplineCurve(control_points=ctrl, knots=knots, degree=3)


class TestExportVertexTolerance:
    """``export_oasis_mbw(vertex_tolerance_nm=...)`` simplifies sampled
    polygons via Ramer-Douglas-Peucker.

    Default 0.0 keeps every sampled vertex (bit-exact academic behaviour).
    A positive value cuts vertex count without measurable area change —
    the Mask Shop / multi-beam writer (MBMW) wins back data volume that
    would otherwise stall MDP tools on full-chip OASIS.
    """

    @staticmethod
    def _circle_curve(radius_nm: float, center: tuple[float, float]) -> BSplineCurve:
        from scipy.interpolate import splprep

        n = 32
        thetas = torch.linspace(0, 2 * 3.14159265, n + 1)[:-1]
        xs = (center[0] + radius_nm * torch.cos(thetas)).numpy()
        ys = (center[1] + radius_nm * torch.sin(thetas)).numpy()
        tck, _ = splprep([xs, ys], s=0.0, per=True, k=3)
        ctrl_x = torch.tensor(tck[1][0], dtype=torch.float32)
        ctrl_y = torch.tensor(tck[1][1], dtype=torch.float32)
        ctrl = torch.stack([ctrl_x, ctrl_y], dim=1)
        knots = torch.tensor(tck[0], dtype=torch.float32)
        return BSplineCurve(control_points=ctrl, knots=knots, degree=3)

    @staticmethod
    def _polygon_vertex_count(layout, layer_idx) -> int:
        total = 0
        for cell in layout.each_cell():
            for shape in cell.shapes(layer_idx).each():
                poly = shape.polygon
                if poly is not None:
                    total += poly.num_points()
        return total

    def test_default_zero_keeps_all_vertices(self, tmp_path):
        db = pytest.importorskip("klayout.db")
        curve = self._circle_curve(radius_nm=10.0, center=(50.0, 50.0))
        out = tmp_path / "no_simplify.oas"
        export_oasis_mbw([curve], str(out), samples_per_curve=64, pixel_size_nm=1.0)
        layout = db.Layout()
        layout.read(str(out))
        # KLayout integerises to DB units (default 1 DBU = 1 nm), so a circle
        # 64 samples coarse can collapse coincident neighbours. Vertex count
        # is capped at samples_per_curve and must remain non-degenerate.
        n = self._polygon_vertex_count(layout, layout.layer(1, 0))
        assert 4 <= n <= 64

    def test_positive_tolerance_reduces_vertices(self, tmp_path, caplog):
        """A 0.5 nm tolerance on a smooth circle should drop a meaningful
        fraction of vertices and emit an INFO log line.
        """
        db = pytest.importorskip("klayout.db")
        curve = self._circle_curve(radius_nm=10.0, center=(50.0, 50.0))
        out = tmp_path / "simplified.oas"
        with caplog.at_level("INFO", logger="openlithohub.workflow.contour.curvilinear"):
            export_oasis_mbw(
                [curve],
                str(out),
                samples_per_curve=64,
                pixel_size_nm=1.0,
                vertex_tolerance_nm=0.5,
            )
        layout = db.Layout()
        layout.read(str(out))
        n_simplified = self._polygon_vertex_count(layout, layout.layer(1, 0))
        assert n_simplified < 64, (
            "RDP at 0.5nm should drop at least one vertex on a 10nm-radius circle"
        )
        assert n_simplified >= 4, "polygon must remain non-degenerate"
        assert any("RDP simplified" in rec.getMessage() for rec in caplog.records)

    def test_simplification_preserves_area(self, tmp_path):
        """Shoelace area before/after RDP should match within 10% — the
        whole point of vertex_tolerance_nm is that printability does not
        move much. (At 1 nm DBU integer rounding, RDP at 0.5 nm chord
        tolerance can drop ~6% area on a 10nm-radius circle; the bound
        is set with headroom for that discrete-grid floor.)
        """
        db = pytest.importorskip("klayout.db")
        curve = self._circle_curve(radius_nm=10.0, center=(50.0, 50.0))

        out_full = tmp_path / "full.oas"
        export_oasis_mbw([curve], str(out_full), samples_per_curve=64, pixel_size_nm=1.0)
        out_simp = tmp_path / "simp.oas"
        export_oasis_mbw(
            [curve],
            str(out_simp),
            samples_per_curve=64,
            pixel_size_nm=1.0,
            vertex_tolerance_nm=0.5,
        )

        def _area(path):
            layout = db.Layout()
            layout.read(str(path))
            poly = next(next(layout.each_cell()).shapes(layout.layer(1, 0)).each()).polygon
            # KLayout polygon area is in DBU². With default dbu=0.001 µm/DBU,
            # 1 DBU² = 1 nm², so the integer area is already in nm². Comparing
            # ratios is unit-agnostic; we keep the multiplication for clarity.
            return poly.area() * (layout.dbu * 1000.0) ** 2

        a_full = _area(out_full)
        a_simp = _area(out_simp)
        assert abs(a_simp - a_full) / a_full < 0.10

    def test_negative_tolerance_raises(self, tmp_path):
        curve = self._circle_curve(radius_nm=10.0, center=(50.0, 50.0))
        with pytest.raises(ValueError, match="vertex_tolerance_nm must be >= 0"):
            export_oasis_mbw(
                [curve],
                str(tmp_path / "x.oas"),
                pixel_size_nm=1.0,
                vertex_tolerance_nm=-0.1,
            )
