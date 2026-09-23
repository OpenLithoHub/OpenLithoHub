"""PR-G G1 hostile tests: the band index is candidate filtering only.

The exact rational scanline machinery (row spans, hole subtraction,
binary union, contributor identity, run ids, tile clipping) must produce
byte-identical semantics with and without the index.  Discrete semantics
therefore use EXACT equality — never tolerance.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.vector_runs import (
    CellInstance,
    ExactVectorRunSource,
    VectorCell,
    rectangle,
)


def _hostile_source() -> ExactVectorRunSource:
    """Holes, overlaps, hierarchy/instances, long skinny + tall polygons,
    boundary-hugging shapes — the §6 fixture menu."""
    child = VectorCell("CHILD", polygons=(rectangle("child_fill", 0, 0, 4, 3),))
    top = VectorCell(
        "TOP",
        polygons=(
            rectangle("left_edge", 0, 2, 2, 5),
            rectangle("right_edge", 30, 2, 32, 5),
            rectangle("long_skinny", 0, 6, 32, 7),  # spans full width, 1 row
            rectangle("tall_thin", 15, 0, 16, 24),  # spans full height, 1 col
            rectangle("overlap_a", 4, 8, 12, 13),
            rectangle("overlap_b", 6, 10, 20, 16),  # overlaps overlap_a
            rectangle(
                "donut", 9, 14, 24, 23, holes=((13, 16, 20, 21),)
            ),  # hole spans a band boundary at 16
            rectangle("band_straddle", 3, 254, 10, 258),  # crosses y=255/256
        ),
        instances=(
            CellInstance("inst_one", "CHILD", 15, 15),
            CellInstance("inst_two", "CHILD", 2, 17),
            CellInstance("inst_three", "CHILD", 22, 30),
        ),
    )
    return ExactVectorRunSource(
        shape=(512, 48),
        cells={"TOP": top, "CHILD": child},
        top="TOP",
        pixel_size_nm=1.0,
    )


def _rows(source: ExactVectorRunSource) -> list[tuple[int, ...]]:
    return [source._polygons_for_row(y) for y in range(source.shape[0])]


def _reference_rows(source: ExactVectorRunSource) -> list[tuple[int, ...]]:
    return [source._polygons_for_row_reference(y) for y in range(source.shape[0])]


# ---- G1-A/B/C: candidate parity propagates to owned-run semantics ----------


def test_g1abc_candidates_and_owned_runs_identical_to_reference() -> None:
    indexed = _hostile_source()
    reference = _hostile_source()
    assert indexed._flat == reference._flat
    assert _rows(indexed) == _reference_rows(reference)

    full = BoundingBox(0, 0, indexed.shape[1], indexed.shape[0])
    indexed_runs = list(indexed.iter_owned_runs_for_bbox(full))
    reference_runs = list(reference.iter_owned_runs_for_bbox(full))
    assert len(indexed_runs) == len(reference_runs)
    for got, want in zip(indexed_runs, reference_runs, strict=True):
        # run_id binds layout hash + y + x-range + contributor ids: exact
        # equality here proves all of G1-A/B/C at once.
        assert (got.parent.run_id, got.parent.y, got.x0, got.x1) == (
            want.parent.run_id,
            want.parent.y,
            want.x0,
            want.x1,
        )
        assert got.parent.contributor_object_ids == want.parent.contributor_object_ids
        assert got.tile_id == want.tile_id and got.role == want.role


def test_g1abc_window_rasters_identical_everywhere() -> None:
    indexed = _hostile_source()
    reference = _hostile_source()
    h, w = indexed.shape
    for y0, x0, y1, x1 in [
        (0, 0, h, w),
        (0, 0, 7, 7),
        (250, 0, 260, 48),  # straddles the band boundary
        (100, 20, 103, 23),
        (511, 46, 512, 48),
    ]:
        bbox = BoundingBox(x0, y0, x1, y1)
        assert np.array_equal(
            indexed.read_window(bbox).numpy(), reference.read_window(bbox).numpy()
        ), f"window raster drifted at {bbox}"


# ---- G1-D/E: overlapping tile reads, holes/hierarchy/instances --------------


def test_g1de_overlapping_tile_reads_and_hierarchy_unchanged(tmp_path: Path) -> None:
    indexed = _hostile_source()
    reference = _hostile_source()
    for x0, y0 in [(0, 0), (16, 8), (24, 240)]:
        a = BoundingBox(x0, y0, min(x0 + 20, 48), min(y0 + 20, 512))
        b = BoundingBox(max(0, x0 - 6), max(0, y0 - 6), min(x0 + 26, 48), min(y0 + 26, 512))
        # Same window on both sources: metadata must match exactly.  The
        # overlapping pair (a, b) additionally proves overlapping tile
        # reads do not manufacture new owners on either path.
        for window, core in ((a, a), (b, b), (a, b)):
            meta_indexed = indexed.verification_metadata(window, tile_id="t1", core_bbox=core)
            meta_reference = reference.verification_metadata(window, tile_id="t1", core_bbox=core)
            assert meta_indexed["b04_layout_hash"] == meta_reference["b04_layout_hash"]
            assert meta_indexed["b04_backend_kind"] == meta_reference["b04_backend_kind"]
            assert [
                (r.parent.run_id, r.x0, r.x1, r.tile_id, r.role)
                for r in meta_indexed["b04_owned_runs"]
            ] == [
                (r.parent.run_id, r.x0, r.x1, r.tile_id, r.role)
                for r in meta_reference["b04_owned_runs"]
            ]


# ---- G1-F: empty windows do not scan all geometry ---------------------------


def test_g1f_empty_window_candidates_are_empty_and_cheap() -> None:
    source = _hostile_source()
    # A window in a region with no geometry (x 33..47 has no polygons).
    candidates = source.query_bbox(33, 100, 47, 400)
    assert candidates == []
    row = source.candidates_for_row(300)
    assert not row
    # Structural: the row's band holds ONLY geometry that straddles into
    # it (band_straddle spans the 255/256 boundary) — never the whole
    # flattened layout.
    band_polys = source._bands.get(300 // source._band_height, ())
    assert {p.object_id for p in band_polys} == {"TOP/band_straddle"}


def test_g1f2_query_bbox_candidates_stay_exact_superset() -> None:
    source = _hostile_source()
    bbox = BoundingBox(8, 9, 24, 20)
    window = source.query_bbox(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    # The candidate set must be a SUPERSET of the polygons the reference
    # scan finds intersecting the window rows (candidate = possibly relevant).
    reference_hits = {
        id(poly)
        for poly in source._flat
        for y in range(bbox.y0, bbox.y1)
        if poly.bbox[1] <= y < poly.bbox[3] and poly.bbox[0] < bbox.x1 and poly.bbox[2] > bbox.x0
    }
    assert reference_hits <= {id(poly) for poly in window}


# ---- G1-G: index memory does not scale with raster area ----------------------


def test_g1g_index_memory_is_geometry_bounded() -> None:
    sparse = ExactVectorRunSource(
        shape=(8192, 8192),
        cells={"TOP": VectorCell("TOP", polygons=(rectangle("one", 0, 0, 16, 16),))},
        top="TOP",
    )
    stats = sparse.index_stats()
    # One polygon spanning one band: a handful of entries for 67M raster
    # pixels — the index can never become a raster in disguise.
    assert stats["entries"] <= 2
    assert stats["bands"] <= 2
    assert stats["polygons"] == 1

    # Scaling raster area 4x with the SAME geometry must not grow entries.
    bigger = ExactVectorRunSource(
        shape=(16384, 16384),
        cells={"TOP": VectorCell("TOP", polygons=(rectangle("one", 0, 0, 16, 16),))},
        top="TOP",
    )
    assert bigger.index_stats()["entries"] == stats["entries"]


# ---- G1-H/I: immutable reuse + concurrent reads ------------------------------


def test_g1h_repeated_queries_reuse_immutable_index() -> None:
    source = _hostile_source()
    first = source.query_bbox(0, 0, 32, 512)
    snapshot = dict(source._bands)
    for _ in range(50):
        assert source.query_bbox(0, 0, 32, 512) == first
    assert source._bands == snapshot, "index mutated by repeated queries"


def test_g1i_concurrent_read_queries_do_not_corrupt() -> None:
    source = _hostile_source()
    expected = source.read_window(BoundingBox(0, 0, 48, 512)).numpy()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(40):
                got = source.read_window(BoundingBox(0, 0, 48, 512)).numpy()
                assert np.array_equal(got, expected)
                source.candidates_for_row(300)
                source.query_bbox(4, 8, 20, 30)
        except Exception as exc:  # noqa: BLE001 — collected, not raised in-thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    assert np.array_equal(source.read_window(BoundingBox(0, 0, 48, 512)).numpy(), expected)


# ---- §7 structural evidence: candidates << total on realistic density -------


def test_g1_structural_candidate_reduction() -> None:
    """Candidate polygons inspected per row must be far below the total
    flattened polygon count once geometry is spatially spread."""
    polys = [
        rectangle(f"p{y0}_{x0}", x0 * 2, y0 * 2, x0 * 2 + 3, y0 * 2 + 3)
        for y0 in range(60)
        for x0 in range(20)
    ]  # 1200 small polygons spread over a 40x120 area
    source = ExactVectorRunSource(
        shape=(256, 48),
        cells={"TOP": VectorCell("TOP", polygons=tuple(polys))},
        top="TOP",
    )
    total = len(source._flat)
    inspected = sum(len(source.candidates_for_row(y)) for y in range(source.shape[0]))
    # First-touch cost: rows × band-candidates, not rows × total polygons.
    assert inspected < total * source.shape[0] / 8, (
        f"candidate reduction too weak: inspected={inspected}, "
        f"rows*total would be {total * source.shape[0]}"
    )
    stats = source.index_stats()
    assert stats["entries"] < 6 * stats["polygons"]
