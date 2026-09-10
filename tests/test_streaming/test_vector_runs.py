from fractions import Fraction

import numpy as np

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.vector_runs import (
    CellInstance,
    ExactVectorRunSource,
    VectorCell,
    rectangle,
)

SHAPE = (24, 32)


def fixture_source():
    child = VectorCell(
        name="CHILD",
        polygons=(rectangle("child_fill", 0, 0, 4, 3),),
    )
    top = VectorCell(
        name="TOP",
        polygons=(
            rectangle("left_edge", 0, 2, 2, 5),
            rectangle("right_edge", 30, 2, 32, 5),
            rectangle("long", 4, 6, 28, 10),
            rectangle("overlap", 6, 8, 12, 13),
            rectangle("donut", 9, 11, 24, 22, holes=((13, 14, 20, 19),)),
        ),
        instances=(
            CellInstance("fill_hole", "CHILD", 15, 15),
            CellInstance("second_child", "CHILD", 2, 17),
        ),
    )
    return ExactVectorRunSource(
        shape=SHAPE, cells={"TOP": top, "CHILD": child}, top="TOP", pixel_size_nm=8.0
    )


def inside_loop(loop, x, y):
    px = Fraction(2 * x + 1, 2)
    py = Fraction(2 * y + 1, 2)
    inside = False
    for (x1, y1), (x2, y2) in zip(loop, loop[1:] + loop[:1], strict=True):
        if (y1 > py) == (y2 > py):
            continue
        xint = Fraction(x1, 1) + (py - y1) * Fraction(x2 - x1, y2 - y1)
        if px < xint:
            inside = not inside
    return inside


def independent_flatten(cells, name, dx=0, dy=0, anc=()):
    if name in anc:
        raise ValueError("cycle")
    cell = cells[name]
    for p in cell.polygons:
        outer = tuple((x + dx, y + dy) for x, y in p.outer)
        holes = tuple(tuple((x + dx, y + dy) for x, y in h) for h in p.holes)
        yield outer, holes
    for inst in cell.instances:
        yield from independent_flatten(
            cells, inst.cell_name, dx + inst.dx, dy + inst.dy, anc + (name,)
        )


def independent_oracle():
    child = VectorCell(name="CHILD", polygons=(rectangle("child_fill", 0, 0, 4, 3),))
    top = VectorCell(
        name="TOP",
        polygons=(
            rectangle("left_edge", 0, 2, 2, 5),
            rectangle("right_edge", 30, 2, 32, 5),
            rectangle("long", 4, 6, 28, 10),
            rectangle("overlap", 6, 8, 12, 13),
            rectangle("donut", 9, 11, 24, 22, holes=((13, 14, 20, 19),)),
        ),
        instances=(
            CellInstance("fill_hole", "CHILD", 15, 15),
            CellInstance("second_child", "CHILD", 2, 17),
        ),
    )
    cells = {"TOP": top, "CHILD": child}
    flat = list(independent_flatten(cells, "TOP"))
    out = np.zeros(SHAPE, dtype=np.uint8)
    for y in range(SHAPE[0]):
        for x in range(SHAPE[1]):
            for outer, holes in flat:
                if inside_loop(outer, x, y) and not any(inside_loop(h, x, y) for h in holes):
                    out[y, x] = 1
                    break
    return out


def test_full_vector_run_source_matches_independent_point_oracle():
    src = fixture_source()
    got = src.read_window(BoundingBox(0, 0, 32, 24)).numpy().astype(np.uint8)
    assert np.array_equal(got, independent_oracle())


def test_hole_is_local_to_parent_and_child_can_fill_it():
    got = fixture_source().read_window(BoundingBox(0, 0, 32, 24)).numpy()
    # Donut hole remains empty away from the child instance.
    assert got[14, 13] == 0
    # Hierarchical child solid fills a subset of the hole.
    assert got[15, 15] == 1
    assert got[17, 18] == 1


def test_left_and_right_physical_edges_are_not_periodically_joined():
    got = fixture_source().read_window(BoundingBox(0, 0, 32, 24)).numpy()
    assert got[2, 0] == 1 and got[2, 31] == 1
    assert got[2, 2] == 0 and got[2, 29] == 0


def test_long_run_keeps_parent_id_across_overlapping_tile_reads():
    src = fixture_source()
    a = list(src.iter_owned_runs_for_bbox(BoundingBox(0, 6, 16, 7), tile_id="a"))
    b = list(src.iter_owned_runs_for_bbox(BoundingBox(12, 6, 32, 7), tile_id="b"))
    ra = next(r for r in a if r.x0 <= 12 < r.x1)
    rb = next(r for r in b if r.x0 <= 12 < r.x1)
    assert ra.run_id == rb.run_id
    assert ra.parent.x0 == 4 and ra.parent.x1 == 28
    assert rb.parent.x0 == 4 and rb.parent.x1 == 28


def test_overlap_preserves_contributor_change_before_binary_union():
    src = fixture_source()
    runs = list(src.iter_owned_runs_for_bbox(BoundingBox(0, 8, 32, 9)))
    # Row 8: long [4,28) and overlap [6,12) -> contributor-set split.
    spans = [(r.parent.x0, r.parent.x1, r.parent.contributor_object_ids) for r in runs]
    assert [s[:2] for s in spans[:3]] == [(4, 6), (6, 12), (12, 28)]
    assert len(spans[1][2]) == 2
    assert len(spans[0][2]) == len(spans[2][2]) == 1


def test_tilewise_core_reconstruction_matches_full_oracle():
    src = fixture_source()
    out = np.zeros(SHAPE, dtype=np.uint8)
    for y0 in range(0, 24, 8):
        for x0 in range(0, 32, 8):
            x1 = min(32, x0 + 8)
            y1 = min(24, y0 + 8)
            win = src.read_window(BoundingBox(x0, y0, x1, y1)).numpy().astype(np.uint8)
            out[y0:y1, x0:x1] = win
    assert np.array_equal(out, independent_oracle())


def test_verify_layout_run_source_protocol_shape():
    src = fixture_source()
    runs = tuple(src.iter_runs_for_bbox((0, 6, 16, 7)))
    assert runs
    assert all(r.y == 6 for r in runs)
    assert sum(r.length for r in runs) == 12


def test_exact_dbu_nm_snaps_foreign_reader_float_artifacts():
    from openlithohub.streaming.vector_runs import exact_dbu_nm

    # The pinned ORFS sky130hd GDS stores a units literal whose nearest
    # binary double reads back as 0.0009999999999999998 (intended: 1 nm).
    assert exact_dbu_nm(0.0009999999999999998) == Fraction(1, 1)
    assert exact_dbu_nm(0.001) == Fraction(1, 1)
    assert exact_dbu_nm(0.0005) == Fraction(1, 2)
    assert exact_dbu_nm(0.002) == Fraction(2, 1)
    assert exact_dbu_nm(0.005) == Fraction(5, 1)
