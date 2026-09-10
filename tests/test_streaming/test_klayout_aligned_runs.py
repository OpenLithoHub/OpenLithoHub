from pathlib import Path

import numpy as np
import pytest

db = pytest.importorskip("klayout.db")  # noqa: E402 — gate before project imports

from openlithohub.streaming.geometry import BoundingBox  # noqa: E402
from openlithohub.streaming.vector_runs import (  # noqa: E402
    CellInstance,
    ExactVectorRunSource,
    KLayoutAlignedRunSource,
    VectorCell,
    rectangle,
)

H, W = 24, 32
STEP = 8


def _physical_box(x0, y0, x1, y1):
    # Desired raster y-down edge coordinates -> KLayout y-up DBU.
    return db.Box(
        x0 * STEP,
        (H - y1) * STEP,
        x1 * STEP,
        (H - y0) * STEP,
    )


def expected_source():
    child = VectorCell(
        "CHILD",
        polygons=(rectangle("child_fill", 0, 0, 4, 3),),
    )
    top = VectorCell(
        "TOP",
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
        shape=(H, W), cells={"TOP": top, "CHILD": child}, top="TOP", pixel_size_nm=8.0
    )


def write_fixture(path: Path):
    ly = db.Layout()
    ly.dbu = 0.001  # 1 nm / DBU
    l1 = ly.layer(1, 0)
    anchor = ly.layer(99, 0)
    child = ly.create_cell("CHILD")
    top = ly.create_cell("TOP")

    # Local child box = 4x3 pixels in physical coordinates.
    child.shapes(l1).insert(db.Box(0, 0, 4 * STEP, 3 * STEP))

    for box in [
        (0, 2, 2, 5),
        (30, 2, 32, 5),
        (4, 6, 28, 10),
        (6, 8, 12, 13),
    ]:
        top.shapes(l1).insert(_physical_box(*box))

    # One polygon with a hole.
    donut = db.Polygon(_physical_box(9, 11, 24, 22))
    donut.insert_hole(_physical_box(13, 14, 20, 19))
    top.shapes(l1).insert(donut)

    # Desired y-down instances [15,18) and [17,20).
    top.insert(
        db.CellInstArray(
            child.cell_index(),
            db.Trans(15 * STEP, (H - 18) * STEP),
        )
    )
    top.insert(
        db.CellInstArray(
            child.cell_index(),
            db.Trans(2 * STEP, (H - 20) * STEP),
        )
    )

    # Force the top-cell bbox to exactly 32x24 pixels without changing layer 1.
    top.shapes(anchor).insert(db.Box(0, 0, 1, 1))
    top.shapes(anchor).insert(db.Box(W * STEP - 1, H * STEP - 1, W * STEP, H * STEP))

    ly.write(str(path))


@pytest.mark.requires_klayout
@pytest.mark.parametrize("suffix", [".gds", ".oas"])
def test_real_klayout_parser_matches_exact_vector_oracle(tmp_path, suffix):
    path = tmp_path / f"b04_inc21_fixture{suffix}"
    write_fixture(path)

    got = KLayoutAlignedRunSource.from_file(path, pixel_size_nm=8.0, layer="1:0")
    want = expected_source()

    assert got.shape == want.shape == (24, 32)

    full = BoundingBox(0, 0, 32, 24)
    assert np.array_equal(
        got.read_window(full).numpy(),
        want.read_window(full).numpy(),
    )

    # Tilewise equivalence without materializing a full-chip raster in source.
    for y0 in range(0, H, 8):
        for x0 in range(0, W, 8):
            box = BoundingBox(x0, y0, min(W, x0 + 8), min(H, y0 + 8))
            assert np.array_equal(
                got.read_window(box).numpy(),
                want.read_window(box).numpy(),
            )

    # Long physical parent retains identity across overlapping reads.
    a = list(got.iter_owned_runs_for_bbox(BoundingBox(0, 6, 16, 7), tile_id="a"))
    b = list(got.iter_owned_runs_for_bbox(BoundingBox(12, 6, 32, 7), tile_id="b"))
    ra = next(r for r in a if r.x0 <= 12 < r.x1)
    rb = next(r for r in b if r.x0 <= 12 < r.x1)
    assert ra.run_id == rb.run_id
    assert (ra.parent.x0, ra.parent.x1) == (4, 28)
    assert (rb.parent.x0, rb.parent.x1) == (4, 28)

    # Hole remains a hole except where a hierarchical child fills it.
    raster = got.read_window(full).numpy()
    assert raster[14, 13] == 0
    assert raster[15, 15] == 1
    assert raster[17, 18] == 1

    # Finite edges remain physical edges, not periodic neighbours.
    assert raster[2, 0] == 1 and raster[2, 31] == 1
    assert raster[2, 2] == 0 and raster[2, 29] == 0
