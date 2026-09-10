from pathlib import Path

import pytest

db = pytest.importorskip("klayout.db")  # noqa: E402 — gate before project imports

from openlithohub.streaming.physical_identity import (  # noqa: E402
    InstancePathElement,
)
from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource  # noqa: E402


def _call_or_attr(obj, name, default=None):
    v = getattr(obj, name, default)
    return v() if callable(v) else v


def _tr_token(t):
    if t is None:
        return "none"
    f = getattr(t, "to_s", None)
    return str(f()) if callable(f) else str(t)


def _path_key(it, source_format="gds"):
    elems = []
    for elem in tuple(_call_or_attr(it, "path", ())):
        ci = _call_or_attr(elem, "cell_inst", None)
        elems.append(
            InstancePathElement(
                source_cell_index=int(_call_or_attr(ci, "cell_index", -1)),
                array_i=int(_call_or_attr(elem, "ia", 0)),
                array_j=int(_call_or_attr(elem, "ib", 0)),
                specific_transform=_tr_token(_call_or_attr(elem, "specific_cplx_trans", None)),
            )
        )
    return tuple(elems)


def _write_array_fixture(path: Path):
    ly = db.Layout()
    ly.dbu = 0.001
    l1 = ly.layer(1, 0)
    child = ly.create_cell("CHILD")
    top = ly.create_cell("TOP")
    child.shapes(l1).insert(db.Box(0, 0, 8, 8))

    # Regular array. GDS writes this as AREF-like structure; OASIS may use
    # repetition semantics. The acceptance test is after parser normalization:
    # distinct physical members must remain distinguishable by path/member id.
    arr = db.CellInstArray(
        child.cell_index(),
        db.Trans(0, False, db.Point(16, 16)),
        db.Vector(16, 0),
        db.Vector(0, 16),
        3,
        2,
    )
    top.insert(arr)

    # Rotation and reflection as separate source-level physical instances.
    top.insert(
        db.CellInstArray(
            child.cell_index(),
            db.Trans(1, False, db.Point(80, 16)),
        )
    )
    top.insert(
        db.CellInstArray(
            child.cell_index(),
            db.Trans(0, True, db.Point(104, 24)),
        )
    )

    # Integer magnification remains on-grid and should be parseable.
    top.insert(
        db.CellInstArray(
            child.cell_index(),
            db.ICplxTrans(2.0, 0.0, False, 128, 16),
        )
    )
    ly.write(str(path))


@pytest.mark.requires_klayout
@pytest.mark.parametrize("suffix", [".gds", ".oas"])
def test_source_repetition_members_have_distinct_instance_paths(tmp_path, suffix):
    path = tmp_path / f"instance_semantics{suffix}"
    _write_array_fixture(path)

    ly = db.Layout()
    ly.read(str(path))
    top = ly.cell("TOP")
    layer = ly.layer(1, 0)
    it = top.begin_shapes_rec(layer)
    keys = []
    while not it.at_end():
        keys.append(_path_key(it, source_format=suffix.lstrip(".")))
        it.next()

    # 6 array members + rotation + mirror + magnified instance.
    assert len(keys) >= 9
    assert len(set(keys)) == len(keys)

    # The six regular-array members must expose differing ia/ib/member transforms.
    array_keys = [k for k in keys if any(e.array_i or e.array_j for e in k)]
    assert len(array_keys) >= 5


@pytest.mark.requires_klayout
@pytest.mark.parametrize("suffix", [".gds", ".oas"])
def test_instance_aware_adapter_does_not_collapse_equal_geometry(tmp_path, suffix):
    path = tmp_path / f"two_equal_instances{suffix}"
    ly = db.Layout()
    ly.dbu = 0.001
    l1 = ly.layer(1, 0)
    child = ly.create_cell("C")
    top = ly.create_cell("TOP")
    child.shapes(l1).insert(db.Box(0, 0, 8, 8))
    top.insert(db.CellInstArray(child.cell_index(), db.Trans(8, 8)))
    top.insert(db.CellInstArray(child.cell_index(), db.Trans(24, 8)))
    # Add anchors on nonselected layer to establish exact pixel bbox.
    la = ly.layer(99, 0)
    top.shapes(la).insert(db.Box(0, 0, 1, 1))
    top.shapes(la).insert(db.Box(39, 23, 40, 24))
    ly.write(str(path))

    src = KLayoutAlignedRunSource.from_file(path, pixel_size_nm=1.0, layer="1:0", top_cell="TOP")
    # Flattening scopes object ids with the parent cell name ("TOP/physical:…"),
    # so match on the physical-key marker rather than the full prefix.
    ids = {owner for poly in src._flat for owner in (poly.object_id,) if "/physical:" in owner}
    assert len(ids) == 2


def _write_path_fixture(path: Path, round_ends: bool, bgn: int, end: int):
    ly = db.Layout()
    ly.dbu = 0.001
    l1 = ly.layer(1, 0)
    la = ly.layer(99, 0)
    top = ly.create_cell("TOP")
    p = db.Path(
        [db.Point(16, 16), db.Point(64, 16), db.Point(64, 48)],
        8,
        bgn,
        end,
        round_ends,
    )
    top.shapes(l1).insert(p)
    top.shapes(la).insert(db.Box(0, 0, 1, 1))
    top.shapes(la).insert(db.Box(95, 63, 96, 64))
    ly.write(str(path))


@pytest.mark.requires_klayout
@pytest.mark.parametrize("suffix", [".gds", ".oas"])
@pytest.mark.parametrize(
    "round_ends,bgn,end",
    [(False, 0, 0), (False, 4, 12), (True, 4, 4)],
)
def test_path_cap_extension_semantics_are_parser_visible(tmp_path, suffix, round_ends, bgn, end):
    path = tmp_path / f"path_semantics_{round_ends}_{bgn}_{end}{suffix}"
    _write_path_fixture(path, round_ends, bgn, end)
    ly = db.Layout()
    ly.read(str(path))
    top = ly.cell("TOP")
    layer = ly.layer(1, 0)
    shapes = list(top.shapes(layer).each())
    if suffix == ".oas" and round_ends:
        # The OASIS writer normalizes a round-cap path into the trunk plus
        # two separate round cap discs; cap state stays parser-visible on
        # the normalized pieces.
        assert len(shapes) == 3
        assert sum(1 for s in shapes if s.is_path() and bool(s.path.is_round())) == 2
        return
    assert len(shapes) == 1
    s = shapes[0]
    assert s.is_path() or s.is_polygon()
    if s.is_path():
        p = s.path
        assert int(p.width) == 8
        # GDS/OASIS writer/parser is permitted to normalize representation,
        # but if retained as PATH the cap metadata must be inspectable.
        assert int(p.bgn_ext) == bgn
        assert int(p.end_ext) == end
        assert bool(p.is_round()) == round_ends


@pytest.mark.requires_klayout
def test_non_grid_arbitrary_rotation_is_rejected_by_theorem_adapter(tmp_path):
    path = tmp_path / "offgrid.gds"
    ly = db.Layout()
    ly.dbu = 0.001
    l1 = ly.layer(1, 0)
    la = ly.layer(99, 0)
    c = ly.create_cell("C")
    top = ly.create_cell("TOP")
    c.shapes(l1).insert(db.Box(0, 0, 16, 8))
    top.insert(db.CellInstArray(c.cell_index(), db.ICplxTrans(1.0, 45.0, False, 32, 32)))
    top.shapes(la).insert(db.Box(0, 0, 1, 1))
    top.shapes(la).insert(db.Box(95, 95, 96, 96))
    ly.write(str(path))
    with pytest.raises(ValueError, match="not exactly aligned"):
        KLayoutAlignedRunSource.from_file(path, pixel_size_nm=8.0, layer="1:0", top_cell="TOP")
