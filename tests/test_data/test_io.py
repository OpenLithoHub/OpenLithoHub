"""Tests for `openlithohub.data.io.load_layout` and the legacy shim."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from openlithohub.cli.optimize_cmd import _load_layout_as_tensor
from openlithohub.data.io import load_layout


def test_load_layout_pt_round_trip(tmp_path: Path) -> None:
    src = torch.zeros(32, 32)
    src[8:24, 8:24] = 1.0
    p = tmp_path / "layout.pt"
    torch.save(src, str(p))
    out = load_layout(p, pixel_nm=1.0)
    assert torch.equal(out, src.float())


def test_load_layout_npy_round_trip(tmp_path: Path) -> None:
    src = np.zeros((16, 16), dtype=np.float32)
    src[4:12, 4:12] = 1.0
    p = tmp_path / "layout.npy"
    np.save(str(p), src)
    out = load_layout(p, pixel_nm=1.0)
    assert torch.equal(out, torch.from_numpy(src))


def test_load_layout_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_layout(tmp_path / "nope.pt", pixel_nm=1.0)


def test_load_layout_pt_rejects_3d(tmp_path: Path) -> None:
    p = tmp_path / "bad.pt"
    torch.save(torch.zeros(4, 4, 4), str(p))
    with pytest.raises(ValueError, match="2-D"):
        load_layout(p, pixel_nm=1.0)


def test_legacy_shim_calls_new_implementation(tmp_path: Path) -> None:
    src = torch.zeros(8, 8)
    src[2:6, 2:6] = 1.0
    p = tmp_path / "shim.pt"
    torch.save(src, str(p))
    via_shim = _load_layout_as_tensor(p, pixel_nm=1.0)
    via_public = load_layout(p, pixel_nm=1.0)
    assert torch.equal(via_shim, via_public)


def test_load_layout_hole_does_not_erase_overlapping_polygon(tmp_path: Path) -> None:
    """Polygon A (with a hole) and polygon B both inserted into a GDS;
    B's solid sits inside A's hole. Earlier per-shape draw order let A's
    hole erase B because holes were drawn into the global canvas after
    each polygon's solid. With Region.merge() applied first, A's hole
    only subtracts from A — B survives where it overlaps A's hole.
    """
    db = pytest.importorskip("klayout.db")

    layout = db.Layout()
    layout.dbu = 0.001  # 1 nm dbu
    cell = layout.create_cell("TOP")
    layer = layout.layer(1, 0)

    # A: 100×100 nm square with a 60×60 hole at (20, 20)-(80, 80).
    a_outer = db.Polygon(db.Box(0, 0, 100, 100))
    a_hole = db.Polygon(db.Box(20, 20, 80, 80))
    a_with_hole = db.Region(a_outer) - db.Region(a_hole)

    # B: 30×30 nm square at (35, 35)-(65, 65), fully inside A's hole.
    b = db.Polygon(db.Box(35, 35, 65, 65))

    for poly in a_with_hole.each():
        cell.shapes(layer).insert(poly)
    cell.shapes(layer).insert(b)

    gds_path = tmp_path / "ab.gds"
    layout.write(str(gds_path))

    raster = load_layout(gds_path, pixel_nm=1.0, layer="1:0")
    # B occupies (35, 35)-(65, 65) — must be foreground despite being
    # inside A's hole. Sample the centre.
    arr = raster.numpy()
    h = arr.shape[0]
    cy, cx = h - 1 - 50, 50  # y-flip from math to image
    assert arr[cy, cx] > 0.5, "B's solid was erased by A's hole — Region.merge regressed"
