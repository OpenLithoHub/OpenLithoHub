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
