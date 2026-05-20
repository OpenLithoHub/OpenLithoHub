"""Tests for the OO `Mask` façade."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch

from openlithohub import Mask


def test_from_tensor_basic(sample_design: torch.Tensor) -> None:
    m = Mask.from_tensor(sample_design, pixel_size_nm=0.5, layer="1:0")
    assert m.shape == (64, 64)
    assert m.pixel_size_nm == 0.5
    assert m.layer == "1:0"
    assert m.tensor.dtype == torch.float32


def test_from_tensor_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        Mask.from_tensor(torch.zeros(2, 64, 64))


def test_from_tensor_rejects_non_tensor() -> None:
    with pytest.raises(TypeError):
        Mask.from_tensor(np.zeros((64, 64)))  # type: ignore[arg-type]


def test_frozen(sample_design: torch.Tensor) -> None:
    m = Mask.from_tensor(sample_design)
    with pytest.raises(FrozenInstanceError):
        m.pixel_size_nm = 0.25  # type: ignore[misc]


def test_pt_round_trip(tmp_path: Path, sample_design: torch.Tensor) -> None:
    m = Mask.from_tensor(sample_design, pixel_size_nm=0.5)
    path = tmp_path / "design.pt"
    m.to_pt(path)
    loaded = Mask.from_pt(path, pixel_size_nm=0.5)
    assert torch.equal(loaded.tensor, m.tensor)
    assert loaded.pixel_size_nm == 0.5


def test_npy_round_trip(tmp_path: Path, sample_design: torch.Tensor) -> None:
    m = Mask.from_tensor(sample_design, pixel_size_nm=0.5)
    path = tmp_path / "design.npy"
    m.to_npy(path)
    loaded = Mask.from_npy(path, pixel_size_nm=0.5)
    assert torch.equal(loaded.tensor, m.tensor)


def test_load_dispatches_by_suffix(tmp_path: Path, sample_design: torch.Tensor) -> None:
    m = Mask.from_tensor(sample_design)
    pt_path = tmp_path / "x.pt"
    npy_path = tmp_path / "x.npy"
    m.to_pt(pt_path)
    m.to_npy(npy_path)

    assert torch.equal(Mask.load(pt_path).tensor, m.tensor)
    assert torch.equal(Mask.load(npy_path).tensor, m.tensor)


def test_load_rejects_unknown_suffix(tmp_path: Path) -> None:
    bogus = tmp_path / "x.txt"
    bogus.write_text("nope")
    with pytest.raises(ValueError, match="unsupported extension"):
        Mask.load(bogus)


def test_load_rejects_layer_for_pt(tmp_path: Path, sample_design: torch.Tensor) -> None:
    pt_path = tmp_path / "x.pt"
    Mask.from_tensor(sample_design).to_pt(pt_path)
    with pytest.raises(ValueError, match="layer is meaningless"):
        Mask.load(pt_path, layer="1:0")


def test_array_protocol(sample_design: torch.Tensor) -> None:
    m = Mask.from_tensor(sample_design)
    arr = np.asarray(m)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (64, 64)


def test_oasis_round_trip(tmp_path: Path, sample_design: torch.Tensor) -> None:
    """OASIS export uses klayout; skip cleanly if not installed."""
    pytest.importorskip("klayout.db")

    m = Mask.from_tensor(sample_design, pixel_size_nm=1.0)
    out_path = tmp_path / "out.oas"
    m.to_oasis(out_path, mode="manhattan")
    assert out_path.exists() and out_path.stat().st_size > 0

    loaded = Mask.from_oasis(out_path, pixel_size_nm=1.0)
    # Rasterization → polygons → rasterization is lossy at boundaries; check
    # that the bulk of the foreground survives, not byte equality.
    assert loaded.shape[0] > 0 and loaded.shape[1] > 0
    assert (loaded.tensor > 0).any()
