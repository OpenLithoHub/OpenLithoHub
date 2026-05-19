"""Tests for `openlithohub._utils.optics` — measured source / Zernike pupil I/O."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from openlithohub._utils.optics import (
    load_source_intensity,
    load_zernike_coefficients,
    zernike_phase_map,
)


def _write_tiff(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.float32), mode="F").save(path, format="TIFF")


def test_load_source_intensity_tiff_normalizes_to_one(tmp_path: Path) -> None:
    arr = np.zeros((32, 32), dtype=np.float32)
    arr[12:20, 12:20] = 5.0  # off-axis dipole-ish blob, sum = 320
    p = tmp_path / "source.tif"
    _write_tiff(p, arr)

    out = load_source_intensity(p)
    assert out.shape == (32, 32)
    assert out.dtype == torch.float32
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)


def test_load_source_intensity_resize(tmp_path: Path) -> None:
    arr = np.zeros((64, 64), dtype=np.float32)
    arr[28:36, 28:36] = 1.0
    p = tmp_path / "source.tif"
    _write_tiff(p, arr)

    out = load_source_intensity(p, grid_size=32)
    assert out.shape == (32, 32)
    # Bilinear resize preserves total mass within rounding once we re-normalize.
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)


def test_load_source_rejects_nonsquare_when_grid_size_unset(tmp_path: Path) -> None:
    arr = np.ones((32, 64), dtype=np.float32)
    p = tmp_path / "rect.tif"
    _write_tiff(p, arr)
    with pytest.raises(ValueError, match="square"):
        load_source_intensity(p)


def test_load_source_rejects_all_zero(tmp_path: Path) -> None:
    arr = np.zeros((16, 16), dtype=np.float32)
    p = tmp_path / "zero.tif"
    _write_tiff(p, arr)
    with pytest.raises(ValueError, match="zero/negative"):
        load_source_intensity(p)


def test_load_source_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_source_intensity(tmp_path / "nope.tif")


def test_load_source_no_normalize(tmp_path: Path) -> None:
    arr = np.full((8, 8), 4.0, dtype=np.float32)  # sum = 256
    p = tmp_path / "src.tif"
    _write_tiff(p, arr)
    out = load_source_intensity(p, normalize=False)
    assert torch.isclose(out.sum(), torch.tensor(256.0))


def test_load_source_png(tmp_path: Path) -> None:
    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[6:10, 6:10] = 255
    p = tmp_path / "src.png"
    Image.fromarray(arr, mode="L").save(p)
    out = load_source_intensity(p)
    assert out.shape == (16, 16)
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)


def test_load_zernike_json_flat(tmp_path: Path) -> None:
    p = tmp_path / "z.json"
    p.write_text(json.dumps({"4": 0.05, "11": -0.02}))
    coeffs = load_zernike_coefficients(p)
    assert coeffs == {4: 0.05, 11: -0.02}


def test_load_zernike_json_nested(tmp_path: Path) -> None:
    p = tmp_path / "z.json"
    p.write_text(json.dumps({"meta": "scanner-X", "zernikes": {"4": 0.05}}))
    coeffs = load_zernike_coefficients(p)
    assert coeffs == {4: 0.05}


def test_load_zernike_csv(tmp_path: Path) -> None:
    p = tmp_path / "z.csv"
    p.write_text("noll,coeff,name\n4,0.05,defocus\n11,-0.02,sphere\n")
    coeffs = load_zernike_coefficients(p)
    assert coeffs == {4: 0.05, 11: -0.02}


def test_load_zernike_txt(tmp_path: Path) -> None:
    p = tmp_path / "z.txt"
    p.write_text("# scanner Z-dump\n4 0.05\n11 -0.02   # primary spherical\n")
    coeffs = load_zernike_coefficients(p)
    assert coeffs == {4: 0.05, 11: -0.02}


def test_load_zernike_drops_piston(tmp_path: Path) -> None:
    p = tmp_path / "z.txt"
    p.write_text("1 0.99\n4 0.05\n")
    coeffs = load_zernike_coefficients(p)
    assert 1 not in coeffs
    assert coeffs[4] == 0.05


def test_load_zernike_rejects_unknown_index(tmp_path: Path) -> None:
    p = tmp_path / "z.txt"
    p.write_text("999 0.1\n")
    with pytest.raises(ValueError, match="beyond the supported range"):
        load_zernike_coefficients(p)


def test_load_zernike_unsupported_format(tmp_path: Path) -> None:
    p = tmp_path / "z.bin"
    p.write_text("4 0.05\n")
    with pytest.raises(ValueError, match="Unsupported Zernike format"):
        load_zernike_coefficients(p)


def test_zernike_phase_map_shape_and_outside_pupil() -> None:
    opd = zernike_phase_map({4: 0.05}, grid_size=32)
    assert opd.shape == (32, 32)
    assert opd.dtype == torch.float32
    # Outside the unit pupil the map is exactly zero (corners of the square).
    assert opd[0, 0] == 0.0
    assert opd[-1, -1] == 0.0


def test_zernike_phase_map_zero_when_no_coeffs() -> None:
    opd = zernike_phase_map({}, grid_size=16)
    assert torch.all(opd == 0.0)


def test_zernike_z4_is_defocus_paraboloid() -> None:
    """Z4 (Noll) is normalized defocus: sqrt(3) * (2ρ² - 1) on the unit pupil.

    The center value should be -sqrt(3) * coeff and the unit-circle edge
    value should be +sqrt(3) * coeff (within sampling).
    """
    opd = zernike_phase_map({4: 1.0}, grid_size=65)
    center = opd[32, 32].item()
    # Edge sample at (1, 0) lives at index (32, 64).
    edge = opd[32, 64].item()
    assert math.isclose(center, -math.sqrt(3), abs_tol=1e-5)
    assert math.isclose(edge, math.sqrt(3), abs_tol=1e-5)


def test_zernike_phase_map_grid_size_too_small() -> None:
    with pytest.raises(ValueError, match=">= 2|≥ 2"):
        zernike_phase_map({4: 0.1}, grid_size=1)
