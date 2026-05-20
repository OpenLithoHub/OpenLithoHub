"""Tests for process-node-aware tile halo sizing (RFC 0005)."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from openlithohub.cli.app import app
from openlithohub.models.base import LithographyModel
from openlithohub.models.registry import register_builtin_models, registry
from openlithohub.workflow.halo import (
    DEFAULT_HALO_PX,
    compute_halo_px,
    describe_halo,
)
from openlithohub.workflow.process_node import PROCESS_NODES, get_node

register_builtin_models()


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# --- compute_halo_px ---------------------------------------------------------


def test_compute_halo_px_default_when_no_node_or_model():
    assert compute_halo_px(None, None, pixel_nm=1.0, tile_size=2048) == DEFAULT_HALO_PX


def test_compute_halo_px_oir_dominates():
    # 3nm-euv OIR=250 nm, pixel_nm=1.0 -> 250 px, RF=0 -> 250 rounded up to 256
    halo = compute_halo_px(get_node("3nm-euv"), None, pixel_nm=1.0, tile_size=2048)
    assert halo == 256


def test_compute_halo_px_rf_dominates():
    class RfHeavyModel(LithographyModel):
        NAME = "_test_rf_heavy"
        RECEPTIVE_FIELD_PX = 200

        def predict(self, design, **kwargs):  # pragma: no cover — never called
            raise NotImplementedError

    halo = compute_halo_px(None, RfHeavyModel(), pixel_nm=1.0, tile_size=2048)
    assert halo == 200  # 200 is already a multiple of 8


def test_compute_halo_px_max_of_oir_and_rf():
    class Rf300Model(LithographyModel):
        NAME = "_test_rf_300"
        RECEPTIVE_FIELD_PX = 300

        def predict(self, design, **kwargs):  # pragma: no cover
            raise NotImplementedError

    # 3nm-euv OIR=250 px, RF=300 px -> max=300, already multiple of 8? 300/8=37.5 -> 304
    halo = compute_halo_px(get_node("3nm-euv"), Rf300Model(), pixel_nm=1.0, tile_size=2048)
    assert halo == 304


def test_compute_halo_px_pixel_nm_scales_oir():
    # 28nm OIR=1500 nm, pixel_nm=2.0 -> 750 px -> rounded to 752
    halo = compute_halo_px(get_node("28nm"), None, pixel_nm=2.0, tile_size=2048)
    assert halo == 752


def test_compute_halo_px_clamp_to_tile_size():
    # 28nm OIR=1500 nm at 1 nm/px = 1500, but tile_size=128 -> clamp to 127
    halo = compute_halo_px(get_node("28nm"), None, pixel_nm=1.0, tile_size=128)
    assert halo == 127
    assert halo < 128  # tile_layout requires overlap < tile_size


def test_compute_halo_px_rejects_bad_inputs():
    with pytest.raises(ValueError, match="pixel_nm"):
        compute_halo_px(None, None, pixel_nm=0.0, tile_size=2048)
    with pytest.raises(ValueError, match="tile_size"):
        compute_halo_px(None, None, pixel_nm=1.0, tile_size=1)


def test_describe_halo_strings_are_informative():
    s = describe_halo(256, get_node("3nm-euv"), None, 1.0)
    assert "3nm-euv" in s
    assert "OIR=250" in s
    assert "256 px" in s


# --- ProcessNodeConfig OIR coverage -----------------------------------------


def test_all_nodes_have_optical_radius():
    for node in PROCESS_NODES.values():
        assert node.optical_radius_nm > 0, f"{node.name}: optical_radius_nm not positive"


def test_euv_oir_smaller_than_duv_oir():
    # Physics: shorter wavelength = smaller kernel.
    assert get_node("3nm-euv").optical_radius_nm < get_node("28nm").optical_radius_nm


# --- LithographyModel receptive field ---------------------------------------


def test_all_registered_models_expose_receptive_field():
    for name in registry.list_models():
        cls = registry._models[name]
        rf = cls.RECEPTIVE_FIELD_PX
        assert isinstance(rf, int) and rf >= 0, (
            f"{name}: RECEPTIVE_FIELD_PX must be a non-negative int, got {rf!r}"
        )


# --- CLI integration --------------------------------------------------------


runner = CliRunner()


def _make_input(tmpdir: str) -> Path:
    arr = np.zeros((64, 64), dtype=np.float32)
    arr[16:48, 16:48] = 1.0
    p = Path(tmpdir) / "in.npy"
    np.save(p, arr)
    return p


def test_cli_halo_auto_default_runs_and_prints_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _make_input(tmp)
        out = Path(tmp) / "out.oas"
        r = runner.invoke(
            app,
            [
                "optimize",
                "run",
                "-i",
                str(in_path),
                "-m",
                "dummy-identity",
                "-n",
                "3nm-euv",
                "-o",
                str(out),
            ],
        )
        assert r.exit_code == 0, r.output
        out_clean = _strip_ansi(r.output)
        assert "Halo:" in out_clean
        assert "auto from" in out_clean
        assert "3nm-euv" in out_clean


def test_cli_halo_explicit_int_overrides_auto():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _make_input(tmp)
        out = Path(tmp) / "out.oas"
        r = runner.invoke(
            app,
            [
                "optimize",
                "run",
                "-i",
                str(in_path),
                "-m",
                "dummy-identity",
                "--halo",
                "32",
                "-o",
                str(out),
            ],
        )
        assert r.exit_code == 0, r.output
        out_clean = _strip_ansi(r.output)
        assert "Halo: 32 px" in out_clean
        assert "explicit --halo" in out_clean


def test_cli_overlap_legacy_still_works():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _make_input(tmp)
        out = Path(tmp) / "out.oas"
        r = runner.invoke(
            app,
            [
                "optimize",
                "run",
                "-i",
                str(in_path),
                "-m",
                "dummy-identity",
                "--overlap",
                "32",
                "-o",
                str(out),
            ],
        )
        assert r.exit_code == 0, r.output
        out_clean = _strip_ansi(r.output)
        assert "Halo: 32 px" in out_clean
        assert "from --overlap" in out_clean


def test_cli_halo_and_overlap_conflict():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _make_input(tmp)
        out = Path(tmp) / "out.oas"
        r = runner.invoke(
            app,
            [
                "optimize",
                "run",
                "-i",
                str(in_path),
                "-m",
                "dummy-identity",
                "--halo",
                "32",
                "--overlap",
                "16",
                "-o",
                str(out),
            ],
        )
        assert r.exit_code != 0
        assert "mutually exclusive" in _strip_ansi(r.output)


def test_cli_halo_invalid_string():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _make_input(tmp)
        out = Path(tmp) / "out.oas"
        r = runner.invoke(
            app,
            [
                "optimize",
                "run",
                "-i",
                str(in_path),
                "-m",
                "dummy-identity",
                "--halo",
                "banana",
                "-o",
                str(out),
            ],
        )
        assert r.exit_code != 0


def test_cli_halo_negative_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _make_input(tmp)
        out = Path(tmp) / "out.oas"
        r = runner.invoke(
            app,
            [
                "optimize",
                "run",
                "-i",
                str(in_path),
                "-m",
                "dummy-identity",
                "--halo",
                "-1",
                "-o",
                str(out),
            ],
        )
        assert r.exit_code != 0
