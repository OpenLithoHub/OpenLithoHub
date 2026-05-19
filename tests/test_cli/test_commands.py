"""Tests for the CLI module."""

import json
import re
import tempfile
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from openlithohub.cli.app import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "openlithohub" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output or "openlithohub" in result.output


def test_eval_run_help():
    result = runner.invoke(app, ["eval", "run", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--model" in output
    assert "--dataset" in output


def test_eval_run_unknown_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(app, ["eval", "run", "-m", "nonexistent", "--data-root", tmpdir])
        assert result.exit_code == 1
        assert "not found" in result.output


def test_eval_run_with_dummy_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir)
        design_dir = data_root / "design"
        mask_dir = data_root / "mask"
        design_dir.mkdir()
        mask_dir.mkdir()

        design = np.zeros((64, 64), dtype=np.float32)
        design[16:48, 16:48] = 1.0
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[14:50, 14:50] = 1.0

        for i in range(3):
            np.save(design_dir / f"sample_{i:04d}.npy", design)
            np.save(mask_dir / f"sample_{i:04d}.npy", mask)

        result = runner.invoke(
            app,
            ["eval", "run", "-m", "dummy-identity", "--data-root", tmpdir, "--limit", "3"],
        )
        assert result.exit_code == 0
        assert "epe_mean_nm" in result.output


def test_eval_run_json_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir)
        design_dir = data_root / "design"
        mask_dir = data_root / "mask"
        design_dir.mkdir()
        mask_dir.mkdir()

        arr = np.zeros((32, 32), dtype=np.float32)
        arr[8:24, 8:24] = 1.0
        np.save(design_dir / "s0.npy", arr)
        np.save(mask_dir / "s0.npy", arr)

        result = runner.invoke(
            app,
            ["eval", "run", "-m", "dummy-identity", "--data-root", tmpdir, "-f", "json"],
        )
        assert result.exit_code == 0
        # Output includes status lines before the JSON block; extract the JSON part.
        json_start = result.output.index("{")
        parsed = json.loads(result.output[json_start:])
        assert "epe_mean_nm" in parsed


def test_eval_run_save_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir)
        design_dir = data_root / "design"
        mask_dir = data_root / "mask"
        design_dir.mkdir()
        mask_dir.mkdir()

        arr = np.zeros((32, 32), dtype=np.float32)
        arr[8:24, 8:24] = 1.0
        np.save(design_dir / "s0.npy", arr)
        np.save(mask_dir / "s0.npy", arr)

        out_path = Path(tmpdir) / "report.md"
        result = runner.invoke(
            app,
            [
                "eval",
                "run",
                "-m",
                "dummy-identity",
                "--data-root",
                tmpdir,
                "-f",
                "markdown",
                "-o",
                str(out_path),
            ],
        )
        assert result.exit_code == 0
        assert out_path.exists()
        content = out_path.read_text()
        assert "epe_mean_nm" in content


def test_optimize_run_help():
    result = runner.invoke(app, ["optimize", "run", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--input" in output
    assert "--writer" in output
    assert "--drc-check" in output
    assert "--sha256" in output
    assert "--pretrained" in output


def test_eval_run_help_lists_sha256():
    result = runner.invoke(app, ["eval", "run", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--sha256" in output
    assert "--pretrained" in output


def test_eval_run_with_mrc():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir)
        design_dir = data_root / "design"
        mask_dir = data_root / "mask"
        design_dir.mkdir()
        mask_dir.mkdir()

        arr = np.zeros((32, 32), dtype=np.float32)
        arr[8:24, 8:24] = 1.0
        np.save(design_dir / "s0.npy", arr)
        np.save(mask_dir / "s0.npy", arr)

        result = runner.invoke(
            app,
            [
                "eval",
                "run",
                "-m",
                "dummy-identity",
                "--data-root",
                tmpdir,
                "--mrc",
                "--min-width-nm",
                "4",
                "--min-spacing-nm",
                "4",
                "-f",
                "json",
            ],
        )
        assert result.exit_code == 0
        json_start = result.output.index("{")
        parsed = json.loads(result.output[json_start:])
        assert "mrc_violation_rate" in parsed
        assert "mrc_passed" in parsed


def test_eval_run_no_mrc():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir)
        design_dir = data_root / "design"
        mask_dir = data_root / "mask"
        design_dir.mkdir()
        mask_dir.mkdir()

        arr = np.zeros((32, 32), dtype=np.float32)
        arr[8:24, 8:24] = 1.0
        np.save(design_dir / "s0.npy", arr)
        np.save(mask_dir / "s0.npy", arr)

        result = runner.invoke(
            app,
            [
                "eval",
                "run",
                "-m",
                "dummy-identity",
                "--data-root",
                tmpdir,
                "--no-mrc",
                "-f",
                "json",
            ],
        )
        assert result.exit_code == 0
        json_start = result.output.index("{")
        parsed = json.loads(result.output[json_start:])
        assert "mrc_violation_rate" not in parsed


def test_optimize_run_drc_check_smoke():
    """End-to-end smoke for `optimize --drc-check`.

    Pins the contract that optimize_cmd reads from MRCResult / DRCResult:
    .violation_count, .violation_rate, and .passed. If compliance.{mrc,drc}
    drift, this catches it via the CLI rather than only in unit tests.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        layout_arr = np.zeros((64, 64), dtype=np.float32)
        layout_arr[16:48, 16:48] = 1.0
        in_path = Path(tmpdir) / "in.npy"
        np.save(in_path, layout_arr)
        out_path = Path(tmpdir) / "out.oas"

        result = runner.invoke(
            app,
            [
                "optimize",
                "run",
                "-i",
                str(in_path),
                "-m",
                "dummy-identity",
                "-o",
                str(out_path),
                "--drc-check",
                "--tile-size",
                "64",
                "--overlap",
                "0",
                "--pixel-nm",
                "1.0",
            ],
        )
        assert result.exit_code == 0, result.output
        # Either explicit "All checks passed" or per-check violation lines —
        # both indicate the MRC/DRC contract was successfully consumed.
        output = _strip_ansi(result.output)
        assert ("All checks passed" in output) or ("violations" in output)
        assert "Optimization complete" in output
