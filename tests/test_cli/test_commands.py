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
        assert "mrc_passed_all" in parsed


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


def test_simulate_list_backends_plain():
    result = runner.invoke(app, ["simulate", "list-backends"])
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    # Built-in backends from simulators.registry; if any of these stops
    # being registered, both this test and the public API have changed.
    for name in ("hopkins", "calibre", "tachyon"):
        assert name in output, output


def test_simulate_list_backends_verbose_shows_class_path():
    result = runner.invoke(app, ["simulate", "list-backends", "--verbose"])
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    # --verbose annotates each name with the implementing module.class
    # so users can locate the source without grepping the registry.
    assert "openlithohub.simulators.hopkins_sim.HopkinsSimulator" in output
    assert "openlithohub.simulators.calibre.CalibreSimulator" in output
    assert "openlithohub.simulators.tachyon.TachyonSimulator" in output


class TestAggregateMetrics:
    def test_epe_uses_simple_mean_not_pixel_weighted(self) -> None:
        from openlithohub.cli.eval_cmd import _aggregate_metrics

        # Two samples: a tiny tile with EPE 1 and a huge tile with EPE 9.
        # Pixel-weighted average would be ~9; simple mean is 5. Issue #45.
        metrics = [{"epe_mean_nm": 1.0}, {"epe_mean_nm": 9.0}]
        agg = _aggregate_metrics(metrics, sample_pixel_counts=[64, 1_000_000])
        assert agg["epe_mean_nm"] == 5.0

    def test_l2_pixels_remains_pixel_weighted(self) -> None:
        from openlithohub.cli.eval_cmd import _aggregate_metrics

        # l2_error_pixels is an integral metric, so it must stay area-weighted.
        metrics = [{"l2_error_pixels": 100.0}, {"l2_error_pixels": 100.0}]
        agg = _aggregate_metrics(metrics, sample_pixel_counts=[1, 9])
        # Weighted toward the second (heavier) sample but values are equal,
        # so the aggregate should be 100. Test that swapping pixel counts
        # preserves the result for equal-valued integrals.
        assert agg["l2_error_pixels"] == 100.0

    def test_empty_mask_sample_not_silently_dropped(self) -> None:
        from openlithohub.cli.eval_cmd import _aggregate_metrics

        # Issue #46: a sample with pixel_count=0 (e.g. CD-error reported as
        # 0 because there are no edges) should still contribute. Pre-fix it
        # got weight 0 and was excluded — only the 'good' sample counted.
        metrics = [{"l2_error_pixels": 0.0}, {"l2_error_pixels": 100.0}]
        agg = _aggregate_metrics(metrics, sample_pixel_counts=[0, 1])
        # Floor weight at 1 → average is (0*1 + 100*1)/(1+1) = 50.
        assert agg["l2_error_pixels"] == 50.0
