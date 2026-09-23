"""PR-C: `optimize run` exposes the shared execution planner."""

from __future__ import annotations

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


def test_optimize_run_streaming_raster_artifact() -> None:
    """`--execution-mode streaming` + .npy output produces the memmap
    raster artifact with the exact-core streaming semantics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        layout = np.zeros((96, 96), dtype=np.float32)
        layout[20:76, 30:80] = 1.0
        in_path = Path(tmpdir) / "in.npy"
        np.save(in_path, layout)
        out_path = Path(tmpdir) / "out.npy"

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
                "--tile-size",
                "32",
                "--halo",
                "8",
                "--pixel-nm",
                "1.0",
                "--threshold",
                "0.5",
                "--execution-mode",
                "streaming",
            ],
        )
        assert result.exit_code == 0, result.output
        output = result.output
        assert "Plan: STREAMING [STREAMING_SUPPORTED]" in output
        assert "Optimization complete" in output
        artifact = np.load(out_path, mmap_mode="r")
        assert artifact.shape == (96, 96)
        # Identity model + 0.5 threshold: artifact equals the binarised input.
        assert np.array_equal(np.asarray(artifact), (layout > 0.5).astype(np.float32))


def test_optimize_run_dense_raster_artifact_matches_streaming() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(3)
        layout = (rng.random((80, 80)) > 0.5).astype(np.float32)
        in_path = Path(tmpdir) / "in.npy"
        np.save(in_path, layout)
        outs = {}
        for mode in ("dense", "streaming"):
            out_path = Path(tmpdir) / f"out_{mode}.npy"
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
                    "--tile-size",
                    "24",
                    "--halo",
                    "6",
                    "--pixel-nm",
                    "1.0",
                    "--threshold",
                    "0.5",
                    "--execution-mode",
                    mode,
                ],
            )
            assert result.exit_code == 0, result.output
            outs[mode] = np.asarray(np.load(out_path, mmap_mode="r"))
        assert np.array_equal(outs["dense"], outs["streaming"])


def test_optimize_run_streaming_unsupported_fails_honestly() -> None:
    """streaming + curvilinear writer is a 400-class CLI error, never a
    silent dense fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "in.npy"
        np.save(in_path, np.zeros((32, 32), dtype=np.float32))
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
                str(Path(tmpdir) / "out.oas"),
                "--writer",
                "mbmw",
                "--pixel-nm",
                "1.0",
                "--execution-mode",
                "streaming",
            ],
        )
        assert result.exit_code == 1
        output = result.output
        assert "STREAMING_UNSUPPORTED_OUTPUT" in output


def test_optimize_run_invalid_execution_mode_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "in.npy"
        np.save(in_path, np.zeros((16, 16), dtype=np.float32))
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
                str(Path(tmpdir) / "out.oas"),
                "--execution-mode",
                "turbo",
            ],
        )
        assert result.exit_code != 0
        assert "execution-mode" in _strip_ansi(result.output)


def test_optimize_run_help_lists_execution_mode() -> None:
    result = runner.invoke(app, ["optimize", "run", "--help"])
    assert result.exit_code == 0
    flat = re.sub(r"\s+", " ", _strip_ansi(result.output))
    assert "--execution-mode" in flat
    # Rich truncates long help in the boxed layout; the env-var prefix and
    # the policy wording still identify the option honestly.
    assert "OPENLITHOHUB_MAX" in flat
    assert "dense memory policy" in flat


def test_optimize_run_json_decision_shape_smoke() -> None:
    """The planner prints a parseable plan line; light structural check on
    the decision vocabulary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "in.npy"
        np.save(in_path, np.zeros((40, 40), dtype=np.float32))
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
                str(Path(tmpdir) / "out.npy"),
                "--tile-size",
                "16",
                "--pixel-nm",
                "1.0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Plan: DENSE [DENSE_SMALL_LAYOUT]" in result.output
        assert "input=dense-raster" in result.output
