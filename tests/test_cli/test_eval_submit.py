"""Tests for eval --submit integration with leaderboard."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from openlithohub.cli.app import app

runner = CliRunner()


def test_eval_run_with_submit() -> None:
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

        store_path = Path(tmpdir) / "lb.json"

        result = runner.invoke(
            app,
            [
                "eval",
                "run",
                "-m",
                "dummy-identity",
                "--data-root",
                tmpdir,
                "--submit",
                "--node",
                "45nm",
                "--topology",
                "manhattan",
            ],
            env={"OPENLITHOHUB_LEADERBOARD_PATH": str(store_path)},
        )
        assert result.exit_code == 0
        assert "Submitted to leaderboard" in result.output
        assert store_path.exists()

        data = json.loads(store_path.read_text())
        assert len(data["entries"]) == 1
        assert data["entries"][0]["model_name"] == "dummy-identity"


def test_eval_run_without_submit() -> None:
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
                "--no-submit",
            ],
        )
        assert result.exit_code == 0
        assert "Submitted to leaderboard" not in result.output
