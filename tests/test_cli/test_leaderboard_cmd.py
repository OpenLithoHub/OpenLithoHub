"""Tests for the leaderboard CLI commands."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from openlithohub.cli.app import app

runner = CliRunner()


def test_leaderboard_view_empty() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "lb.json"
        result = runner.invoke(app, ["leaderboard", "view", "--store", str(store)])
        assert result.exit_code == 0
        assert "No leaderboard entries" in result.output


def test_leaderboard_submit_inline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "lb.json"
        result = runner.invoke(
            app,
            [
                "leaderboard",
                "submit",
                "--model",
                "my-model",
                "--dataset",
                "lithobench",
                "--node",
                "7nm",
                "--topology",
                "manhattan",
                "--epe-mean",
                "2.5",
                "--epe-max",
                "8.0",
                "--paper-url",
                "https://arxiv.org/abs/2024.12345",
                "--store",
                str(store),
            ],
        )
        assert result.exit_code == 0
        assert "Submitted!" in result.output


def test_leaderboard_submit_from_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "lb.json"
        data_file = Path(tmpdir) / "entry.json"
        data_file.write_text(
            json.dumps(
                {
                    "model_name": "file-model",
                    "dataset": "lithosim",
                    "process_node": "3nm-euv",
                    "mask_topology": "curvilinear",
                    "epe_mean_nm": 1.1,
                    "epe_max_nm": 3.0,
                }
            )
        )
        result = runner.invoke(
            app,
            ["leaderboard", "submit", "--file", str(data_file), "--store", str(store)],
        )
        assert result.exit_code == 0
        assert "Submitted!" in result.output


def test_leaderboard_view_after_submit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "lb.json"
        runner.invoke(
            app,
            [
                "leaderboard",
                "submit",
                "--model",
                "model-a",
                "--dataset",
                "lithobench",
                "--node",
                "7nm",
                "--topology",
                "manhattan",
                "--epe-mean",
                "3.0",
                "--epe-max",
                "9.0",
                "--store",
                str(store),
            ],
        )
        result = runner.invoke(app, ["leaderboard", "view", "--store", str(store)])
        assert result.exit_code == 0
        assert "model-a" in result.output


def test_leaderboard_view_json_format() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "lb.json"
        runner.invoke(
            app,
            [
                "leaderboard",
                "submit",
                "--model",
                "model-x",
                "--dataset",
                "lithobench",
                "--node",
                "5nm",
                "--topology",
                "curvilinear",
                "--epe-mean",
                "1.5",
                "--epe-max",
                "4.0",
                "--store",
                str(store),
            ],
        )
        result = runner.invoke(
            app, ["leaderboard", "view", "-f", "json", "--store", str(store)]
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) == 1
        assert parsed[0]["model_name"] == "model-x"


def test_leaderboard_export() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "lb.json"
        out = Path(tmpdir) / "export.json"
        runner.invoke(
            app,
            [
                "leaderboard",
                "submit",
                "--model",
                "m1",
                "--dataset",
                "lithobench",
                "--node",
                "7nm",
                "--topology",
                "manhattan",
                "--epe-mean",
                "2.0",
                "--epe-max",
                "6.0",
                "--store",
                str(store),
            ],
        )
        result = runner.invoke(
            app,
            ["leaderboard", "export", "-o", str(out), "--store", str(store)],
        )
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == 1


def test_leaderboard_submit_validation_error() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "lb.json"
        result = runner.invoke(
            app,
            [
                "leaderboard",
                "submit",
                "--model",
                "bad",
                "--dataset",
                "lithobench",
                "--node",
                "invalid-node",
                "--topology",
                "manhattan",
                "--epe-mean",
                "2.0",
                "--epe-max",
                "6.0",
                "--store",
                str(store),
            ],
        )
        assert result.exit_code == 1
        assert "Validation error" in result.output


def test_leaderboard_help() -> None:
    result = runner.invoke(app, ["leaderboard", "--help"])
    assert result.exit_code == 0
    assert "view" in result.output
    assert "submit" in result.output
    assert "export" in result.output
