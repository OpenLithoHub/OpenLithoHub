"""Tests for the CLI module."""

from typer.testing import CliRunner

from openlithohub.cli.app import app

runner = CliRunner()


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
    assert "--model" in result.output
    assert "--dataset" in result.output


def test_optimize_run_help():
    result = runner.invoke(app, ["optimize", "run", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--writer" in result.output
    assert "--drc-check" in result.output
