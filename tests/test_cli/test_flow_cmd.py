"""Tests for `openlithohub flow` CLI observability flags."""

from __future__ import annotations

import json

import pytest

typer = pytest.importorskip("typer")
kdb = pytest.importorskip("klayout.db")

from typer.testing import CliRunner  # noqa: E402

from openlithohub.cli.flow_cmd import flow_app  # noqa: E402


@pytest.fixture
def tiny_gds(tmp_path):
    """A small GDS with metal1 rectangles on the orfs_asap7 layermap (20/0)."""
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    m1 = layout.layer(20, 0)
    top.shapes(m1).insert(kdb.Box(0, 0, 1500, 300))
    top.shapes(m1).insert(kdb.Box(0, 500, 1500, 900))
    top.shapes(m1).insert(kdb.Box(0, 1100, 1500, 1600))
    path = tmp_path / "tiny.gds"
    layout.write(str(path))
    return path


def test_flow_report_json_and_profile(tiny_gds, tmp_path) -> None:
    report_path = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        flow_app,
        [
            str(tiny_gds),
            "--pdk",
            "orfs_asap7",
            "--layer",
            "metal1",
            "--pixel-nm",
            "20.0",
            "--tile-nm",
            "1000.0",
            "--node",
            "45nm",
            "--profile",
            "--report-json",
            str(report_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert "report" in data and "profile" in data
    profile = data["profile"]
    for key in (
        "parse_and_tile_wall_s",
        "tile_count",
        "forward_wall_s_total",
        "metrics_wall_s_total",
        "end_to_end_wall_s",
        "peak_rss_bytes",
        "forward_calls",
    ):
        assert key in profile, f"missing profile key {key}"
    assert profile["tile_count"] == profile["forward_calls"]
    assert profile["tile_count"] >= 1
    assert "Flow Profile" in result.output
