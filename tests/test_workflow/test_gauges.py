"""Tests for `openlithohub.workflow.gauges` — Calibre / CSV gauge parsing."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from openlithohub.workflow.gauges import GaugePoint, GaugeTable, parse_gauge


def test_parse_calibre_no_header(tmp_path: Path) -> None:
    """No header → assume canonical column order."""
    p = tmp_path / "site.gg"
    p.write_text(
        "# Calibre OPCverify gauge dump\n"
        "100.0 200.0 0.0 32.0 31.5 1.0\n"
        "150.5 200.0 90.0 32.0 32.4 2.0\n"
    )
    gt = parse_gauge(p)
    assert len(gt) == 2
    assert gt.points[0] == GaugePoint(100.0, 200.0, 0.0, 32.0, 31.5, 1.0)
    assert gt.points[1].weight == 2.0


def test_parse_calibre_with_header(tmp_path: Path) -> None:
    """Header line names columns; column order may differ from canonical."""
    p = tmp_path / "site.gg"
    p.write_text(
        "# id\n"  # not a header (single non-canonical token, but this is fine — it's a comment)
        "# x y angle target measured weight\n"
        "100.0 200.0 0.0 32.0 31.5 1.0\n"
    )
    gt = parse_gauge(p)
    assert gt.points[0].tangent == 0.0
    assert gt.points[0].target_cd == 32.0


def test_parse_csv(tmp_path: Path) -> None:
    p = tmp_path / "site.csv"
    p.write_text(
        "x_nm,y_nm,angle,cd_target,cd_measured,w\n"
        "100.0,200.0,0.0,32.0,31.5,1.0\n"
        "150.5,200.0,90.0,32.0,32.4,2.0\n"
    )
    gt = parse_gauge(p)
    assert len(gt) == 2
    assert gt.points[0].x == 100.0
    assert gt.points[1].weight == 2.0


def test_parse_csv_missing_measured_is_none(tmp_path: Path) -> None:
    """Pre-measurement gauges: target only, measured left blank or NA."""
    p = tmp_path / "site.csv"
    p.write_text("x,y,tangent,target_cd,measured_cd,weight\n0,0,0,32.0,,1.0\n1,1,0,32.0,NA,1.0\n")
    gt = parse_gauge(p)
    assert gt.points[0].measured_cd is None
    assert gt.points[1].measured_cd is None


def test_weight_defaults_to_one_when_column_missing(tmp_path: Path) -> None:
    p = tmp_path / "site.csv"
    p.write_text("x,y,tangent,target_cd,measured_cd\n0,0,0,32.0,31.0\n")
    gt = parse_gauge(p)
    assert gt.points[0].weight == 1.0


def test_missing_required_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("x,y,target_cd,measured_cd\n0,0,32.0,31.0\n")  # no tangent
    with pytest.raises(ValueError, match="missing required column"):
        parse_gauge(p)


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported gauge format"):
        parse_gauge(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_gauge(tmp_path / "nope.csv")


def test_epe_and_weighted_rms(tmp_path: Path) -> None:
    p = tmp_path / "site.csv"
    p.write_text(
        "x,y,tangent,target_cd,measured_cd,weight\n0,0,0,32.0,33.0,1.0\n1,1,0,32.0,30.0,3.0\n"
    )
    gt = parse_gauge(p)
    assert gt.epe() == (1.0, -2.0)
    # sqrt((1*1 + 3*4) / 4) = sqrt(13/4)
    assert math.isclose(gt.weighted_rms_epe(), math.sqrt(13 / 4))


def test_epe_raises_when_unmeasured(tmp_path: Path) -> None:
    p = tmp_path / "site.csv"
    p.write_text("x,y,tangent,target_cd,measured_cd\n0,0,0,32.0,\n")
    gt = parse_gauge(p)
    with pytest.raises(ValueError, match="no measured_cd"):
        gt.epe()


def test_zero_weight_total_raises(tmp_path: Path) -> None:
    p = tmp_path / "site.csv"
    p.write_text("x,y,tangent,target_cd,measured_cd,weight\n0,0,0,32.0,33.0,0.0\n")
    gt = parse_gauge(p)
    with pytest.raises(ValueError, match="weights are zero"):
        gt.weighted_rms_epe()


def test_reexport_from_workflow_namespace() -> None:
    import openlithohub.workflow as wf

    assert wf.GaugePoint is GaugePoint
    assert wf.GaugeTable is GaugeTable
    assert wf.parse_gauge is parse_gauge


def test_blank_lines_ignored_in_calibre(tmp_path: Path) -> None:
    p = tmp_path / "site.gg"
    p.write_text("\n# header comment\n100 200 0 32 31.5 1\n\n150 200 90 32 32.4 2\n\n")
    gt = parse_gauge(p)
    assert len(gt) == 2
