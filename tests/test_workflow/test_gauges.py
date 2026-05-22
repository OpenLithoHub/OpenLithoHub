"""Tests for `openlithohub.workflow.gauges` — Calibre / CSV gauge parsing."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from openlithohub.workflow.gauges import (
    GaugePoint,
    GaugeTable,
    parse_gauge,
    parse_iccad13_gauge,
    write_iccad13_gauge,
)


def test_parse_calibre_no_header(tmp_path: Path) -> None:
    """No recognizable header → refuse the file rather than silently
    assuming a column order. Wrong-order columns previously produced
    confidently-wrong EPE numbers."""
    p = tmp_path / "site.gg"
    p.write_text(
        "# Calibre OPCverify gauge dump\n"
        "100.0 200.0 0.0 32.0 31.5 1.0\n"
        "150.5 200.0 90.0 32.0 32.4 2.0\n"
    )
    with pytest.raises(ValueError, match="no recognizable header"):
        parse_gauge(p)


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
    # CD error = measured - target (counts both edges).
    assert gt.cd_error() == (1.0, -2.0)
    # Single-edge EPE = half the CD error (matches compute_epe units).
    assert gt.epe() == (0.5, -1.0)
    # weighted RMS uses single-edge EPE: sqrt((1*0.25 + 3*1) / 4)
    assert math.isclose(gt.weighted_rms_epe(), math.sqrt((0.25 + 3.0) / 4))


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
    p.write_text(
        "\n# x y tangent target_cd measured_cd weight\n"
        "100 200 0 32 31.5 1\n\n"
        "150 200 90 32 32.4 2\n\n"
    )
    gt = parse_gauge(p)
    assert len(gt) == 2


def test_um_columns_are_converted_to_nm(tmp_path: Path) -> None:
    """``_um`` aliases are scaled by 1000x at parse time so downstream
    consumers (which assume nm) get correct values."""
    p = tmp_path / "site.csv"
    p.write_text("x_um,y_um,tangent,target_um,measured_um,weight\n0.1,0.2,0.0,0.032,0.0315,1.0\n")
    gt = parse_gauge(p)
    assert math.isclose(gt.points[0].x, 100.0)
    assert math.isclose(gt.points[0].y, 200.0)
    assert math.isclose(gt.points[0].target_cd, 32.0)
    assert math.isclose(gt.points[0].measured_cd or 0.0, 31.5)


class TestICCAD13Gauge:
    def test_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "iccad13.txt"
        p.write_text(
            "1\t100.0\t200.0\t0.0\t1\n2\t150.0\t200.0\t90.0\t-1\n3\t300.0\t400.0\t45.0\t1\n"
        )
        gt = parse_iccad13_gauge(p)
        assert len(gt) == 3
        assert gt.points[0].x == 100.0
        assert gt.points[1].tangent == 90.0
        assert gt.iccad13_polarities == (1, -1, 1)
        # Contest format has no measurement column.
        assert all(p.measured_cd is None for p in gt.points)

    def test_writer_round_trip(self, tmp_path: Path) -> None:
        points = (
            GaugePoint(x=10.0, y=20.0, tangent=0.0, target_cd=0.0, measured_cd=None, weight=1.0),
            GaugePoint(x=30.0, y=40.0, tangent=90.0, target_cd=0.0, measured_cd=None, weight=1.0),
        )
        out = tmp_path / "round.txt"
        write_iccad13_gauge(out, points, [1, -1])
        gt = parse_iccad13_gauge(out)
        assert len(gt) == 2
        assert gt.iccad13_polarities == (1, -1)
        assert math.isclose(gt.points[0].x, 10.0)
        assert math.isclose(gt.points[1].tangent, 90.0)

    def test_comments_and_blank_lines_ignored(self, tmp_path: Path) -> None:
        p = tmp_path / "iccad13.txt"
        p.write_text(
            "# ICCAD'13 gauge dump\n\n1 100 200 0 1\n# mid-file comment\n2 150 200 90 -1\n"
        )
        gt = parse_iccad13_gauge(p)
        assert len(gt) == 2

    def test_bad_polarity_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "iccad13.txt"
        p.write_text("1 100 200 0 2\n")
        with pytest.raises(ValueError, match="polarity"):
            parse_iccad13_gauge(p)

    def test_wrong_column_count_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "iccad13.txt"
        p.write_text("1 100 200 0\n")  # 4 columns, not 5
        with pytest.raises(ValueError, match="5 columns"):
            parse_iccad13_gauge(p)

    def test_writer_length_mismatch_rejected(self, tmp_path: Path) -> None:
        points = (
            GaugePoint(x=10.0, y=20.0, tangent=0.0, target_cd=0.0, measured_cd=None, weight=1.0),
        )
        with pytest.raises(ValueError, match="length mismatch"):
            write_iccad13_gauge(tmp_path / "x.txt", points, [1, -1])

    def test_writer_bad_polarity_rejected(self, tmp_path: Path) -> None:
        points = (
            GaugePoint(x=10.0, y=20.0, tangent=0.0, target_cd=0.0, measured_cd=None, weight=1.0),
        )
        with pytest.raises(ValueError, match="polarity"):
            write_iccad13_gauge(tmp_path / "x.txt", points, [3])
