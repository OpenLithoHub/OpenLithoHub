"""Tests for SARIF 2.1.0 export."""

from __future__ import annotations

import json
from pathlib import Path

from openlithohub.benchmark.compliance.drc import DRCResult
from openlithohub.benchmark.compliance.mrc import MRCResult
from openlithohub.benchmark.compliance.sarif import to_sarif, write_sarif


def _drc_result() -> DRCResult:
    return DRCResult(
        passed=False,
        violation_count=3,
        violations=[
            {"rule": 0.0, "type": 0.0, "x_nm": 100.0, "y_nm": 200.0, "threshold_nm": 40.0},
            {"rule": 1.0, "type": 1.0, "x_nm": 300.0, "y_nm": 400.0, "threshold_nm": 40.0},
            {
                "rule": 2.0,
                "type": 2.0,
                "x_nm": 500.0,
                "y_nm": 600.0,
                "actual_nm2": 50.0,
                "required_nm2": 100.0,
            },
        ],
        rule_summary={"min_width": 1, "min_spacing": 1, "min_area": 1},
    )


def _mrc_result() -> MRCResult:
    return MRCResult(
        passed=False,
        violation_count=2,
        violation_rate=0.001,
        violations=[
            {"type_code": 0.0, "x_nm": 10.0, "y_nm": 20.0, "actual_nm": 30.0, "required_nm": 40.0},
            {"type_code": 1.0, "x_nm": 50.0, "y_nm": 60.0, "actual_nm": 25.0, "required_nm": 40.0},
        ],
        width_violation_count=1,
        spacing_violation_count=1,
    )


def test_sarif_schema_version():
    sarif = to_sarif()
    assert sarif["version"] == "2.1.0"
    assert "$schema" in sarif
    assert "runs" in sarif


def test_sarif_empty_results():
    sarif = to_sarif()
    assert len(sarif["runs"]) == 1
    assert sarif["runs"][0]["results"] == []
    assert sarif["runs"][0]["tool"]["driver"]["name"] == "OpenLithoHub"


def test_sarif_drc_violations():
    sarif = to_sarif(drc_result=_drc_result(), pixel_size_nm=1.0)
    results = sarif["runs"][0]["results"]
    assert len(results) == 3

    assert results[0]["ruleId"] == "DRC/MIN-WIDTH"
    assert results[0]["level"] == "error"
    loc = results[0]["locations"][0]["physicalLocation"]["region"]
    assert loc["startLine"] == 201
    assert loc["startColumn"] == 101

    assert results[1]["ruleId"] == "DRC/MIN-SPACING"
    assert results[2]["ruleId"] == "DRC/MIN-AREA"
    assert "Area" in results[2]["message"]["text"]


def test_sarif_mrc_violations():
    sarif = to_sarif(mrc_result=_mrc_result(), pixel_size_nm=2.0)
    results = sarif["runs"][0]["results"]
    assert len(results) == 2

    assert results[0]["ruleId"] == "MRC/MIN-WIDTH"
    assert results[1]["ruleId"] == "MRC/MIN-SPACING"

    loc = results[0]["locations"][0]["physicalLocation"]["region"]
    assert loc["startLine"] == 11
    assert loc["startColumn"] == 6


def test_sarif_combined_drc_mrc():
    sarif = to_sarif(drc_result=_drc_result(), mrc_result=_mrc_result())
    assert len(sarif["runs"][0]["results"]) == 5


def test_sarif_tool_driver_rules():
    sarif = to_sarif(drc_result=_drc_result(), mrc_result=_mrc_result())
    rules = sarif["runs"][0]["tool"]["driver"]["rules"]
    rule_ids = [r["id"] for r in rules]
    assert "DRC/MIN-WIDTH" in rule_ids
    assert "DRC/MIN-SPACING" in rule_ids
    assert "DRC/MIN-AREA" in rule_ids
    assert "MRC/MIN-WIDTH" in rule_ids
    assert "MRC/MIN-SPACING" in rule_ids


def test_sarif_zero_violations():
    sarif = to_sarif(
        drc_result=DRCResult(passed=True, violation_count=0, violations=[], rule_summary={}),
        mrc_result=MRCResult(
            passed=True,
            violation_count=0,
            violation_rate=0.0,
            violations=[],
            width_violation_count=0,
            spacing_violation_count=0,
        ),
    )
    assert sarif["runs"][0]["results"] == []


def test_write_sarif_file(tmp_path: Path):
    out = tmp_path / "results.sarif"
    write_sarif(out, drc_result=_drc_result(), pixel_size_nm=1.0)

    data = json.loads(out.read_text())
    assert data["version"] == "2.1.0"
    assert len(data["runs"][0]["results"]) == 3


def test_write_sarif_round_trip(tmp_path: Path):
    out = tmp_path / "rt.sarif"
    drc = _drc_result()
    mrc = _mrc_result()
    write_sarif(out, drc_result=drc, mrc_result=mrc, pixel_size_nm=1.0)

    data = json.loads(out.read_text())
    assert data["$schema"] == "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json"
    assert len(data["runs"][0]["results"]) == 5

    for result in data["runs"][0]["results"]:
        assert "ruleId" in result
        assert "level" in result
        assert result["level"] == "error"
        assert "message" in result
        assert "locations" in result
        assert len(result["locations"]) == 1
