"""SARIF 2.1.0 export for DRC/MRC violation results.

Converts DRCResult and MRCResult violation lists into a SARIF 2.1.0
JSON structure consumable by GitHub Code Scanning and VSCode SARIF Viewer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openlithohub.benchmark.compliance.drc import DRCResult
from openlithohub.benchmark.compliance.mrc import MRCResult
from openlithohub.benchmark.compliance.rule_deck import RuleDeck

_SARIF_SCHEMA = "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json"
_SARIF_VERSION = "2.1.0"
_TOOL_NAME = "OpenLithoHub"
_TOOL_INFO_URI = "https://github.com/OpenLithoHub/OpenLithoHub"

_DRC_RULE_MAP: dict[float, str] = {
    0.0: "DRC/MIN-WIDTH",
    1.0: "DRC/MIN-SPACING",
    2.0: "DRC/MIN-AREA",
    3.0: "DRC/MIN-NOTCH",
}

_DRC_RULE_DESCRIPTIONS: dict[str, str] = {
    "DRC/MIN-WIDTH": "Feature width below minimum threshold",
    "DRC/MIN-SPACING": "Spacing between features below minimum threshold",
    "DRC/MIN-AREA": "Feature area below minimum threshold",
    "DRC/MIN-NOTCH": "Notch width below minimum threshold",
}

_MRC_RULE_MAP: dict[float, str] = {
    0.0: "MRC/MIN-WIDTH",
    1.0: "MRC/MIN-SPACING",
}

_MRC_RULE_DESCRIPTIONS: dict[str, str] = {
    "MRC/MIN-WIDTH": "Mask feature width below manufacturing minimum",
    "MRC/MIN-SPACING": "Mask feature spacing below manufacturing minimum",
}


def _drc_violation_to_result(v: dict[str, float], pixel_size_nm: float) -> dict[str, Any]:
    rule_id = _DRC_RULE_MAP.get(v.get("rule", -1.0), "DRC/UNKNOWN")
    x_nm = v.get("x_nm", 0.0)
    y_nm = v.get("y_nm", 0.0)
    threshold = v.get("threshold_nm", 0.0)

    if rule_id == "DRC/MIN-AREA":
        actual = v.get("actual_nm2", 0.0)
        required = v.get("required_nm2", 0.0)
        msg = f"Area {actual:.1f} nm² < required {required:.1f} nm²"
    else:
        msg = f"Threshold {threshold:.1f} nm violated at ({x_nm:.1f}, {y_nm:.1f}) nm"

    col = int(x_nm / pixel_size_nm) if pixel_size_nm > 0 else 0
    row = int(y_nm / pixel_size_nm) if pixel_size_nm > 0 else 0

    return {
        "ruleId": rule_id,
        "level": "error",
        "message": {"text": msg},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": "mask"},
                    "region": {
                        "startLine": row + 1,
                        "startColumn": col + 1,
                    },
                }
            }
        ],
    }


def _mrc_violation_to_result(v: dict[str, float], pixel_size_nm: float) -> dict[str, Any]:
    rule_id = _MRC_RULE_MAP.get(v.get("type_code", -1.0), "MRC/UNKNOWN")
    x_nm = v.get("x_nm", 0.0)
    y_nm = v.get("y_nm", 0.0)
    actual = v.get("actual_nm", 0.0)
    required = v.get("required_nm", 0.0)

    col = int(x_nm / pixel_size_nm) if pixel_size_nm > 0 else 0
    row = int(y_nm / pixel_size_nm) if pixel_size_nm > 0 else 0

    return {
        "ruleId": rule_id,
        "level": "error",
        "message": {"text": f"Actual {actual:.1f} nm < required {required:.1f} nm"},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": "mask"},
                    "region": {
                        "startLine": row + 1,
                        "startColumn": col + 1,
                    },
                }
            }
        ],
    }


def _build_tool_driver(
    drc_result: DRCResult | None = None,
    mrc_result: MRCResult | None = None,
    rule_deck: RuleDeck | None = None,
) -> dict[str, Any]:
    rules: list[dict[str, Any]] = []
    seen: set[str] = set()

    if drc_result is not None:
        for v in drc_result.violations:
            rid = _DRC_RULE_MAP.get(v.get("rule", -1.0), "DRC/UNKNOWN")
            if rid not in seen:
                seen.add(rid)
                rules.append(
                    {"id": rid, "shortDescription": {"text": _DRC_RULE_DESCRIPTIONS.get(rid, "")}}
                )

    if mrc_result is not None:
        for v in mrc_result.violations:
            rid = _MRC_RULE_MAP.get(v.get("type_code", -1.0), "MRC/UNKNOWN")
            if rid not in seen:
                seen.add(rid)
                rules.append(
                    {"id": rid, "shortDescription": {"text": _MRC_RULE_DESCRIPTIONS.get(rid, "")}}
                )

    if rule_deck is not None:
        for rname, threshold in rule_deck.rules.items():
            rid = f"RULE-DECK/{rname.upper()}"
            if rid not in seen:
                seen.add(rid)
                rules.append(
                    {
                        "id": rid,
                        "shortDescription": {"text": f"{rname} (threshold: {threshold})"},
                    }
                )

    driver: dict[str, Any] = {
        "name": _TOOL_NAME,
        "informationUri": _TOOL_INFO_URI,
        "rules": rules,
    }
    return driver


def to_sarif(
    drc_result: DRCResult | None = None,
    mrc_result: MRCResult | None = None,
    pixel_size_nm: float = 1.0,
    rule_deck: RuleDeck | None = None,
) -> dict[str, Any]:
    """Convert DRC/MRC results to SARIF 2.1.0 JSON structure.

    Args:
        drc_result: Optional DRC result.
        mrc_result: Optional MRC result.
        pixel_size_nm: Pixel size for coordinate conversion.
        rule_deck: Optional rule deck for tool driver rules.

    Returns:
        SARIF 2.1.0 compliant dict.
    """
    results: list[dict[str, Any]] = []

    if drc_result is not None:
        results.extend(_drc_violation_to_result(v, pixel_size_nm) for v in drc_result.violations)

    if mrc_result is not None:
        results.extend(_mrc_violation_to_result(v, pixel_size_nm) for v in mrc_result.violations)

    return {
        "version": _SARIF_VERSION,
        "$schema": _SARIF_SCHEMA,
        "runs": [
            {
                "tool": {
                    "driver": _build_tool_driver(drc_result, mrc_result, rule_deck),
                },
                "results": results,
            }
        ],
    }


def write_sarif(
    path: str | Path,
    drc_result: DRCResult | None = None,
    mrc_result: MRCResult | None = None,
    pixel_size_nm: float = 1.0,
    rule_deck: RuleDeck | None = None,
) -> None:
    """Write DRC/MRC results as SARIF 2.1.0 JSON file.

    Args:
        path: Output file path.
        drc_result: Optional DRC result.
        mrc_result: Optional MRC result.
        pixel_size_nm: Pixel size for coordinate conversion.
        rule_deck: Optional rule deck.
    """
    sarif = to_sarif(drc_result, mrc_result, pixel_size_nm, rule_deck)
    Path(path).write_text(json.dumps(sarif, indent=2))
