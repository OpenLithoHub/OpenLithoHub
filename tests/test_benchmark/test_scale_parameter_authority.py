"""Scale parameter authority + layer-audit evidence tests (PR-V3.1.3).

The 2026-10-05 review found that Microwatt's ``66:44`` had no authority
of its own: the value propagated from Ibex by analogy and its rationale
lived only in prose.  These tests close that class mechanically:

* the source-owned authority table validates fail-closed (``analogy``
  is never a valid basis; every design declares fixture-tracking
  semantics and a frozen GDS sha);
* the committed layer-audit evidence re-derives the Microwatt layer
  decision from the frozen GDS bytes and is internally consistent with
  the tracked fixture manifest;
* the docs quote the evidence numbers (a doc that loses a number, or
  drifts from the evidence, fails CI);
* the audit TOOL is exercised end to end on a synthetic GDS and proven
  to reject stale/substituted artifacts (the machine-hygiene gate, CI-
  testable even though host-local machine state itself is not).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from openlithohub.benchmark.scale_parameter_authority import (
    IBEX_FROZEN_GDS_SHA256,
    SCALE_PARAMETER_AUTHORITY,
    VALID_AUTHORITY_BASES,
    parameter_authority,
    validate_scale_parameter_authority,
)

REPO = Path(__file__).resolve().parents[2]
MICROWATT_MANIFEST = (
    REPO / "benchmarks/results/industrial-scale/fixtures/microwatt/fixture-manifest.json"
)
MICROWATT_AUDIT = REPO / "benchmarks/results/industrial-scale/audits/microwatt-layer-audit.json"
RUNBOOK = REPO / "docs/industrial-scale-gpu-runbook.md"
BENCH_DOC = REPO / "docs/industrial-scale-benchmark.md"
AUDIT_SCRIPT = REPO / "scripts" / "audit_industrial_scale_fixture_layer.py"


def _norm(text: str) -> str:
    return " ".join(text.split())


# ---- frozen authority table ------------------------------------------------------


def test_authority_table_self_validates() -> None:
    assert validate_scale_parameter_authority() == []


def test_analogy_is_never_a_valid_basis() -> None:
    assert "analogy" not in VALID_AUTHORITY_BASES
    assert set(VALID_AUTHORITY_BASES) == {"lineage", "direct_audit"}
    for design, entry in SCALE_PARAMETER_AUTHORITY["designs"].items():
        for name, record in entry["parameters"].items():
            assert record["authority_basis"] in VALID_AUTHORITY_BASES, (
                f"{design}.{name} has no own authority — analogy inheritance "
                "is the exact defect class PR-V3.1.3 closes"
            )
            assert record["status"] == "FROZEN"


def test_rule_text_bans_inheritance() -> None:
    rule = SCALE_PARAMETER_AUTHORITY["rule"]
    assert "analogy" in rule and "UNRESOLVED" in rule


def test_microwatt_layer_matches_tracked_manifest() -> None:
    manifest = json.loads(MICROWATT_MANIFEST.read_text(encoding="utf-8"))
    record = parameter_authority("microwatt", "selected_layer")
    assert record["value"] == manifest["selected_layer"] == "66:44"
    entry = SCALE_PARAMETER_AUTHORITY["designs"]["microwatt"]
    assert entry["frozen_gds_sha256"] == manifest["gds_sha256"]
    assert record["authority_basis"] == "direct_audit"


def test_ibex_basis_is_lineage_not_analogy() -> None:
    record = parameter_authority("ibex", "selected_layer")
    assert record["authority_basis"] == "lineage"
    assert record["value"] == "66:44"
    assert len(IBEX_FROZEN_GDS_SHA256) == 64
    assert all(c in "0123456789abcdef" for c in IBEX_FROZEN_GDS_SHA256)


def test_every_design_declares_fixture_tracking_semantics() -> None:
    for _design, entry in SCALE_PARAMETER_AUTHORITY["designs"].items():
        tracking = entry["fixture_tracking"]
        assert "tracked_paths" in tracking and "host_local_paths" in tracking
        assert "tracked_tree_gate_visibility" in tracking
        # the declared visibility sentence must state the gate's blindness
        # or tripwire behavior — the asymmetry is protocol, not accident
        visibility = tracking["tracked_tree_gate_visibility"]
        assert "gate" in visibility and "SHA-256" in visibility


# ---- committed layer-audit evidence ----------------------------------------------


@pytest.fixture(scope="module")
def microwatt_audit() -> dict:
    assert MICROWATT_AUDIT.is_file(), (
        "committed audit evidence missing — regenerate with "
        "scripts/audit_industrial_scale_fixture_layer.py"
    )
    return json.loads(MICROWATT_AUDIT.read_text(encoding="utf-8"))


def test_audit_schema_and_sha_binding(microwatt_audit: dict) -> None:
    manifest = json.loads(MICROWATT_MANIFEST.read_text(encoding="utf-8"))
    assert microwatt_audit["schema"] == "OpenLithoHub.scale-fixture-layer-audit.v1"
    assert microwatt_audit["gds_sha256"] == manifest["gds_sha256"]
    assert microwatt_audit["gds_bytes"] == manifest["gds_bytes"]
    assert microwatt_audit["source_commit"] == manifest["source_commit"]
    assert microwatt_audit["top_cell"] == manifest["top_cell"]
    assert microwatt_audit["selected_layer"] == manifest["selected_layer"]
    assert microwatt_audit["layer_count"] == len(manifest["layers"])
    assert microwatt_audit["fixture_manifest_sha256"] is not None


def test_audit_verdict_matches_the_prose_claims(microwatt_audit: dict) -> None:
    verdict = microwatt_audit["selected_layer_audit"]
    assert verdict["layer"] == "66:44"
    assert verdict["non_empty"] is True
    assert verdict["flattened_instances"] == 27_487_849
    assert verdict["density_rank_by_flattened_instances"] == 2
    assert verdict["non_empty_layers"] == 41
    assert verdict["coverage_x_pct"] == 98.8
    assert verdict["coverage_y_pct"] == 99.4
    # the recorded rank must be recomputable from the committed per-layer data
    ranked = sorted(
        microwatt_audit["per_layer"],
        key=lambda e: e["flattened_instances"],
        reverse=True,
    )
    assert ranked[verdict["density_rank_by_flattened_instances"] - 1]["layer"] == "66:44"


def test_docs_quote_the_evidence_numbers(microwatt_audit: dict) -> None:
    verdict = microwatt_audit["selected_layer_audit"]
    runbook = _norm(RUNBOOK.read_text(encoding="utf-8"))
    bench = _norm(BENCH_DOC.read_text(encoding="utf-8"))
    # the exact instance count, with thousands separators, in both docs
    assert f"{verdict['flattened_instances']:,}" in runbook
    assert f"{verdict['flattened_instances']:,}" in bench
    assert "second densest of 41 non-empty layers" in runbook
    assert "98.8% × 99.4%" in runbook
    # both frozen GDS sha256 values must appear in full in the runbook —
    # the missing-ibex-number defect class
    assert SCALE_PARAMETER_AUTHORITY["designs"]["ibex"]["frozen_gds_sha256"] in runbook
    assert SCALE_PARAMETER_AUTHORITY["designs"]["microwatt"]["frozen_gds_sha256"] in runbook
    # the audit artifact must be reachable from the docs
    assert "audits/microwatt-layer-audit.json" in runbook
    assert "audits/microwatt-layer-audit.json" in bench


def test_review_questions_section_is_present() -> None:
    bench = _norm(BENCH_DOC.read_text(encoding="utf-8"))
    assert "Adding a new design — fixed review questions" in bench
    for needle in (
        "analogy",
        "tracked-tree gate",
        "resource margins",
        "stale artifacts",
    ):
        assert needle.lower() in bench.lower(), f"review question missing: {needle}"


# ---- the audit tool itself (synthetic GDS, no real PDB data) ----------------------

AUDIT_MODULE_NAME = "audit_industrial_scale_fixture_layer"


def _build_synthetic_fixture(tmp_path: Path) -> tuple[Path, dict]:
    """A tiny GDS exercising both metrics: 66:44 has 3 cell-local
    elements (1 top + 2 in the child cell) and 7 flattened instances
    (1 + 3 placements × 2); 67:20 has 1 of each."""
    import klayout.db as kdb

    ly = kdb.Layout()
    ly.dbu = 0.001  # 1 nm per DBU
    li = ly.layer(66, 44)
    extra = ly.layer(67, 20)
    top = ly.create_cell("test_top")
    sub = ly.create_cell("test_sub")
    sub.shapes(li).insert(kdb.Box(0, 0, 100, 100))
    sub.shapes(li).insert(kdb.Box(150, 0, 250, 100))
    top.shapes(li).insert(kdb.Box(0, 0, 1000, 200))
    top.shapes(extra).insert(kdb.Box(900, 150, 1000, 200))
    for x in (0, 300, 600):
        top.insert(kdb.CellInstArray(sub.cell_index(), kdb.Trans(x, 50)))
    gds_path = tmp_path / "synthetic.gds"
    ly.write(str(gds_path))

    manifest = {
        "schema": "OpenLithoHub.scale-fixture.v1",
        "design": "testdesign",
        "gds_sha256": _sha256(gds_path),
        "gds_bytes": gds_path.stat().st_size,
        "top_cell": "test_top",
        "dbu_nm": 1.0,
        "bbox_dbu": [0, 0, 1000, 200],
        "layers": ["66:44", "67:20"],
        "selected_layer": "66:44",
        "source_commit": "a" * 40,
        "source_repository": "example/synthetic",
        "pixel_nm": 1.0,
    }
    manifest_path = tmp_path / "fixture-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return gds_path, manifest_path


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_audit(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(AUDIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("klayout") is None,
    reason="KLayout Python API not installed",
)
def test_audit_tool_end_to_end_on_synthetic_gds(tmp_path: Path) -> None:
    gds_path, manifest_path = _build_synthetic_fixture(tmp_path)
    out = tmp_path / "audit.json"
    proc = _run_audit(
        "--gds",
        str(gds_path),
        "--manifest",
        str(manifest_path),
        "--design",
        "testdesign",
        "--selected-layer",
        "66:44",
        "--out",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    assert "LAYER AUDIT: PASS" in proc.stdout
    evidence = json.loads(out.read_text(encoding="utf-8"))
    verdict = evidence["selected_layer_audit"]
    assert evidence["schema"] == "OpenLithoHub.scale-fixture-layer-audit.v1"
    assert verdict["shape_elements"] == 3
    assert verdict["flattened_instances"] == 7
    assert verdict["density_rank_by_flattened_instances"] == 1
    assert verdict["non_empty_layers"] == 2
    assert verdict["coverage_x_pct"] == 100.0
    # the per-layer checkpoint must be cleaned up on success
    assert not list(tmp_path.glob("*.checkpoint.json"))


@pytest.mark.skipif(
    importlib.util.find_spec("klayout") is None,
    reason="KLayout Python API not installed",
)
def test_audit_tool_rejects_stale_gds_bytes(tmp_path: Path) -> None:
    """The machine-hygiene mechanism, CI-testable: a GDS whose bytes no
    longer match the manifest can never be audited into authority."""
    gds_path, manifest_path = _build_synthetic_fixture(tmp_path)
    stale = tmp_path / "stale.gds"
    data = bytearray(gds_path.read_bytes())
    data[-1] ^= 0xFF
    stale.write_bytes(bytes(data))
    proc = _run_audit(
        "--gds",
        str(stale),
        "--manifest",
        str(manifest_path),
        "--design",
        "testdesign",
        "--selected-layer",
        "66:44",
        "--out",
        str(tmp_path / "audit.json"),
    )
    assert proc.returncode == 1
    assert "no authority" in proc.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("klayout") is None,
    reason="KLayout Python API not installed",
)
def test_audit_tool_never_picks_a_layer(tmp_path: Path) -> None:
    gds_path, manifest_path = _build_synthetic_fixture(tmp_path)
    proc = _run_audit(
        "--gds",
        str(gds_path),
        "--manifest",
        str(manifest_path),
        "--design",
        "testdesign",
        "--selected-layer",
        "67:20",
        "--out",
        str(tmp_path / "audit.json"),
    )
    assert proc.returncode == 1
    assert "never picks a layer" in proc.stderr


def test_audit_script_is_importable_and_fail_closed_helper_exists() -> None:
    spec = importlib.util.spec_from_file_location(AUDIT_MODULE_NAME, AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.AUDIT_SCHEMA == "OpenLithoHub.scale-fixture-layer-audit.v1"
    with pytest.raises(SystemExit):
        module._fail("boom")
