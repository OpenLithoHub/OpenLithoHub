"""Scale gate semantics tests (PR-V3.1.3).

The 2026-10-05 review found that one and the same tracked-tree gate is
a live tripwire for Microwatt (committed manifests) and structurally
blind for Ibex (fully gitignored fixture), and that nobody had declared
this asymmetry — it was implied by `.gitignore` and documented nowhere.
These tests make the semantics a CI-enforced contract:

* reality must match the declared per-design fixture tracking
  (`git check-ignore` / `git ls-files` against the authority table);
* the preflight's tracked-tree STOP text must explain itself (gate
  reach per design, reproducibility-verification meaning, no
  `git checkout --`) and must render the declared table, so the
  message cannot drift from the authority source;
* the operator checklist for the three script-blind-spot classes must
  exist in the GPU runbook.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

from openlithohub.benchmark.scale_parameter_authority import (
    SCALE_PARAMETER_AUTHORITY,
)

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "docs/industrial-scale-gpu-runbook.md"
PREFLIGHT_SCRIPT = REPO / "scripts" / "preflight_industrial_scale.py"
AUDIT_EVIDENCE = REPO / "benchmarks/results/industrial-scale/audits/microwatt-layer-audit.json"


def _norm(text: str) -> str:
    return " ".join(text.split())


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60
    )


# ---- declared tracking vs .gitignore reality -------------------------------------


def test_declared_tracked_paths_are_tracked_and_not_ignored() -> None:
    for design, entry in SCALE_PARAMETER_AUTHORITY["designs"].items():
        for path in entry["fixture_tracking"]["tracked_paths"]:
            ignored = _git("check-ignore", "-q", "--", path)
            assert ignored.returncode == 1, (
                f"{design}: declared tracked path {path} is gitignored — "
                "the authority table and .gitignore have drifted"
            )
            listed = _git("ls-files", "--error-unmatch", "--", path)
            assert listed.returncode == 0, (
                f"{design}: declared tracked path {path} is not in the index"
            )


def test_declared_host_local_paths_are_ignored() -> None:
    for design, entry in SCALE_PARAMETER_AUTHORITY["designs"].items():
        for path in entry["fixture_tracking"]["host_local_paths"]:
            ignored = _git("check-ignore", "-q", "--", path)
            assert ignored.returncode == 0, (
                f"{design}: declared host-local path {path} is not gitignored — "
                "fixture bytes must never silently enter the tracked tree "
                "(or leave it), the gate semantics depend on this"
            )


def test_ibex_fixture_is_invisible_to_the_tracked_tree_gate() -> None:
    """The exact asymmetry the review flagged: the whole ibex fixture
    (manifest included) is host-local, so 'tracked tree clean' proves
    nothing about it — SHA-256 revalidation is the only gate."""
    tracking = SCALE_PARAMETER_AUTHORITY["designs"]["ibex"]["fixture_tracking"]
    assert tracking["tracked_paths"] == []
    ibex_manifest = "benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json"
    assert _git("check-ignore", "-q", "--", ibex_manifest).returncode == 0


def test_microwatt_manifests_are_a_live_tripwire() -> None:
    tracking = SCALE_PARAMETER_AUTHORITY["designs"]["microwatt"]["fixture_tracking"]
    assert len(tracking["tracked_paths"]) == 2
    for path in tracking["tracked_paths"]:
        listed = _git("ls-files", "--error-unmatch", "--", path)
        assert listed.returncode == 0, f"declared tripwire path not tracked: {path}"
    gds = "benchmarks/results/industrial-scale/fixtures/microwatt/microwatt.gds"
    assert _git("check-ignore", "-q", "--", gds).returncode == 0


def test_layer_audit_evidence_is_tracked() -> None:
    audit_rel = "benchmarks/results/industrial-scale/audits/microwatt-layer-audit.json"
    assert _git("check-ignore", "-q", "--", audit_rel).returncode == 1


# ---- the preflight STOP text is part of the protocol surface ----------------------

PREFLIGHT_MODULE = "preflight_industrial_scale"


def _load_preflight():
    spec = importlib.util.spec_from_file_location(PREFLIGHT_MODULE, PREFLIGHT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_stop_text_explains_the_gate() -> None:
    module = _load_preflight()
    text = _norm(module.tracked_tree_gate_explanation())
    # what it is: reproducibility verification, never a tool bug
    assert "REPRODUCIBILITY" in text
    assert "not a tool bug" in text.lower()
    # what it watches vs cannot see
    assert "TRACKED files only" in text
    assert "INVISIBLE" in text
    # the runtime remedy rule
    assert "git checkout --" in text
    assert "STOP" in text
    # per-design reach, rendered from the declared table (cannot drift)
    for design, entry in SCALE_PARAMETER_AUTHORITY["designs"].items():
        assert design in text
        for path in entry["fixture_tracking"]["tracked_paths"] or ["(none"]:
            assert path in text


def test_preflight_stop_text_covers_both_designs_symmetrically() -> None:
    module = _load_preflight()
    text = module.tracked_tree_gate_explanation()
    assert "ibex: tracked = (none" in text
    assert "microwatt: tracked =" in text
    assert "SHA-256" in text


# ---- the runbook carries the script-blind-spot checklist --------------------------


def test_runbook_has_the_operator_checklist() -> None:
    runbook = _norm(RUNBOOK.read_text(encoding="utf-8"))
    assert "## 9. Operator checklist" in runbook
    for blind_spot in (
        "what the scripts cannot check",
        "Resource margins, per design",
        "Machine hygiene",
    ):
        assert blind_spot in runbook, f"checklist blind spot missing: {blind_spot}"


def test_runbook_states_the_gate_asymmetry_and_stop_semantics() -> None:
    runbook = _norm(RUNBOOK.read_text(encoding="utf-8"))
    assert "gate sees NOTHING" in runbook
    assert "live tripwire" in runbook
    assert "REPRODUCIBILITY-VERIFICATION failure" in runbook
    assert "git checkout --" in runbook


def test_runbook_machine_hygiene_names_known_stale_artifacts() -> None:
    runbook = _norm(RUNBOOK.read_text(encoding="utf-8"))
    assert "144,566-byte" in runbook
    assert "9b1790b9" in runbook
    assert "pdb-standin" in runbook
