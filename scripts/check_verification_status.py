#!/usr/bin/env python3
"""Fail closed when the verification state drifts from the frozen basis.

P-054 repo integration (re-audit PR-3B): the frozen basis is enforced
UNCONDITIONALLY — never relative to the moving repo HEAD.

Checks, in order:
1. ``verification-status.json`` parses and carries the required sections.
2. ``p054/frozen-basis.json`` is the immutable release anchor: its
   implementation_commit equals the audit-frozen P-054 basis, and its
   content hashes match the actual profile files (any catalog edit must
   regenerate the anchor consciously, in review).
3. ``status.repo_commit_basis == manifest.implementation_commit ==
   frozen-basis.implementation_commit`` — a basis rewritten to the
   current HEAD fails, as does any cross-file disagreement.
4. Activation gate: ``finite_declared_model.full_artifact_replay`` may not
   be a passing state while the registry artifact is UNFETCHED.

The moving HEAD is reported but never used as an input to a verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "proof_artifacts" / "verification-status.json"
RECEIPT_PATH = ROOT / "proof_artifacts" / "p054" / "replay-receipt.json"
REPLAY_ENGINE = ROOT / "src" / "openlithohub" / "verify" / "phase_diagram.py"
REGISTRY_PATH = ROOT / "proof_artifacts" / "registry.json"
P054 = ROOT / "proof_artifacts" / "p054"
P054_README = P054 / "README.md"
QDM_READINESS = ROOT / "docs" / "qdm-readiness.md"

# The immutable P-054 paper basis (re-audit §8: never rewritten to HEAD).
EXPECTED_P054_BASIS = "348fa5d86d5355465af98e2c4ce3deac60081a4c"
ANCHOR_FILES = (
    "manifest.json",
    "fixture.json",
    "event_catalog.json",
    "chamber_catalog.json",
)
REQUIRED_SECTIONS = ("finite_declared_model", "repository_engineering", "bridges")
# Contract B (re-audit PR-5B): the repository implements imported
# frozen-certificate verification; SOURCE_NATIVE_RECOMPUTED requires a
# pinned recomputation engine (Contract A) and backs FULL_REPLAY_PASSED.
# PR-3D: strict status/mode lattices.  Unknown values hard-fail.
_PASSING_REPLAY_STATES = {
    "IMPORTED_CERTIFICATE_VERIFIED",
    "SOURCE_NATIVE_RECOMPUTED",
    "FULL_REPLAY_PASSED",  # legacy alias; requires source-native strength
}
MODE_STRENGTH = {
    "STRUCTURE_ONLY": 0,
    "IMPORTED_CERTIFICATE_VERIFIED": 1,
    "SOURCE_NATIVE_RECOMPUTED": 2,
}
STATUS_REQUIRED_STRENGTH = {
    "BLOCKED_ARTIFACT_UNFETCHED": 0,
    "IMPORTED_CERTIFICATE_VERIFIED": 1,
    "SOURCE_NATIVE_RECOMPUTED": 2,
    "FULL_REPLAY_PASSED": 2,
}


def current_head(repo: Path = ROOT) -> str:
    return subprocess.run(  # noqa: S603 — fixed argv, repo-local
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo,
    ).stdout.strip()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_state(
    *,
    status: dict,
    manifest: dict,
    frozen_basis: dict,
    registry: dict,
    file_hashes: dict[str, str],
    readiness_text: str,
) -> list[str]:
    """Pure checker: returns a list of failures (empty == pass)."""
    failures: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section not in status:
            failures.append(f"verification-status.json missing section {section!r}")
    basis = status.get("repo_commit_basis")
    manifest_basis = manifest.get("implementation_commit")
    anchor_basis = frozen_basis.get("implementation_commit")
    # PR-3B: unconditional triple agreement with the audit-frozen constant.
    if basis != EXPECTED_P054_BASIS:
        failures.append(f"status repo_commit_basis {basis!r} is not the frozen P-054 basis")
    if manifest_basis != EXPECTED_P054_BASIS:
        failures.append(
            f"manifest implementation_commit {manifest_basis!r} is not the frozen P-054 basis"
        )
    if anchor_basis != EXPECTED_P054_BASIS:
        failures.append(
            f"frozen-basis implementation_commit {anchor_basis!r} is not the frozen P-054 basis"
        )
    if len({basis, manifest_basis, anchor_basis}) != 1:
        failures.append("basis disagreement across status / manifest / frozen-basis")

    # content-hash anchor over the profile files
    for name, expected in frozen_basis.get("content_hashes", {}).items():
        actual = file_hashes.get(name)
        if actual is None:
            failures.append(f"frozen-basis references missing file {name!r}")
        elif actual != expected:
            failures.append(
                f"{name} hash drifted from the frozen anchor ({actual[:12]}… "
                f"!= {expected[:12]}…); regenerate frozen-basis.json in review"
            )

    # activation gate (PR-3C/3D): a passing full_artifact_replay requires a
    # replay receipt whose identity is bound end-to-end; unknown status
    # values hard-fail instead of silently skipping the binding
    replay_state = status.get("finite_declared_model", {}).get("full_artifact_replay", "")
    if replay_state and replay_state not in STATUS_REQUIRED_STRENGTH:
        print(
            f"FAIL: unknown full_artifact_replay state {replay_state!r}; "
            f"allowed: {sorted(STATUS_REQUIRED_STRENGTH)}",
            file=sys.stderr,
        )
        return 1
    if replay_state in _PASSING_REPLAY_STATES:
        failures.extend(
            _verify_replay_binding(
                replay_state=replay_state,
                receipt_path=RECEIPT_PATH,
                registry=registry,
                frozen_basis=frozen_basis,
                manifest=manifest,
                engine_engine_hash=sha256_of(REPLAY_ENGINE) if REPLAY_ENGINE.exists() else None,
            )
        )
    elif replay_state and receipt_path_exists():
        failures.append(
            "a replay receipt exists but the status is not a passing state; "
            "promote the status in a reviewed release change instead"
        )

    if basis not in P054_README.read_text():
        failures.append("P-054 README does not carry the frozen implementation basis")
    canonical = status.get("canonical_entry", "")
    if canonical and canonical not in readiness_text and "p054" not in readiness_text:
        failures.append("docs/qdm-readiness.md does not reference the canonical entry")
    return failures


def receipt_path_exists() -> bool:
    return RECEIPT_PATH.exists()


def _verify_replay_binding(
    *,
    replay_state: str,
    receipt_path: Path,
    registry: dict,
    frozen_basis: dict,
    manifest: dict,
    engine_engine_hash: str | None,
) -> list[str]:
    """S1-S6: the receipt, registry, frozen basis and engine must agree."""
    failures: list[str] = []
    if not receipt_path.exists():
        return [
            f"full_artifact_replay={replay_state!r} but no replay receipt at "
            f"{receipt_path} (S1: a populated registry hash alone is not evidence)"
        ]
    receipt = json.loads(receipt_path.read_text())
    entry = next(
        (
            a
            for a in registry.get("artifacts", [])
            if "p054-full-replay" in a.get("required_for", [])
        ),
        {},
    )
    registry_sha = entry.get("sha256")
    registry_bytes = entry.get("bytes")
    basis_external = frozen_basis.get("external_artifact_sha256")
    receipt_sha = receipt.get("artifact_sha256")
    receipt_bytes = receipt.get("artifact_bytes")
    # S1/S2/S3: end-to-end hash binding
    if not registry_sha:
        failures.append("registry sha256 is null; a passing state is not possible (S1)")
    if receipt_sha != registry_sha:
        failures.append("receipt artifact hash != registry hash (S2)")
    if basis_external is None or receipt_sha != basis_external:
        failures.append("receipt artifact hash != frozen-basis external hash (S3)")
    # S4: replay engine drift
    if engine_engine_hash and receipt.get("replay_engine_sha256") != engine_engine_hash:
        failures.append("receipt replay engine hash drifted from the live engine (S4)")
    # S5 (PR-3D lattice): receipt mode strength must satisfy the claimed
    # status; unknown modes hard-fail
    mode = receipt.get("replay_mode")
    if mode not in MODE_STRENGTH:
        failures.append(f"unknown receipt replay mode {mode!r} (S5)")
    elif MODE_STRENGTH[mode] < STATUS_REQUIRED_STRENGTH.get(replay_state, 99):
        failures.append(
            f"receipt mode {mode!r} (strength {MODE_STRENGTH[mode]}) is too weak "
            f"for status {replay_state!r} "
            f"(required {STATUS_REQUIRED_STRENGTH.get(replay_state)})"
        )
    # S6: byte binding
    if registry_bytes is not None and receipt_bytes != registry_bytes:
        failures.append("receipt artifact bytes != registry bytes (S6)")
    # commit binding
    if receipt.get("implementation_commit") != manifest.get("implementation_commit"):
        failures.append("receipt implementation_commit != manifest basis")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-head", action="store_true", help="print moving HEAD (informational)")
    args = ap.parse_args()

    for required in (
        STATUS_PATH,
        P054 / "frozen-basis.json",
        REGISTRY_PATH,
        P054_README,
        QDM_READINESS,
    ):
        if not required.exists():
            print(f"FAIL: {required} missing", file=sys.stderr)
            return 1

    status = json.loads(STATUS_PATH.read_text())
    manifest = json.loads((P054 / "manifest.json").read_text())
    frozen_basis = json.loads((P054 / "frozen-basis.json").read_text())
    registry = json.loads(REGISTRY_PATH.read_text())
    file_hashes = {name: sha256_of(P054 / name) for name in ANCHOR_FILES}

    failures = verify_frozen_state(
        status=status,
        manifest=manifest,
        frozen_basis=frozen_basis,
        registry=registry,
        file_hashes=file_hashes,
        readiness_text=QDM_READINESS.read_text(),
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    head = current_head()
    if args.report_head:
        print(f"moving repo HEAD (informational): {head[:12]}…")
    print(
        "verification-status OK: "
        f"paper={status.get('paper')} frozen_basis={EXPECTED_P054_BASIS[:12]}… "
        f"replay_state={status.get('finite_declared_model', {}).get('full_artifact_replay')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
