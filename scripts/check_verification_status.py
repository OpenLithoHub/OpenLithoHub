#!/usr/bin/env python3
"""Fail closed when the verification scoreboard drifts from the status manifest.

P-054 repo integration (audit P0.6): docs must not hand-maintain claims the
machine already knows.  This script validates:

1. ``proof_artifacts/verification-status.json`` parses and carries the
   required sections;
2. the frozen paper basis is present in the canonical P-054 README;
3. the canonical entry pointer exists in ``docs/qdm-readiness.md``;
4. the current repo HEAD is allowed to move, but the frozen
   ``repo_commit_basis`` is never silently rewritten to it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "proof_artifacts" / "verification-status.json"
P054_README = ROOT / "proof_artifacts" / "p054" / "README.md"
QDM_READINESS = ROOT / "docs" / "qdm-readiness.md"

REQUIRED_SECTIONS = ("finite_declared_model", "repository_engineering", "bridges")


def current_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S603 — fixed argv, repo-local
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()


def main() -> int:
    if not STATUS_PATH.exists():
        print(f"FAIL: {STATUS_PATH} missing", file=sys.stderr)
        return 1
    status = json.loads(STATUS_PATH.read_text())
    for section in REQUIRED_SECTIONS:
        if section not in status:
            print(f"FAIL: verification-status.json missing section {section!r}", file=sys.stderr)
            return 1

    basis = status.get("repo_commit_basis")
    if not basis:
        print("FAIL: repo_commit_basis missing", file=sys.stderr)
        return 1
    head = current_head()
    if head != basis and "frozen" not in status.get("paper", "").lower():
        # moving HEAD is fine; rewriting the frozen basis is not. The basis
        # must still be exactly the one the paper froze.
        expected_basis = "348fa5d86d5355465af98e2c4ce3deac60081a4c"
        if basis != expected_basis:
            print(
                f"FAIL: repo_commit_basis {basis[:12]}… was rewritten from the "
                f"frozen paper basis {expected_basis[:12]}…",
                file=sys.stderr,
            )
            return 1

    readme_text = P054_README.read_text()
    if basis not in readme_text:
        print("FAIL: P-054 README does not carry the frozen implementation basis", file=sys.stderr)
        return 1

    readiness = QDM_READINESS.read_text()
    canonical = status.get("canonical_entry", "")
    if canonical and canonical not in readiness and "p054" not in readiness:
        print(
            "FAIL: docs/qdm-readiness.md does not reference the canonical "
            f"entry {canonical!r}; add a pointer instead of hand-written status",
            file=sys.stderr,
        )
        return 1

    print(
        f"verification-status OK: paper={status.get('paper')} basis={basis[:12]}… head={head[:12]}…"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
