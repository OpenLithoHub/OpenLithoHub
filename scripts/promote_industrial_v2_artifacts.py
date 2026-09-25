"""Fail-closed operator CLI for Industrial Benchmark v2 canonical promotion.

The runbook rule is: promote only through ``promote_canonical_family()``
— never ``cp``, never an ad-hoc Python snippet.  This script IS the
supported operator path (2B.2-E)::

    python scripts/promote_industrial_v2_artifacts.py \\
        --workspace benchmarks/results/industrial-v2/runs/<RUN_ID> \\
        --canonical-root <staging-root>

It promotes NOTHING unless every gate closes:

* the workspace is a formal v2 run workspace (run-config, environment
  lock, all three tier rows) holding the exact complete seven-member
  canonical family, internally closed by the v2 verifier (strict JSON,
  SHA256SUMS, manifest, run-identity recomputation);
* the formal blockers re-run clean against the WORKSPACE's own recorded
  facts AND the live repository state: CUDA environment lock present,
  every tier SUCCESS with its correctness witness, formal (never
  provisional) run, tracked tree clean now as well as at run time;
* the canonical root is not the frozen v1.1 root nor inside it — v1.1 is
  never mutated;
* the canonical root is nonexistent/empty, or already holds THIS run's
  byte-identical family (idempotent re-promotion).  An unrelated or
  drifted canonical authority is never silently overwritten.

Exit code 0 = promoted and re-verified; nonzero = refused, nothing
written to the canonical root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openlithohub.benchmark.industrial_v2 import (  # noqa: E402
    CANONICAL_FAMILY,
    formal_publication_blockers,
    promote_canonical_family,
)

WORKSPACE_REQUIRED_FILES = (
    "run-config.json",
    "environment-lock.json",
    "tier-a.json",
    "tier-b.json",
    "tier-c.json",
)
TIER_KEYS = ("a", "b", "c")


def _load_verifier() -> Any:
    """Load the v2 artifact verifier module from its repo path."""
    import importlib.util

    verifier_path = REPO / "scripts" / "verify_industrial_v2_artifacts.py"
    spec = importlib.util.spec_from_file_location("verify_industrial_v2_artifacts", verifier_path)
    if spec is None or spec.loader is None:  # pragma: no cover — repo layout invariant
        raise RuntimeError(f"cannot load v2 verifier from {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tracked_tree_clean(repo: Path) -> bool:
    """Live git cleanliness of the measurement repo (fail-closed: git
    unavailability counts as dirty)."""
    import subprocess

    git = shutil.which("git")
    if git is None:
        return False
    try:
        out = subprocess.run(  # noqa: S603 — fixed-argv git query
            [git, "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.stdout.strip() == ""


def _workspace_run_config(workspace: Path) -> dict[str, Any]:
    return json.loads((workspace / "run-config.json").read_text())  # type: ignore[no-any-return]


def _workspace_tier_rows(workspace: Path) -> dict[str, dict[str, Any]]:
    return {tier: json.loads((workspace / f"tier-{tier}.json").read_text()) for tier in TIER_KEYS}


def _frozen_v11_blockers(canonical_root: Path, workspace: Path, repo: Path) -> list[str]:
    """v1.1 is frozen: neither the canonical root nor the workspace may be
    the v1.1 results root or live inside it."""
    v11_root = (repo / "benchmarks" / "results" / "industrial").resolve()
    blockers: list[str] = []
    for label, path in (("canonical root", Path(canonical_root)), ("workspace", Path(workspace))):
        resolved = Path(path).resolve()
        if resolved == v11_root or v11_root in resolved.parents:
            blockers.append(
                f"{label} {path} is inside the frozen v1.1 root {v11_root} "
                "(v1.1 artifacts are never mutated)"
            )
    return blockers


def _target_blockers(canonical_root: Path, workspace: Path) -> list[str]:
    """Never silently overwrite an unrelated or drifted canonical
    authority: the target must be absent, empty, or hold THIS run's
    byte-identical family (SHA256SUMS equality covers every member)."""
    target = Path(canonical_root)
    if not target.exists():
        return []
    present = {p.name for p in target.iterdir() if p.is_file()}
    if not present:
        return []
    workspace_sums = Path(workspace) / "SHA256SUMS.txt"
    if not workspace_sums.is_file():
        # incomplete-family blockers are already recorded; nothing to compare
        return []
    if present != set(CANONICAL_FAMILY):
        unexpected = sorted(present - set(CANONICAL_FAMILY)) or "partial family"
        return [
            f"canonical root {target} already holds files that are not exactly the "
            f"seven-member canonical family: {unexpected} — refusing to overwrite"
        ]
    existing_sums = (target / "SHA256SUMS.txt").read_bytes()
    staged_sums = workspace_sums.read_bytes()
    if existing_sums != staged_sums:
        return [
            f"canonical root {target} already holds a DIFFERENT (or drifted) canonical "
            f"authority — refusing to overwrite; choose a fresh canonical root or "
            "retire the old authority explicitly"
        ]
    return []


def promotion_blockers(
    workspace: str | Path,
    canonical_root: str | Path,
    repo: Path,
    *,
    tree_clean: bool | None = None,
    verifier: Any = None,
) -> list[str]:
    """Every reason the promotion must be refused — empty list is the ONLY
    license to promote.  Accumulates ALL blockers (operator sees the full
    refusal reasons, not just the first)."""
    workspace = Path(workspace)
    canonical_root = Path(canonical_root)
    blockers: list[str] = []

    missing_ws = sorted(
        name for name in WORKSPACE_REQUIRED_FILES if not (workspace / name).is_file()
    )
    if missing_ws:
        return [
            f"workspace missing required file(s) {missing_ws} — "
            "not a formal Industrial Benchmark v2 run workspace"
        ]
    run_config = _workspace_run_config(workspace)
    identity = str(run_config.get("run_identity") or "")
    if not identity:
        return ["workspace run-config.json carries no run_identity"]
    if not run_config.get("tracked_tree_clean"):
        blockers.append(
            "workspace run-config records a dirty tracked tree at measurement time (B2-A)"
        )

    # stage the exact seven-member family and close it with the verifier:
    # a partial/stale/unprovable family never reaches the canonical root.
    staging = Path(tempfile.mkdtemp(prefix="v2-promotion-stage-"))
    try:
        missing_members = [
            member for member in sorted(CANONICAL_FAMILY) if not (workspace / member).is_file()
        ]
        if missing_members:
            blockers.append(
                f"workspace has no built canonical family, missing {missing_members} — "
                "formal runs must build it through build_canonical_family_in_workspace (2B.1-J)"
            )
        else:
            for member in sorted(CANONICAL_FAMILY):
                shutil.copy2(workspace / member, staging / member)
            if verifier is None:
                verifier = _load_verifier()
            try:
                verifier.verify(staging)
            except verifier.VerifyError as exc:
                blockers.append(f"workspace canonical family does not close: {exc}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    blockers += _frozen_v11_blockers(canonical_root, workspace, repo)
    blockers += _target_blockers(canonical_root, workspace)

    if tree_clean is None:
        tree_clean = _tracked_tree_clean(repo)
    if not tree_clean:
        blockers.append(
            "current tracked tree is dirty — promotion re-runs the formal blockers (B2-A)"
        )
    blockers += formal_publication_blockers(
        env_lock=json.loads((workspace / "environment-lock.json").read_text()),
        tier_rows=_workspace_tier_rows(workspace),
        git_clean=bool(tree_clean),
        provisional=bool(run_config.get("provisional")),
    )
    return blockers


def promote(
    workspace: str | Path,
    canonical_root: str | Path,
    repo: Path,
    *,
    tree_clean: bool | None = None,
    verifier: Any = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Refuse-or-promote.  Returns (promoted, blockers, info); on refusal
    the canonical root is untouched."""
    workspace_path = Path(workspace)
    blockers = promotion_blockers(
        workspace_path, canonical_root, repo, tree_clean=tree_clean, verifier=verifier
    )
    if blockers:
        return False, blockers, {}

    run_config = _workspace_run_config(workspace_path)
    clean = _tracked_tree_clean(repo) if tree_clean is None else tree_clean
    leftover = promote_canonical_family(
        workspace_dir=workspace_path,
        canonical_root=Path(canonical_root),
        env_lock=json.loads((workspace_path / "environment-lock.json").read_text()),
        tier_rows=_workspace_tier_rows(workspace_path),
        git_clean=clean,
        provisional=bool(run_config.get("provisional")),
    )
    if leftover:
        return False, leftover, {}

    if verifier is None:
        verifier = _load_verifier()
    try:
        claims = verifier.verify(Path(canonical_root))
    except verifier.VerifyError as exc:  # pragma: no cover — belt-and-braces closure
        return False, [f"promoted canonical root failed verification: {exc}"], {}
    return (
        True,
        [],
        {
            "run_identity": run_config.get("run_identity"),
            "measurement_commit": run_config.get("measurement_commit"),
            "claims": claims,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="formal run workspace (…/runs/<run_identity>/) holding the built canonical family",
    )
    parser.add_argument(
        "--canonical-root",
        required=True,
        help="canonical v2 family root to create or idempotently refresh (never v1.1)",
    )
    args = parser.parse_args()

    try:
        promoted, blockers, info = promote(Path(args.workspace), Path(args.canonical_root), REPO)
    except Exception as exc:  # noqa: BLE001 — operator-facing CLI, fail closed
        print(f"V2 PROMOTION: FAIL — {exc}", file=sys.stderr)
        return 1
    if not promoted:
        print("V2 PROMOTION: REFUSED — nothing was promoted:", file=sys.stderr)
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 1
    print(f"V2 PROMOTION: PASS — run identity {info['run_identity']}")
    print(f"  measurement commit: {info['measurement_commit']}")
    print(f"  canonical root: {Path(args.canonical_root).resolve()}")
    print(f"  claims: {info['claims'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
