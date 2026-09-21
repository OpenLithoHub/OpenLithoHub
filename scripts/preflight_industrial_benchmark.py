#!/usr/bin/env python3
"""Pre-rerun preflight for the Industrial Benchmark (audit Phase A).

Runs every cheap check that must PASS before a formal long measurement
is allowed to start.  The harness itself re-checks the critical ones, so
this gate cannot be bypassed by forgetting the script — the script exists
so an operator can verify a machine + tree BEFORE committing hours.

Checks:

- measurement source: full 40-hex commit, clean tracked tree
- source-hash closure inputs present (harness/core/generator/support)
- parent GDS exists and passes fail-closed layout validation
  (top cell, DBU arithmetic, layer presence)
- environment lock derivable (packages, threads, hardware)
- run identity computable and stable across two computations
- run workspace is fresh or resumable under the SAME identity

Output on success: ``PRE-RERUN PREFLIGHT: PASS`` (exit 0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "industrial"))

import run_support as rs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "benchmarks" / "industrial" / "run_industrial_benchmark.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gds", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/industrial"))
    ap.add_argument("--iccad16-dir", type=Path, default=None)
    args = ap.parse_args()
    checks: list[tuple[str, bool, str]] = []

    source = rs.measurement_source(REPO_ROOT)
    checks.append(
        (
            "measurement commit is full 40-hex",
            source["commit_valid"] is True,
            f"commit={source['commit']}",
        )
    )
    checks.append(
        (
            "working tree clean (tracked files)",
            source["working_tree_dirty"] is False,
            "uncommitted tracked changes would poison the measurement identity",
        )
    )
    missing_hashes = [
        k
        for k in (
            "harness_sha256",
            "industrial_core_sha256",
            "claim_generator_sha256",
            "run_support_sha256",
        )
        if not source.get(k)
    ]
    checks.append(
        (
            "source-hash closure inputs present",
            not missing_hashes,
            f"missing: {missing_hashes}",
        )
    )

    gds_ok = args.gds.exists()
    checks.append(("parent GDS exists", gds_ok, str(args.gds)))
    layout_ok = False
    layout_detail = ""
    if gds_ok:
        try:
            rs.validate_parent_layout(
                args.gds,
                expected_top_cell="ibex_core",
                layer="66:44",
                expected_dbu_nm=1.0,
            )
            layout_ok = True
            layout_detail = "top cell / DBU / layer validated"
        except Exception as e:  # noqa: BLE001 - surfaced as a failed check
            layout_detail = str(e)
    checks.append(("layout fail-closed validation", layout_ok, layout_detail))

    env_ok = False
    env_lock: dict = {}
    env_detail = ""
    try:
        env_lock = rs.environment_lock(REPO_ROOT)
        env_ok = bool(env_lock.get("lock_sha256"))
        env_detail = f"lock_sha256={env_lock.get('lock_sha256', '')[:16]}..."
    except Exception as e:  # noqa: BLE001
        env_detail = str(e)
    checks.append(("environment lock derivable", env_ok, env_detail))

    identity_ok = False
    identity_detail = ""
    if gds_ok and env_ok:

        iccad_hashes = {}
        if args.iccad16_dir:
            for rel in ("testcase1.oas", "test1.csv"):
                p = args.iccad16_dir / rel
                if p.exists():
                    iccad_hashes[rel] = rs.sha256_file(p)
        payload = {
            "source": source,
            "environment_lock_sha256": env_lock["lock_sha256"],
            "parent_gds_sha256": rs.sha256_file(args.gds),
            "iccad_fixture_hashes": iccad_hashes,
            "args_payload": {"preflight": True, "layer": "66:44", "seed": 0},
        }
        identity_a = rs.compute_run_identity(**payload)
        identity_b = rs.compute_run_identity(**payload)
        identity_ok = identity_a == identity_b and len(identity_a) == 64
        identity_detail = f"identity={identity_a[:16]}... (stable={identity_ok})"
        runs_dir = args.out / "runs"
        if runs_dir.exists():
            others = [d.name for d in runs_dir.iterdir() if d.is_dir() and d.name != identity_a]
            if others:
                identity_detail += (
                    f"; NOTE: {len(others)} prior run workspace(s) present — new "
                    "identity gets a fresh workspace (stale reuse impossible)"
                )
    checks.append(("run identity computable + stable", identity_ok, identity_detail))

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if failed:
        print(f"PRE-RERUN PREFLIGHT: FAIL ({len(failed)} check(s) failed)", file=sys.stderr)
        return 1
    print("PRE-RERUN PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
