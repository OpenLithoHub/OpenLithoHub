#!/usr/bin/env python3
"""Pre-rerun preflight for the Industrial Benchmark (audit Phase A).

Runs every cheap check that must PASS before a formal long measurement
is allowed to start.  Crucially (audit B0.3) the computed identity uses
THE HARNESS'S OWN argument parser and ``compute_run_identity_from_args``
— the exact function the formal run uses — so a passing preflight proves
the identity of the prospective run, not an approximation.

Output on success: ``PRE-RERUN PREFLIGHT: PASS`` (exit 0) plus the
64-hex ``RUN_IDENTITY`` that the formal measurement will use.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load the harness by file path (same mechanism as the authority scripts)
# and reuse its parser + identity function verbatim.
_spec = importlib.util.spec_from_file_location(
    "olh_industrial_harness", REPO_ROOT / "benchmarks/industrial/run_industrial_benchmark.py"
)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = harness
_spec.loader.exec_module(harness)

import run_support as rs  # noqa: E402 - sibling module of the harness


def main() -> int:
    # Same parser, same flags, same defaults as the formal harness — the
    # ONLY difference is that we stop after the identity computation.
    harness_parser = harness.build_arg_parser()
    harness_parser.description = (
        "Pre-rerun preflight: verify every gate and print the EXACT run "
        "identity the formal measurement will use."
    )
    harness_parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="report the dirty-tree failure without exiting non-zero (NEVER "
        "valid for a formal measurement)",
    )
    args = harness_parser.parse_args()
    checks: list[tuple[str, bool, str]] = []

    source = rs.measurement_source(REPO_ROOT)
    checks.append(
        (
            "measurement commit is full 40-hex",
            source["commit_valid"] is True,
            f"commit={source['commit']}",
        )
    )
    tree_clean = source["working_tree_dirty"] is False
    checks.append(
        (
            "working tree clean (tracked files)",
            tree_clean,
            (
                "formal runs refuse a dirty tree; uncommitted changes poison the "
                "measurement identity"
                if not tree_clean
                else ""
            ),
        )
    )
    if not tree_clean and not args.allow_dirty:
        for name, ok, detail in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        print("PRE-RERUN PREFLIGHT: FAIL (dirty tree; see above)", file=sys.stderr)
        return 1

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

    gds_ok = args.gds is not None and Path(args.gds).exists()
    checks.append(("parent GDS exists", gds_ok, str(args.gds)))
    layout_ok = False
    layout_detail = ""
    if gds_ok:
        try:
            rs.validate_parent_layout(
                Path(args.gds),
                expected_top_cell="ibex_core",
                layer="66:44",
                expected_dbu_nm=1.0,
            )
            layout_ok = True
            layout_detail = "top cell / DBU / layer validated"
        except Exception as e:  # noqa: BLE001 - surfaced as a failed check
            layout_detail = str(e)
    checks.append(("layout fail-closed validation", layout_ok, layout_detail))

    identity_ok = False
    identity_detail = ""
    identity = ""
    if gds_ok:
        try:
            args.measurement_source = source
            args.parent_gds_sha256 = rs.sha256_file(Path(args.gds))
            # The EXACT harness identity function on the EXACT parsed args.
            identity, run_config, _freeze = harness.compute_run_identity_from_args(args, source)
            identity_ok = len(identity) == 64 and run_config["run_identity"] == identity
            identity_detail = (
                f"identity={identity} "
                f"env_lock={run_config['environment_lock']['lock_sha256'][:16]}... "
                f"freeze_sha={run_config['environment_lock']['distribution_freeze_sha256'][:16]}..."
            )
            runs_dir = Path(args.out) / "runs"
            if runs_dir.exists():
                others = [d.name for d in runs_dir.iterdir() if d.is_dir() and d.name != identity]
                if others:
                    identity_detail += (
                        f"; NOTE: {len(others)} prior run workspace(s) present — "
                        "the new identity gets a fresh workspace (stale reuse impossible)"
                    )
        except Exception as e:  # noqa: BLE001 - surfaced as a failed check
            identity_detail = str(e)
    checks.append(
        (
            "formal run identity computable via harness identity function",
            identity_ok,
            identity_detail,
        )
    )

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if failed:
        print(f"PRE-RERUN PREFLIGHT: FAIL ({len(failed)} check(s) failed)", file=sys.stderr)
        return 1
    print("PRE-RERUN PREFLIGHT: PASS")
    print(f"RUN_IDENTITY: {identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
