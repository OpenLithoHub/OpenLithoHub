"""Industrial Scale artifact verifier (scale track, S10).

Closes the 8-member scale family: strict JSON, known schemas, exact
family membership (fail-closed on unknown files), SHA256SUMS closure,
manifest closure, run-identity recomputation, fixture-manifest
revalidation, environment-lock completeness, per-row status/timing/
correctness rules and Lane-B worker-topology closure.

Two strictness tiers:

* default (structural closure): a complete, internally consistent
  family PASSes — this is the CPU development dry-run tier;
* ``--require-formal``: additionally refuses provisional runs, non-cuda
  backends, dirty tracked trees and sub-formal repeat counts — the
  tier required before anything may be promoted or published.

Usage::

    python scripts/verify_industrial_scale_artifacts.py --root <family-dir>
    python scripts/verify_industrial_scale_artifacts.py --root <dir> --require-formal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openlithohub.benchmark.industrial_scale import (  # noqa: E402
    RUN_CONFIG_SCHEMA,
    SCALE_CANONICAL_FAMILY,
    SCHEMA_NAME,
    ScaleRunConfig,
    ScaleStatus,
    compute_scale_run_identity,
    load_scale_fixture_manifest,
)

KNOWN_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
NANO_RE = re.compile(r"(?<!\")\b(NaN|Infinity|-Infinity)\b(?!\")")
STATUS_VALUES = {status.value for status in ScaleStatus}
FORMAL_MIN_REPEATS = 5


class VerifyError(Exception):
    pass


def _fail(message: str) -> None:
    raise VerifyError(message)


def _load_strict_json(path: Path) -> dict:
    text = path.read_text()
    match = NANO_RE.search(text)
    if match:
        _fail(f"{path.name}: non-strict JSON token {match.group(1)!r}")
    data = json.loads(text)
    if not isinstance(data, dict):
        _fail(f"{path.name}: top-level JSON must be an object")
    return data


def verify(root: Path, *, require_formal: bool = False) -> list[str]:
    """Run every closure check; returns the verified lane summary.  Raises
    :class:`VerifyError` on the first hard failure."""
    if not root.is_dir():
        _fail(f"family root missing: {root}")

    present = {p.name for p in root.iterdir() if p.is_file()}
    unknown = sorted(present - SCALE_CANONICAL_FAMILY)
    if unknown:
        _fail(f"unknown file(s) in family root (fail closed): {unknown}")
    missing = sorted(SCALE_CANONICAL_FAMILY - present)
    if missing:
        _fail(f"scale family incomplete, missing: {missing} (all-or-nothing)")

    # SHA256SUMS closure over the exact family
    sums: dict[str, str] = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        name = name.strip()
        if name.startswith("/") or ".." in name:
            _fail(f"SHA256SUMS contains unsafe path {name!r}")
        if name in sums:
            _fail(f"SHA256SUMS has duplicate entry for {name!r}")
        sums[name] = digest
    if set(sums) != (SCALE_CANONICAL_FAMILY - {"SHA256SUMS.txt"}):
        _fail("SHA256SUMS set != family member set (stale or incomplete)")
    for name, digest in sorted(sums.items()):
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != digest:
            _fail(f"SHA256 mismatch for {name!r}")
        if not KNOWN_HASH_RE.fullmatch(digest):
            _fail(f"malformed digest for {name!r}")

    # manifest closure (lists every member except itself and SHA256SUMS)
    manifest = _load_strict_json(root / "manifest.json")
    members = manifest.get("members")
    if not isinstance(members, list):
        _fail("manifest has no members list")
    expected_members = SCALE_CANONICAL_FAMILY - {"manifest.json", "SHA256SUMS.txt"}
    if {entry.get("name") for entry in members} != expected_members:
        _fail("manifest membership != family member set")
    for entry in members:
        if int(entry.get("bytes", -1)) != (root / entry["name"]).stat().st_size:
            _fail(f"manifest byte count mismatch for {entry['name']!r}")

    # run config member: schema, provisional flag, clean tree, identity
    run_config_member = _load_strict_json(root / "industrial-scale-run-config.json")
    if run_config_member.get("schema") != RUN_CONFIG_SCHEMA:
        _fail(f"run-config schema is not {RUN_CONFIG_SCHEMA!r}")
    payload = run_config_member.get("run_config")
    source_hashes = run_config_member.get("source_hashes")
    if not isinstance(payload, dict) or not isinstance(source_hashes, dict):
        _fail("run-config missing run_config/source_hashes")
    known_fields = set(ScaleRunConfig.__dataclass_fields__.keys())
    unknown_fields = set(payload) - known_fields
    if unknown_fields:
        _fail(f"run config carries unknown semantic fields: {sorted(unknown_fields)}")
    run_config = ScaleRunConfig(
        **{
            **payload,
            "lanes": tuple(payload.get("lanes", [])),
            "window_sizes": tuple(payload.get("window_sizes", [])),
        }
    )

    # environment lock from the freeze file participates in identity
    freeze = _load_strict_json(root / "industrial-scale-distribution-freeze.txt")
    env_lock = freeze.get("gpu") if isinstance(freeze.get("gpu"), dict) else None
    if env_lock is None:
        _fail("distribution freeze has no environment lock")
    from openlithohub.benchmark.industrial_scale import (
        canonical_json,
        scale_environment_lock_sha256,
    )

    env_lock_sha = hashlib.sha256(canonical_json(dict(env_lock)).encode()).hexdigest()
    if run_config_member.get("environment_lock_sha256") != env_lock_sha:
        _fail("run-config environment_lock_sha256 does not match the freeze")
    if env_lock_sha != scale_environment_lock_sha256(env_lock):
        _fail("environment lock hash is not canonical")

    recomputed = compute_scale_run_identity(
        run_config,
        measurement_commit=run_config_member["measurement_commit"],
        harness_sha256=source_hashes["harness"],
        core_sha256=source_hashes["core"],
        verifier_sha256=source_hashes["verifier"],
        claim_generator_sha256=source_hashes["claim_generator"],
        fixture_manifest_sha256=run_config.fixture_manifest_sha256,
        environment_lock_sha256=env_lock_sha,
    )
    recorded = run_config_member.get("run_identity") or manifest.get("run_identity")
    if recorded != recomputed:
        _fail(f"run identity drift: recorded {recorded!r} != recomputed {recomputed!r}")

    # fixture member: manifest revalidated, bytes bound to the run config
    fixture_member = _load_strict_json(root / "industrial-scale-fixture.json")
    if fixture_member.get("schema") != SCHEMA_NAME:
        _fail(f"fixture member schema is not {SCHEMA_NAME!r}")
    fixture_payload = fixture_member.get("fixture")
    if not isinstance(fixture_payload, dict):
        _fail("fixture member has no fixture manifest")
    from openlithohub.benchmark.industrial_scale import write_strict_json

    staging = root / "fixture-manifest.recheck.json"
    write_strict_json(staging, fixture_payload)
    try:
        revalidated = load_scale_fixture_manifest(staging)
    except ValueError as exc:
        _fail(f"fixture manifest fails revalidation: {exc}")
    finally:
        staging.unlink(missing_ok=True)
    if revalidated.gds_sha256 != run_config.fixture_gds_sha256:
        _fail("fixture manifest gds_sha256 != run config fixture_gds_sha256")
    if revalidated.selected_layer != run_config.selected_layer:
        _fail("fixture manifest selected_layer != run config selected_layer")
    if fixture_member.get("run_identity") not in (None, recomputed):
        _fail("fixture member bound to a different run identity")

    # lane rows: statuses, timing, correctness, repeat closure, topology
    lane_members = (
        ("industrial-scale-index.json", "A_LARGE_LAYOUT_STREAMING"),
        ("industrial-scale-runtime.json", "B_MULTI_GPU_SCALING"),
    )
    claims: list[str] = []
    for member_name, lane in lane_members:
        payload_member = _load_strict_json(root / member_name)
        if payload_member.get("lane") not in (None, lane):
            _fail(f"{member_name}: wrong lane {payload_member.get('lane')!r}")
        if payload_member.get("run_identity") not in (None, recomputed):
            _fail(f"{member_name}: bound to a different run identity")
        rows = payload_member.get("rows")
        if rows is None:
            continue  # lane not part of this run
        if not isinstance(rows, list):
            _fail(f"{member_name}: rows must be a list")
        for row in rows:
            status = row.get("status")
            if status not in STATUS_VALUES:
                _fail(f"{member_name}: unknown status {status!r}")
            if status == "SUCCESS":
                if not row.get("correctness_witness_pass"):
                    _fail(f"{member_name}: SUCCESS row failed the correctness witness")
                if int(row.get("aggregate_n") or 0) != int(row.get("repeat_count") or -1):
                    _fail(
                        f"{member_name}: SUCCESS row aggregate n != repeat_count "
                        "(stale/missing metric)"
                    )
                if row.get("forward_profile") == "P0_IDENTITY" and row.get("headline_eligible"):
                    _fail(f"{member_name}: P0_IDENTITY row must never be headline-eligible")
                timing = row.get("timing_method")
                if run_config.device_backend == "cuda" and timing != "cuda_synchronized":
                    _fail(f"{member_name}: cuda row timing is not synchronized")
            if lane == "B_MULTI_GPU_SCALING" and status == "SUCCESS":
                repeat_rows = row.get("per_repeat") or []
                for repeat in repeat_rows:
                    counts = repeat.get("worker_counts") or {}
                    if sum(counts.values()) != repeat.get("n_tiles"):
                        _fail(f"{member_name}: worker topology does not cover all tiles")
        if rows:
            claims.append(f"{lane}:{len(rows)} rows")

    if require_formal:
        if run_config_member.get("provisional"):
            _fail("provisional run cannot be promoted to a canonical family")
        if not run_config_member.get("tracked_tree_clean"):
            _fail("run-config records a dirty tracked tree (B2-A)")
        if run_config.device_backend != "cuda":
            _fail("formal scale runs require the cuda backend")
        if not env_lock.get("available"):
            _fail("FORMAL_SCALE_BLOCKED: CUDA measurement environment unavailable")
        if int(env_lock.get("count") or 0) < run_config.gpu_count:
            _fail(
                f"environment has {env_lock.get('count')} GPU(s); "
                f"run config requests {run_config.gpu_count}"
            )
        for member_name, _ in lane_members:
            payload_member = _load_strict_json(root / member_name)
            for row in payload_member.get("rows", []):
                if (
                    row.get("status") == "SUCCESS"
                    and int(row.get("repeat_count") or 0) < FORMAL_MIN_REPEATS
                ):
                    _fail(
                        f"{member_name}: SUCCESS row has {row.get('repeat_count')} repeats "
                        f"< {FORMAL_MIN_REPEATS}"
                    )

    return claims


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="scale family root directory")
    parser.add_argument(
        "--require-formal",
        action="store_true",
        help="additionally refuse provisional/cpu-backed/sub-formal families",
    )
    args = parser.parse_args()
    try:
        claims = verify(Path(args.root), require_formal=args.require_formal)
    except VerifyError as exc:
        print(f"SCALE VERIFIER: FAIL — {exc}", file=sys.stderr)
        return 1
    tier = "formal tier" if args.require_formal else "structural tier"
    print(f"SCALE VERIFIER: PASS — family closed ({tier}); {claims or 'no lane rows'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
