#!/usr/bin/env python3
"""CI verifier for Industrial Benchmark artifacts (authority, not measurement).

Runs the cheap, deterministic checks that keep artifact authority intact:

1. Every ``*.json`` in the artifacts dir parses as STRICT JSON (NaN /
   Infinity tokens are rejected at parse time) and passes the
   ``OpenLithoHub.industrial-benchmark.v1`` structural validation,
   including full-hex commits, ``measurement_source`` closure (clean
   tree, 64-hex source hashes) and fixture SHA-256.
2. ``SHA256SUMS.txt`` matches the artifact bytes exactly.
3. The claims manifest references the same artifact hashes.

This script never runs the benchmark itself — CI verifies authority;
measurements come from the manual/release harness run.  Exits 0 with a
skip notice when the artifacts dir has no artifacts yet (e.g. the
baseline state before the first measurement lands).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def _load_industrial_authority():
    """Load the stdlib-only industrial module by file path.

    The authority tooling must run in environments WITHOUT torch (the
    lint job): importing ``openlithohub.benchmark.industrial`` normally
    executes the package ``__init__`` and drags in the heavy stack.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "src/openlithohub/benchmark/industrial.py"
    spec = importlib.util.spec_from_file_location("olh_industrial_authority", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve __module__ during exec
    spec.loader.exec_module(mod)
    return mod


_ind = _load_industrial_authority()
SCHEMA_NAME = _ind.SCHEMA_NAME
RUN_CONFIG_SCHEMA = _ind.RUN_CONFIG_SCHEMA
validate_artifact = _ind.validate_artifact
validate_run_config = _ind.validate_run_config
validate_environment_lock = _ind.validate_environment_lock
validate_artifact_family = _ind.validate_artifact_family
recompute_run_identity = _ind.recompute_run_identity
load_artifact = _ind.load_artifact
validate_artifact = _ind.validate_artifact
load_artifact = _ind.load_artifact

_SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def _reject_constant(token: str) -> float:
    raise ValueError(f"non-finite JSON constant {token!r}")


def _git_show_sha256(commit: str, path: str, repo_root: Path) -> str:
    """SHA-256 of a file AS COMMITTED (semantic source closure, audit G0.8)."""
    import subprocess

    out = subprocess.run(  # noqa: S603 - fixed-argv git probe only
        ["git", "show", f"{commit}:{path}"],
        cwd=str(repo_root),
        capture_output=True,
        check=True,
        timeout=30,
    )
    return hashlib.sha256(out.stdout).hexdigest()


def verify_family_closure(parsed: dict[str, dict]) -> list[str]:
    """Single family validator path (audit P0.1/P0.3/P0.4).

    All published artifacts must belong to ONE run; the run identity must
    be the RECOMPUTED content digest of the published run-config; every
    benchmark artifact's git_commit must equal its measurement source
    commit; and the environment lock hash must recompute.
    """
    problems: list[str] = []
    if not parsed:
        return problems
    roles = set(parsed)
    required = {"runtime", "quality", "fulldie", "run-config", "manifest"}
    missing = required - roles
    if missing:
        problems.append(f"incomplete published family: missing {sorted(missing)}")

    # P0.3: recompute the identity FROM the published run-config.
    if "run-config" in parsed:
        config = parsed["run-config"]
        problems.extend(f"run-config: {p}" for p in validate_run_config(config))
        recomputed = recompute_run_identity(config)
        for role, data in sorted(parsed.items()):
            claimed = str(data.get("run_identity"))
            if claimed != recomputed:
                problems.append(
                    f"{role}: run_identity {claimed} != recomputed {recomputed} "
                    "from the published run-config"
                )

    # P0.4: every benchmark artifact ties git_commit to its source commit.
    for role, data in sorted(parsed.items()):
        source_commit = str((data.get("measurement_source") or {}).get("commit") or "")
        git_commit = str(data.get("git_commit") or "")
        if source_commit and git_commit != source_commit:
            problems.append(
                f"{role}: git_commit {git_commit} != measurement_source.commit {source_commit}"
            )

    # Single family validator: the industrial module's own.
    problems.extend(f"family: {p}" for p in validate_artifact_family(parsed))
    return problems


def verify_source_closure(artifacts: list[tuple[Path, dict]], repo_root: Path) -> list[str]:
    """Each artifact's stored source hashes must match the committed bytes.

    ``git show <measurement commit>:<file>`` is re-hashed and compared to
    the artifact's recorded harness/core/generator/support hashes: the
    artifact really was produced by the committed source it names.
    """
    problems: list[str] = []
    checked: set[tuple[str, str, str]] = set()
    for path, data in artifacts:
        source = data.get("measurement_source") or {}
        commit = source.get("commit")
        if not isinstance(commit, str):
            continue
        for key, rel in (
            ("harness_sha256", "benchmarks/industrial/run_industrial_benchmark.py"),
            ("industrial_core_sha256", "src/openlithohub/benchmark/industrial.py"),
            ("claim_generator_sha256", "scripts/generate_industrial_claims.py"),
            ("run_support_sha256", "benchmarks/industrial/run_support.py"),
        ):
            stored = source.get(key)
            triple = (commit, rel, stored or "")
            if stored is None or triple in checked:
                continue
            checked.add(triple)
            try:
                actual = _git_show_sha256(commit, rel, repo_root)
            except Exception as e:  # noqa: BLE001 - surfaced as a failure
                problems.append(f"{path.name}: source closure git show failed: {e}")
                continue
            if actual != stored:
                problems.append(
                    f"{path.name}: source closure MISMATCH for {rel}: "
                    f"artifact {stored} != committed {actual} at {commit[:12]}"
                )
    return problems


def verify_claims_linkage(artifacts_dir: Path, claims_path: Path, sums_path: Path) -> list[str]:
    """Every claim's artifact reference must resolve to the exact bytes
    measured (audit G0.9): file exists, per-claim sha256 == actual ==
    SHA256SUMS entry."""
    import hashlib

    if not claims_path.exists():
        return []
    problems: list[str] = []
    try:
        with open(claims_path, encoding="utf-8") as handle:
            claims = json.load(handle, parse_constant=_reject_constant)
    except ValueError as e:
        return [f"{claims_path}: NOT strict JSON: {e}"]
    sums: dict[str, str] = {}
    if sums_path.exists():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            m = _SHA_LINE.match(line)
            if m:
                sums[m.group(2)] = m.group(1)
    for claim in claims.get("claims", []):
        name = claim.get("artifact")
        stored = claim.get("artifact_sha256")
        cid = claim.get("claim_id", "?")
        if not name or not stored:
            problems.append(f"claim {cid}: missing artifact reference")
            continue
        target = artifacts_dir / name
        if not target.exists():
            problems.append(f"claim {cid}: artifact {name} does not exist")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != stored:
            problems.append(f"claim {cid}: artifact {name} sha256 mismatch")
        if name in sums and sums[name] != actual:
            problems.append(f"claim {cid}: artifact {name} contradicts SHA256SUMS")
    return problems


def verify(artifacts_dir: Path, repo_root: Path) -> list[str]:
    problems: list[str] = []
    json_files = sorted(artifacts_dir.glob("*.json"))
    if not json_files:
        print("no industrial artifacts present — skipping authority checks")
        return problems

    parsed: dict[str, dict] = {}
    for path in json_files:
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle, parse_constant=_reject_constant)
        except ValueError as e:
            problems.append(f"{path.name}: NOT strict JSON: {e}")
            continue
        if not isinstance(data, dict):
            continue
        schema = data.get("schema")
        role = path.name.replace("industrial-", "").replace(".json", "")
        if schema == SCHEMA_NAME:
            # P0.4: every benchmark artifact ties git_commit to its source
            # commit (identity fields checked in family closure).
            problems.extend(f"{path.name}: {p}" for p in validate_artifact(data))
            source_commit = str((data.get("measurement_source") or {}).get("commit") or "")
            git_commit = str(data.get("git_commit") or "")
            if source_commit and git_commit != source_commit:
                problems.append(
                    f"{path.name}: git_commit {git_commit} != "
                    f"measurement_source.commit {source_commit}"
                )
            parsed[role] = data
        elif schema == RUN_CONFIG_SCHEMA:
            # P0.1: run-config is a first-class family member with its own
            # schema and validator; structural + identity-recompute checks
            # happen in family closure.
            problems.extend(f"{path.name}: {p}" for p in validate_run_config(data))
            parsed["run-config"] = data

    sums = artifacts_dir / "SHA256SUMS.txt"
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = _SHA_LINE.match(line)
            if not m:
                problems.append(f"SHA256SUMS.txt: malformed line {line!r}")
                continue
            expected, name = m.group(1), m.group(2)
            target = artifacts_dir / name
            if not target.exists():
                problems.append(f"SHA256SUMS.txt: missing file {name}")
                continue
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected:
                problems.append(
                    f"SHA256SUMS.txt: {name} hash mismatch (expected {expected}, actual {actual})"
                )
    else:
        problems.append("SHA256SUMS.txt missing — artifact index is not verifiable")

    sums = artifacts_dir / "SHA256SUMS.txt"
    sums_index: dict[str, str] = {}
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = _SHA_LINE.match(line)
            if not m:
                problems.append(f"SHA256SUMS.txt: malformed line {line!r}")
                continue
            expected, name = m.group(1), m.group(2)
            sums_index[name] = expected
            target = artifacts_dir / name
            if not target.exists():
                problems.append(f"SHA256SUMS.txt: missing file {name}")
                continue
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected:
                problems.append(
                    f"SHA256SUMS.txt: {name} hash mismatch (expected {expected}, actual {actual})"
                )
        # B0.5: exact set equality over the DATA members — every data
        # artifact must be indexed; any entry NOT indexing a data artifact
        # is stale/foreign. (manifest.json is the commit marker and
        # SHA256SUMS.txt cannot index itself.)
        data_members = ({p.name for p in json_files} - {"manifest.json"}) | {
            "industrial-distribution-freeze.txt"
        }
        missing_from_sums = data_members - set(sums_index)
        extra_in_sums = set(sums_index) - data_members
        if missing_from_sums:
            problems.append(f"SHA256SUMS.txt: no entry for {sorted(missing_from_sums)}")
        if extra_in_sums:
            problems.append(f"SHA256SUMS.txt: stale entries for {sorted(extra_in_sums)}")
    else:
        problems.append("SHA256SUMS.txt missing — artifact index is not verifiable")

    # P0.6/P0.7: distribution-freeze bytes + manifest membership/hash/size
    # closure — attempted once the basic layer is clean.
    freeze_path = artifacts_dir / "industrial-distribution-freeze.txt"
    freeze_sha = ""
    if freeze_path.exists():
        freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
        lock_freeze = str(
            (parsed.get("run-config", {}).get("environment_lock") or {}).get(
                "distribution_freeze_sha256"
            )
            or ""
        )
        if lock_freeze and freeze_sha != lock_freeze:
            problems.append(
                "industrial-distribution-freeze.txt sha256 does not match "
                "environment_lock.distribution_freeze_sha256"
            )
    else:
        problems.append("industrial-distribution-freeze.txt missing from the published family")

    problems.extend(
        verify_manifest_closure(
            artifacts_dir, parsed, freeze_sha, sums_index if sums.exists() else {}
        )
    )

    # B0.4 family closure + source closure + claims linkage.
    if not problems:
        problems.extend(verify_family_closure(parsed))
        problems.extend(verify_source_closure(sorted(parsed.items()), repo_root))
        problems.extend(
            verify_claims_linkage(
                artifacts_dir,
                Path("docs/generated/industrial-claims.json"),
                artifacts_dir / "SHA256SUMS.txt",
            )
        )

    return problems


def verify_manifest_closure(
    artifacts_dir: Path,
    parsed: dict[str, dict],
    freeze_sha: str,
    sums_index: dict[str, str],
) -> list[str]:
    """Manifest membership/hash/size closure (audit P0.7): the manifest's
    artifact set must equal the canonical family exactly, and every entry
    must agree with SHA256SUMS and the actual bytes."""
    problems: list[str] = []
    manifest = parsed.get("manifest")
    if manifest is None:
        return problems
    canonical = {
        "industrial-runtime.json",
        "industrial-quality.json",
        "industrial-fulldie.json",
        "industrial-run-config.json",
        "industrial-distribution-freeze.txt",
    }
    entries = manifest.get("artifacts") or []
    declared = {
        e.get("file"): e
        for e in entries
        if isinstance(e, dict) and e.get("file") != "manifest.json"
    }
    declared_names = set(declared)
    if declared_names != canonical:
        problems.append(
            f"manifest.json membership mismatch: missing "
            f"{sorted(canonical - declared_names)}, unexpected "
            f"{sorted(declared_names - canonical)}"
        )
    for name in sorted(canonical):
        target = artifacts_dir / name
        if not target.exists():
            problems.append(f"manifest.json: family member {name} missing on disk")
            continue
        actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        actual_bytes = target.stat().st_size
        entry = declared.get(name)
        if entry is not None:
            if str(entry.get("sha256")) != actual_sha:
                problems.append(f"manifest.json: {name} sha256 != actual bytes")
            if int(entry.get("bytes") or -1) != actual_bytes:
                problems.append(f"manifest.json: {name} byte size != actual size")
        if name in sums_index and sums_index[name] != actual_sha:
            problems.append(f"manifest.json: {name} contradicts SHA256SUMS.txt")
        if name == "industrial-distribution-freeze.txt" and freeze_sha and actual_sha != freeze_sha:
            problems.append("industrial-distribution-freeze.txt contradicts the environment lock")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", type=Path, default=Path("benchmarks/results/industrial"))
    args = ap.parse_args()

    problems = verify(args.artifacts, Path.cwd())
    for p in problems:
        print(f"VERIFY FAIL: {p}", file=sys.stderr)
    if problems:
        return 1
    print("industrial artifact authority checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
