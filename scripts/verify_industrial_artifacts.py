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

    parsed: list[tuple[Path, dict]] = []
    for path in json_files:
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle, parse_constant=_reject_constant)
        except ValueError as e:
            problems.append(f"{path.name}: NOT strict JSON: {e}")
            continue
        if isinstance(data, dict) and data.get("schema") == SCHEMA_NAME:
            problems.extend(f"{path.name}: {p}" for p in validate_artifact(data))
            parsed.append((path, data))

    for path in json_files:
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle, parse_constant=_reject_constant)
        except ValueError as e:
            problems.append(f"{path.name}: NOT strict JSON: {e}")
            continue
        if isinstance(data, dict) and data.get("schema") == SCHEMA_NAME:
            problems.extend(f"{path.name}: {p}" for p in validate_artifact(data))

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

    if not problems:
        # Deeper authority checks only run once the basic layer is clean.
        problems.extend(verify_source_closure(parsed, repo_root))
        problems.extend(
            verify_claims_linkage(
                artifacts_dir,
                Path("docs/generated/industrial-claims.json"),
                artifacts_dir / "SHA256SUMS.txt",
            )
        )

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
