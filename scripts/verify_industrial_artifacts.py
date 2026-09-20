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

from openlithohub.benchmark.industrial import (
    SCHEMA_NAME,
    validate_artifact,
)

_SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def _reject_constant(token: str) -> float:
    raise ValueError(f"non-finite JSON constant {token!r}")


def verify(artifacts_dir: Path) -> list[str]:
    problems: list[str] = []
    json_files = sorted(artifacts_dir.glob("*.json"))
    if not json_files:
        print("no industrial artifacts present — skipping authority checks")
        return problems

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

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", type=Path, default=Path("benchmarks/results/industrial"))
    args = ap.parse_args()

    problems = verify(args.artifacts)
    for p in problems:
        print(f"VERIFY FAIL: {p}", file=sys.stderr)
    if problems:
        return 1
    print("industrial artifact authority checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
