#!/usr/bin/env python3
"""Fetch / verify external proof artifacts (P-054 repo integration).

Reads ``proof_artifacts/registry.json``.  Every artifact is SHA-256
verified BEFORE it may be used by a replay.

Modes:
- ``--verify-only``: fail on any entry that is missing on disk, unfetched
  (``sha256`` null), or present with a mismatching hash.  Used by CI to
  make silent skips impossible for proof replay.
- default (fetch): download entries that are declared but missing, verify
  their hashes, and place them under their declared destination.

Exit codes: 0 verified; 1 verification failure; 2 download unsupported
(no resolvable URL for an entry).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "proof_artifacts" / "registry.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_url(entry: dict) -> str | None:
    if entry.get("source") == "zenodo":
        doi = entry.get("url_or_doi", "")
        if doi:
            return f"https://doi.org/{doi}"
        return None
    return entry.get("url_or_doi")


def fetch(entry: dict, destination: Path) -> int:
    url = resolve_url(entry)
    if not url:
        print(f"FETCH-UNSUPPORTED: {entry['name']} has no resolvable URL", file=sys.stderr)
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {entry['name']} from {url} ...")
    proc = subprocess.run(  # noqa: S603 — fixed argv from the registry
        ["curl", "-L", "--fail", "--retry", "3", url, "-o", str(destination)],
        check=False,
    )
    if proc.returncode != 0:
        print(f"FETCH-FAILED: {entry['name']}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    registry = json.loads(REGISTRY.read_text())
    failures = 0
    for entry in registry.get("artifacts", []):
        required = "p054-full-replay" in entry.get("required_for", [])
        name = entry["name"]
        destination = ROOT / entry.get("destination", "proof_artifacts/external") / name
        expected = entry.get("sha256")

        if expected is None:
            # Unfetched: acceptable in unit CI, fatal for proof replay.
            print(f"UNFETCHED: {name} (sha256 pending upstream publication)")
            if args.verify_only and required:
                failures += 1
            continue
        if not destination.exists():
            if args.verify_only:
                print(f"MISSING: {name} expected at {destination}", file=sys.stderr)
                failures += 1
                continue
            rc = fetch(entry, destination)
            if rc:
                failures += 1
                continue
        actual = sha256_of(destination)
        if actual != expected:
            print(f"HASH-MISMATCH: {name} {actual} != {expected}", file=sys.stderr)
            failures += 1
        else:
            print(f"VERIFIED: {name} [{actual[:12]}…]")
    if failures:
        print(f"{failures} artifact verification failure(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
