#!/usr/bin/env python3
"""Fetch / verify external proof artifacts (P-054 repo integration).

Reads ``proof_artifacts/registry.json``.  Every artifact is SHA-256
verified BEFORE it may be used by a replay.

Modes:
- default (fetch): download declared artifacts that are absent locally,
  then verify sha256 and byte length.  A fetch requires a *file-level*
  ``download_url`` (a DOI landing page is not a transport locator) and a
  non-null ``sha256``; without them the entry is UNFETCHED and the script
  fails loudly.
- ``--verify-only``: never downloads; fails on any required entry that is
  missing on disk, unfetched (``sha256`` null), or hash-mismatched.
  Used by CI to make silent skips impossible for proof replay.

Exit codes: 0 verified; 1 verification failure; 2 download unsupported or
failed.
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


def fetch(entry: dict, destination: Path) -> int:
    """Download one entry and verify it before declaring success."""
    download_url = entry.get("download_url")
    expected = entry.get("sha256")
    if not download_url or not expected:
        print(
            f"FETCH-UNSUPPORTED: {entry['name']} has no file-level download_url "
            "and/or sha256; it cannot be fetched until the frozen publication "
            "provides its exact content identity",
            file=sys.stderr,
        )
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Atomic download: a failed/cancelled transfer must never leave a
    # plausible final artifact path (re-audit §10.2).
    part = destination.with_suffix(destination.suffix + ".part")
    print(f"fetching {entry['name']} from {download_url} ...")
    proc = subprocess.run(  # noqa: S603 — fixed argv from the registry
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--connect-timeout",
            "30",
            "--max-time",
            "1800",
            download_url,
            "-o",
            str(part),
        ],
        check=False,
    )
    if proc.returncode != 0:
        part.unlink(missing_ok=True)
        print(f"FETCH-FAILED: {entry['name']}", file=sys.stderr)
        return 2
    if entry.get("bytes") is not None and part.stat().st_size != entry["bytes"]:
        part.unlink(missing_ok=True)
        print(
            f"BYTE-LENGTH-MISMATCH: {name_of(entry)} {part.stat().st_size} != {entry['bytes']}",
            file=sys.stderr,
        )
        return 1
    actual = sha256_of(part)
    if actual != expected:
        part.unlink(missing_ok=True)
        print(f"HASH-MISMATCH: {entry['name']} {actual} != {expected}", file=sys.stderr)
        return 1
    part.replace(destination)
    print(f"VERIFIED after fetch: {entry['name']} [{actual[:12]}…]")
    return 0


def name_of(entry: dict) -> str:
    return entry.get("name", "<unnamed>")


def verify_entry(entry: dict) -> tuple[int, str]:
    """Verify one entry without downloading. Returns (rc, state)."""
    name = name_of(entry)
    expected = entry.get("sha256")
    destination = ROOT / entry.get("destination", "proof_artifacts/external") / name
    if expected is None:
        print(f"UNFETCHED: {name} (sha256 pending upstream publication)")
        return (1, "UNFETCHED")
    if not destination.exists():
        print(f"MISSING: {name} expected at {destination}", file=sys.stderr)
        return (1, "MISSING")
    actual = sha256_of(destination)
    if actual != expected:
        print(f"HASH-MISMATCH: {name} {actual} != {expected}", file=sys.stderr)
        return (1, "HASH_MISMATCH")
    if entry.get("bytes") is not None and destination.stat().st_size != entry["bytes"]:
        print(
            f"BYTE-LENGTH-MISMATCH: {name} {destination.stat().st_size} != {entry['bytes']}",
            file=sys.stderr,
        )
        return (1, "BYTE_LENGTH_MISMATCH")
    print(f"VERIFIED: {name} [{actual[:12]}…]")
    return (0, "VERIFIED")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="p054-arf37", help="frozen profile to fetch for")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    registry = json.loads(REGISTRY.read_text())
    failures = 0
    for entry in registry.get("artifacts", []):
        if args.profile not in entry.get("profiles", ["p054-arf37"]):
            continue
        if args.verify_only:
            rc, _state = verify_entry(entry)
            if rc:
                failures += 1
        else:
            destination = (
                ROOT / entry.get("destination", "proof_artifacts/external") / entry["name"]
            )
            if destination.exists():
                rc, state = verify_entry(entry)
                if rc == 0:
                    continue
                # present but wrong: re-fetch only with a declared identity
                if entry.get("sha256") is None:
                    failures += 1
                    continue
            failures += fetch(entry, destination)
    if failures:
        print(f"{failures} artifact failure(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
