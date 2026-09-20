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
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "proof_artifacts" / "registry.json"


def write_fetch_report(
    path: Path,
    *,
    profile: str,
    effective_url: str,
    artifact_sha256: str,
    artifact_bytes: int,
) -> Path:
    """Canonical fetch evidence (PR-5E5): written only after host, size and
    hash validation succeeded; removed together with the partial download
    on any failure."""
    report = {
        "schema": "P054.fetch-report.v1",
        "profile": profile,
        "effective_url": effective_url,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": artifact_bytes,
    }
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)
    return path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    entry: dict,
    destination: Path,
    *,
    profile: str,
    fetch_report_out: Path | None = None,
) -> int:
    """Download one entry and verify it before declaring success.

    ``profile`` is the requested CLI profile and is bound into the fetch
    report.  ``fetch_report_out`` is the canonical report path; when
    omitted a deterministic default next to the artifact is used — the
    path is never silently ignored.
    """
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
    # Transport-origin binding (re-audit R123): a Zenodo-sourced artifact
    # must be fetched from zenodo.org; the hash remains the authority.
    if entry.get("source") == "zenodo" and not download_url.startswith(
        ("https://zenodo.org/", "https://doi.org/")
    ):
        print(
            f"FETCH-REJECTED: {entry['name']} declares Zenodo provenance but "
            f"the initial URL is off-origin",
            file=sys.stderr,
        )
        return 2
    # Atomic download: a failed/cancelled transfer must never leave a
    # plausible final artifact path (re-audit §10.2).
    part = destination.with_suffix(destination.suffix + ".part")
    max_bytes = entry.get("bytes")
    print(f"fetching {entry['name']} from {download_url} ...")
    curl_cmd = [
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
    ]
    if max_bytes:
        # cap the transfer at the declared size (+slack for the cap check)
        curl_cmd += ["--max-filesize", str(int(max_bytes) + (1 << 20))]
    # PR-5E3: ONE curl invocation writes the artifact to .part and emits
    # the final effective URL (after redirects) — the validated origin is
    # the request that actually produced the accepted bytes, and the
    # payload is never transferred twice.
    # PR-5E6: the report path comes from the explicit CLI argument; when
    # omitted a deterministic default next to the artifact is used.
    fetch_report_path = (
        fetch_report_out
        if fetch_report_out is not None
        else destination.parent / (Path(entry["name"]).stem + ".fetch-report.json")
    )

    def cleanup_partial() -> None:
        """Remove the partial download and any report sidecar together."""
        part.unlink(missing_ok=True)
        fetch_report_path.unlink(missing_ok=True)

    curl_cmd += ["-w", "%{url_effective}\\n", "-o", str(part)]
    proc = subprocess.run(  # noqa: S603 — fixed argv from the registry
        curl_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        cleanup_partial()
        print(f"FETCH-FAILED: {entry['name']}", file=sys.stderr)
        return 2
    # PR-5E6: parse the effective URL for EVERY source, then apply
    # source-specific host policy — the generic parser never leaves
    # effective_url undefined.
    from urllib.parse import urlparse

    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    effective_url = lines[-1] if lines else ""
    parsed_host = urlparse(effective_url).hostname or ""
    if entry.get("source") == "zenodo" and parsed_host not in (
        "zenodo.org",
        "www.zenodo.org",
    ):
        cleanup_partial()
        print(
            f"REDIRECT-REJECTED: {entry['name']} landed off-origin at {parsed_host!r}",
            file=sys.stderr,
        )
        return 2
    # PR-5E2: capture the size BEFORE unlinking so the mismatch path cannot
    # crash with FileNotFoundError instead of the intended diagnosis.
    actual_bytes = part.stat().st_size
    if entry.get("bytes") is not None and actual_bytes != entry["bytes"]:
        cleanup_partial()
        print(
            f"BYTE-LENGTH-MISMATCH: {name_of(entry)} {actual_bytes} != {entry['bytes']}",
            file=sys.stderr,
        )
        return 1
    actual = sha256_of(part)
    if actual != expected:
        cleanup_partial()
        print(f"HASH-MISMATCH: {entry['name']} {actual} != {expected}", file=sys.stderr)
        return 1
    # PR-5E5: write the canonical fetch report ONLY after host, size and
    # hash validation all succeeded.
    write_fetch_report(
        fetch_report_path,
        profile=profile,
        effective_url=effective_url,
        artifact_sha256=actual,
        artifact_bytes=actual_bytes,
    )
    print(f"fetch report: {fetch_report_path}")
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


def build_arg_parser() -> argparse.ArgumentParser:
    """The fetcher's public CLI contract (PR-5E6) — testable in isolation."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        default="p054-arf37",
        help="frozen profile to fetch for",
    )
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument(
        "--fetch-report-out",
        type=Path,
        default=None,
        help="canonical fetch report path (P054.fetch-report.v1); defaults "
        "to a deterministic path next to the artifact",
    )
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

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
            fetch_report_path = args.fetch_report_out
            if fetch_report_path is None:
                fetch_report_path = destination.parent / (
                    Path(entry["name"]).stem + ".fetch-report.json"
                )
            failures += fetch(
                entry,
                destination,
                profile=args.profile,
                fetch_report_out=fetch_report_path,
            )
    if failures:
        print(f"{failures} artifact failure(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
