#!/usr/bin/env python3
"""Canonical P-054 replay receipt producer (re-audit PR-5C).

Runs the single verified factory over the frozen artifact and serializes
the resulting evidence as an atomically-written receipt plus a replay
report.  Exits nonzero on any verification failure.

    python scripts/replay_p054_artifact.py \
        --profile p054-arf37 \
        --artifact proof_artifacts/p054/external/p054-arf37-frozen-artifact.zip \
        --receipt-out /tmp/p054-replay-receipt.json \
        --report-out /tmp/p054-replay-report.json

The receipt is the ONLY evidence the status checker accepts for a passing
``full_artifact_replay`` state; a CI run writes it to an ephemeral path
and a reviewed release change commits it — CI never edits the repository
status silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openlithohub.verify.phase_diagram import (  # noqa: E402
    load_verified_frozen_phase_diagram,
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)
    return path


def result_digest(diagram: Any) -> str:
    """Deterministic digest of the FULL theorem-facing replay surface."""
    payload = {
        "schema": diagram.model_schema,
        "implementation_commit": diagram.implementation_commit,
        "events": [
            [
                e.event_id,
                e.layer.value,
                e.kind.value,
                e.multiplicity,
                list(e.focus_interval_nm or ()),
                getattr(e, "owner_before", None),
                getattr(e, "owner_after", None),
                getattr(e, "component_count_before", None),
                getattr(e, "component_count_after", None),
            ]
            for e in diagram.events
        ],
        "chambers": [
            [
                c.chamber_id,
                list(c.focus_interval_nm or ()),
                c.lower_owner,
                c.upper_owner,
                c.target_component_count,
                list(c.bounded_by_events),
            ]
            for c in diagram.chambers
        ],
        "witnesses": [
            [
                w.critical_event_id,
                w.owner_before,
                w.owner_after,
                w.left_chamber_id,
                w.right_chamber_id,
            ]
            for w in diagram.witnesses
        ],
        "target_component_sequence": list(diagram.target_component_sequence),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def validate_fetch_report(
    fetch_report: dict[str, Any],
    *,
    artifact_sha256: str,
    artifact_bytes: int,
    expected_profile: str = "p054-arf37",
    allowed_hosts: tuple[str, ...] = ("zenodo.org", "www.zenodo.org"),
) -> str:
    """Bind the fetch report to the verified artifact; return its URL.

    Provenance binding (PR-5E6): schema, profile, artifact sha256/bytes
    and the effective-URL host are all validated — a report for another
    profile, or with an off-provenance transport origin, is rejected even
    when the artifact identity itself matches.
    """
    if fetch_report.get("schema") != "P054.fetch-report.v1":
        raise ValueError(
            f"fetch report schema {fetch_report.get('schema')!r} != 'P054.fetch-report.v1'"
        )
    if fetch_report.get("profile") != expected_profile:
        raise ValueError(
            f"fetch report profile {fetch_report.get('profile')!r} != expected {expected_profile!r}"
        )
    if fetch_report.get("artifact_sha256") != artifact_sha256:
        raise ValueError("fetch report artifact_sha256 does not match the verified artifact")
    if fetch_report.get("artifact_bytes") != artifact_bytes:
        raise ValueError("fetch report artifact_bytes does not match the verified artifact")
    effective_url: str = fetch_report.get("effective_url", "")
    if not effective_url.startswith("https://"):
        raise ValueError("fetch report effective_url must be an https URL")
    host = urlparse(effective_url).hostname or ""
    if allowed_hosts and host not in allowed_hosts:
        raise ValueError(
            f"fetch report effective_url host {host!r} is outside the "
            f"expected provenance {list(allowed_hosts)}"
        )
    return effective_url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="p054-arf37")
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--receipt-out", type=Path, required=True)
    ap.add_argument("--report-out", type=Path, required=True)
    ap.add_argument(
        "--fetch-report",
        type=Path,
        default=None,
        help="canonical fetch report (P054.fetch-report.v1) produced by the "
        "fetch; when present its identity is validated against the artifact "
        "and its effective_url is copied into the replay report",
    )
    args = ap.parse_args()

    if not args.artifact.exists():
        print(f"FAIL: artifact missing at {args.artifact}", file=sys.stderr)
        return 1

    try:
        diagram = load_verified_frozen_phase_diagram(
            args.profile, artifact_path=args.artifact, root=ROOT
        )
    except Exception as exc:  # noqa: BLE001 — the failure IS the result
        print(f"REPLAY-FAILED: {exc}", file=sys.stderr)
        return 1

    # PR-5E5: provenance handoff — when a canonical fetch report is
    # supplied, validate it against the verified artifact and carry the
    # real effective URL into the replay report.
    actual_sha = sha256_of(args.artifact)
    actual_bytes = args.artifact.stat().st_size
    effective_url = ""
    if args.fetch_report is not None:
        try:
            fetch_report = json.loads(args.fetch_report.read_text(encoding="utf-8"))
        except ValueError as exc:
            print(f"FAIL: fetch report is not valid JSON: {exc}", file=sys.stderr)
            return 1
        try:
            effective_url = validate_fetch_report(
                fetch_report,
                artifact_sha256=actual_sha,
                artifact_bytes=actual_bytes,
                expected_profile=args.profile,
            )
        except ValueError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1

    receipt = diagram.receipt
    assert receipt is not None  # the verified factory always attaches one
    receipt_payload = {
        "schema": "P054.replay-receipt.v2",
        "profile": receipt.profile,
        "artifact_sha256": receipt.artifact_sha256,
        "artifact_bytes": receipt.artifact_bytes,
        "implementation_commit": receipt.implementation_commit,
        "manifest_sha256": receipt.manifest_sha256,
        "replay_engine_sha256": receipt.replay_engine_sha256,
        "replay_mode": receipt.mode.value,
        "result_digest": result_digest(diagram),
        "event_count": len(diagram.events),
        "chamber_count": len(diagram.chambers),
        "witness_count": len(diagram.witnesses),
    }
    report = {
        "effective_url": effective_url,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "hermetic": True,
        },
        "schema": "P054.replay-report.v2",
        "profile": receipt.profile,
        "artifact_sha256": sha256_of(args.artifact),
        "artifact_bytes": args.artifact.stat().st_size,
        "manifest_sha256": receipt_payload["manifest_sha256"],
        "replay_engine_sha256": receipt_payload["replay_engine_sha256"],
        "mode": receipt.mode.value,
        "events_verified": sorted(e.event_id for e in diagram.events),
        "chambers_verified": sorted(c.chamber_id for c in diagram.chambers),
        "witnesses_verified": sorted(w.critical_event_id for w in diagram.witnesses),
        "result_digest": receipt_payload["result_digest"],
    }
    atomic_write_json(args.receipt_out, receipt_payload)
    atomic_write_json(args.report_out, report)
    print(f"replay receipt: {args.receipt_out}")
    print(f"replay report:  {args.report_out}")
    print(
        f"mode={receipt.mode.value} events={len(diagram.events)} "
        f"chambers={len(diagram.chambers)} witnesses={len(diagram.witnesses)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
