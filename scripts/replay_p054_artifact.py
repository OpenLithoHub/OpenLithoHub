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


def atomic_write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)
    return path


def result_digest(diagram) -> str:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="p054-arf37")
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--receipt-out", type=Path, required=True)
    ap.add_argument("--report-out", type=Path, required=True)
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
