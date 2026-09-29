"""Industrial Scale Lane-A campaign — window-scoped formal authority.

GPU Authority Repair §19: the formal Lane A campaign is WINDOW-SCOPED —
each frozen window (4096 / 8192 / 16384 / 32768) is its own independent
run with its own canonical family and its own verifier PASS, and the
campaign manifest binds exactly those four windows under ONE shared
source / fixture / environment / protocol identity.  An interrupted
campaign restarts at only the missing window; arbitrary rows are never
stitched together after the fact.

The manifest is content-validated and fail-closed:

* exactly the frozen windows, each present exactly once;
* every family re-verified at the formal tier by the Scale verifier;
* every recorded family/row SHA re-checked against the family bytes;
* ONE measurement source SHA, ONE fixture SHA, ONE environment-lock
  SHA and ONE frozen protocol across all four windows — any mixture is
  a hard blocker.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openlithohub.benchmark.industrial_scale import (
    FROZEN_LANE_A_WINDOWS,
    LANE_A,
)

LANE_A_CAMPAIGN_SCHEMA = "OpenLithoHub.industrial-scale-lane-a-campaign.v1"

ROW_MEMBER = "industrial-scale-index.json"

# The frozen protocol knobs the campaign binds (§19: "P1 protocol;
# tile / halo / microbatch; warmup / repeats").
PROTOCOL_KEYS: tuple[str, ...] = (
    "forward_profile",
    "tile_size",
    "halo_px",
    "microbatch",
    "warmup_count",
    "repeat_count",
    "device_backend",
    "gpu_count",
    "selected_layer",
    "sink_kind",
)


def _load_verifier() -> Any:
    """Load the Scale artifact verifier — the ONLY family-closure
    authority; the campaign never re-implements family verification."""
    import importlib.util

    repo = Path(__file__).resolve().parents[3]
    verifier_path = repo / "scripts" / "verify_industrial_scale_artifacts.py"
    if not verifier_path.is_file():
        raise RuntimeError(f"scale verifier not found at {verifier_path}")
    spec = importlib.util.spec_from_file_location("verify_scale_artifacts_campaign", verifier_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the scale verifier at {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_sums(root: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        sums[name.strip()] = digest.strip()
    return sums


def _family_facts(family_dir: Path) -> dict[str, Any]:
    """Extract the authority facts of one window family — the family MUST
    already be a verifier-closed formal family."""
    run_config_member = json.loads((family_dir / "industrial-scale-run-config.json").read_text())
    payload = run_config_member["run_config"]
    freeze = json.loads((family_dir / "industrial-scale-distribution-freeze.txt").read_text())
    env_lock = freeze["gpu"]
    from openlithohub.benchmark.industrial_scale import scale_environment_lock_sha256

    sums = _parse_sums(family_dir)
    return {
        "run_identity": run_config_member["run_identity"],
        "measurement_commit": run_config_member["measurement_commit"],
        "run_config": payload,
        "fixture_gds_sha256": payload["fixture_gds_sha256"],
        "environment_lock": env_lock,
        "environment_lock_sha256": run_config_member["environment_lock_sha256"],
        "recomputed_env_lock_sha256": scale_environment_lock_sha256(env_lock),
        "member_sha256": sums,
        "row_member_sha256": sums[ROW_MEMBER],
        "formal": bool(run_config_member.get("tracked_tree_clean"))
        and not bool(run_config_member.get("provisional")),
    }


def build_lane_a_campaign(families: Mapping[int, Path]) -> dict[str, Any]:
    """Build the Lane-A campaign manifest from exactly the four frozen
    window families.  Every family is re-verified (formal tier) here —
    a builder that accepts an unverified family would be a manual
    stitching path, which §19 forbids."""
    verifier = _load_verifier()
    windows = sorted(families)
    if tuple(windows) != FROZEN_LANE_A_WINDOWS:
        raise ValueError(
            f"campaign windows {windows} != the frozen Lane-A ladder "
            f"{list(FROZEN_LANE_A_WINDOWS)} — exactly the four frozen windows, "
            "each exactly once"
        )
    members: list[dict[str, Any]] = []
    for window in windows:
        family_dir = Path(families[window])
        try:
            result = verifier.verify(family_dir, require_formal=True)
        except Exception as exc:  # noqa: BLE001 — any family failure blocks the campaign
            raise ValueError(f"window {window}: family verifier PASS missing — {exc}") from exc
        facts = _family_facts(family_dir)
        if not facts["formal"]:
            raise ValueError(f"window {window}: family is not a formal run")
        protocol = {key: facts["run_config"].get(key) for key in PROTOCOL_KEYS}
        members.append(
            {
                "window": window,
                "run_identity": facts["run_identity"],
                "measurement_commit": facts["measurement_commit"],
                "fixture_gds_sha256": facts["fixture_gds_sha256"],
                "environment_lock_sha256": facts["environment_lock_sha256"],
                "protocol": protocol,
                "member_sha256": facts["member_sha256"],
                "row_member_sha256": facts["row_member_sha256"],
                "verifier_pass": True,
                "verifier_tier": "formal",
                "verifier_claims": list(result),
            }
        )
    first = members[0]
    manifest: dict[str, Any] = {
        "schema": LANE_A_CAMPAIGN_SCHEMA,
        "lane": LANE_A,
        "frozen_windows": list(FROZEN_LANE_A_WINDOWS),
        "measurement_commit": first["measurement_commit"],
        "fixture_gds_sha256": first["fixture_gds_sha256"],
        "environment_lock_sha256": first["environment_lock_sha256"],
        "protocol": first["protocol"],
        "members": members,
    }
    blockers = verify_lane_a_campaign(manifest, families)
    if blockers:
        raise ValueError("campaign manifest failed self-verification: " + "; ".join(blockers))
    return manifest


def verify_lane_a_campaign(manifest: Mapping[str, Any], families: Mapping[int, Path]) -> list[str]:
    """Every reason the campaign manifest cannot be accepted — fail-closed;
    an empty list is the ONLY acceptance license (§19)."""
    blockers: list[str] = []
    if manifest.get("schema") != LANE_A_CAMPAIGN_SCHEMA:
        blockers.append(f"campaign schema is not {LANE_A_CAMPAIGN_SCHEMA!r}")
    if manifest.get("lane") != LANE_A:
        blockers.append(f"campaign lane {manifest.get('lane')!r} is not {LANE_A!r}")
    recorded_windows = manifest.get("frozen_windows") or []
    if sorted(int(w) for w in recorded_windows) != sorted(FROZEN_LANE_A_WINDOWS):
        blockers.append(
            f"campaign windows {recorded_windows} != the frozen Lane-A ladder "
            f"{list(FROZEN_LANE_A_WINDOWS)}"
        )
    members = manifest.get("members") or []
    member_windows = [int(m.get("window", -1)) for m in members]
    if sorted(member_windows) != sorted(FROZEN_LANE_A_WINDOWS):
        blockers.append(
            f"campaign members cover {sorted(member_windows)} — exactly the four frozen "
            "windows, each exactly once, is required (missing/duplicate window)"
        )
        return blockers
    if set(member_windows) != set(families):
        blockers.append(
            "family dirs provided do not cover exactly the campaign windows: "
            f"{sorted(families)} vs {sorted(member_windows)}"
        )
        return blockers

    verifier = _load_verifier()
    reference = members[0]
    for member in members:
        window = int(member["window"])
        family_dir = Path(families[window])
        # 1. the family itself must still be a formal verifier PASS
        try:
            verifier.verify(family_dir, require_formal=True)
        except Exception as exc:  # noqa: BLE001 — any family failure blocks the campaign
            blockers.append(f"window {window}: family verifier PASS missing — {exc}")
            continue
        facts = _family_facts(family_dir)
        # 2. run identity + SHAs must match the recorded manifest entry
        if facts["run_identity"] != member.get("run_identity"):
            blockers.append(
                f"window {window}: run identity drift {facts['run_identity'][:16]}… != "
                f"recorded {str(member.get('run_identity'))[:16]}…"
            )
        if facts["member_sha256"] != member.get("member_sha256"):
            blockers.append(f"window {window}: family member SHA256SUMS drift")
        if facts["row_member_sha256"] != member.get("row_member_sha256"):
            blockers.append(f"window {window}: row SHA drift ({ROW_MEMBER})")
        # 3. ONE source across windows — mixed measurement source is fatal
        if facts["measurement_commit"] != reference.get("measurement_commit"):
            recorded_commit = str(reference.get("measurement_commit", ""))
            blockers.append(
                f"window {window}: mixed measurement source "
                f"{facts['measurement_commit'][:12]}… != {recorded_commit[:12]}…"
            )
        if facts["measurement_commit"] != manifest.get("measurement_commit"):
            blockers.append(f"window {window}: measurement commit != campaign manifest")
        # 4. ONE fixture across windows
        if facts["fixture_gds_sha256"] != reference.get("fixture_gds_sha256"):
            blockers.append(f"window {window}: mixed fixture sha256")
        if facts["fixture_gds_sha256"] != manifest.get("fixture_gds_sha256"):
            blockers.append(f"window {window}: fixture sha256 != campaign manifest")
        # 5. ONE environment across windows
        if facts["environment_lock_sha256"] != reference.get("environment_lock_sha256"):
            blockers.append(f"window {window}: mixed environment-lock sha256")
        if facts["environment_lock_sha256"] != manifest.get("environment_lock_sha256"):
            blockers.append(f"window {window}: environment-lock sha256 != campaign manifest")
        # 6. ONE frozen protocol across windows
        protocol = {key: facts["run_config"].get(key) for key in PROTOCOL_KEYS}
        if protocol != reference.get("protocol"):
            reference_protocol = reference.get("protocol") or {}
            drifted = [
                key for key in PROTOCOL_KEYS if protocol.get(key) != reference_protocol.get(key)
            ]
            blockers.append(f"window {window}: mixed protocol (drifted knobs: {sorted(drifted)})")
        if protocol != manifest.get("protocol"):
            blockers.append(f"window {window}: protocol != campaign manifest")
    return blockers


def campaign_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """The campaign manifest's own canonical SHA-256 (for evidence
    packaging: the manifest is content-addressed like every artifact)."""
    from openlithohub.benchmark.industrial_scale import canonical_json

    return hashlib.sha256(canonical_json(dict(manifest)).encode()).hexdigest()


def write_campaign_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    from openlithohub.benchmark.industrial_scale import write_strict_json

    write_strict_json(str(path), dict(manifest))
    return _sha256_file(Path(path))


__all__ = [
    "FROZEN_LANE_A_WINDOWS",
    "LANE_A_CAMPAIGN_SCHEMA",
    "PROTOCOL_KEYS",
    "ROW_MEMBER",
    "build_lane_a_campaign",
    "campaign_manifest_sha256",
    "verify_lane_a_campaign",
    "write_campaign_manifest",
]
