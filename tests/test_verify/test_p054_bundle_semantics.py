"""P-054 PR-5D2 — synthetic hostile tests for the canonical bundle verifier.

The re-audit found that `_verify_bundle_semantics()` had no offline
tests (the artifact shard is skipped until publication).  These tests
exercise the healthy path and every audited mutation against a
synthetic bundle dict — no external artifact needed:

- healthy input passes;
- wrong fixture identity, duplicate/permuted events, bad global focus
  order, duplicate/permuted chambers, duplicate witnesses, bad
  intervals, adjacency mismatch, owner/target mismatch, inner member
  hash mismatch, duplicate JSON keys and oversized members all fail.
"""

import hashlib
import json
import math
import zipfile

import pytest

from openlithohub.verify.phase_diagram import (
    MAX_BUNDLE_MEMBERS,
    OwnershipEventCertificate,
    TargetTopologyEventCertificate,
    _reject_duplicate_json_keys,
    _verify_bundle_semantics,
    load_frozen_phase_diagram,
)

PD = load_frozen_phase_diagram("p054-arf37")
FIXTURE = {
    "model_schema": PD.model_schema,
    "wavelength_nm": 193.0,
    "na": 1.35,
    "sigma": 0.7,
    "grid_shape": [72, 72],
    "pixel_size_nm": 8.0,
    "pupil_support_count": 49,
    "boundary": "PERIODIC_FINITE_TILE",
}
SHA = "a" * 64
SNAPSHOT_SHA = "b" * 64
COEFF_SHA = "c" * 64


def _healthy_replay() -> dict:
    events = []
    lo = 100.0
    for event in PD.events:
        hi = lo + 10.0
        entry = {
            "event_id": event.event_id,
            "layer": event.layer.value,
            "kind": event.kind.value,
            "multiplicity": event.multiplicity,
            "focus_interval_nm": [lo, hi],
        }
        if isinstance(event, OwnershipEventCertificate):
            entry["owner_before"] = event.owner_before
            entry["owner_after"] = event.owner_after
        if isinstance(event, TargetTopologyEventCertificate):
            entry["component_count_before"] = event.component_count_before
            entry["component_count_after"] = event.component_count_after
        events.append(entry)
        lo = hi
    chambers = [
        {
            "chamber_id": c.chamber_id,
            "focus_interval_nm": [lo + i * 10, lo + (i + 1) * 10],
            "lower_owner": "C",
            "upper_owner": "E",
            "target_component_count": c.target_component_count,
            "bounded_by_events": list(c.bounded_by_events),
        }
        for i, c in enumerate(PD.chambers)
    ]
    witnesses = [
        {
            "critical_event_id": w.critical_event_id,
            "owner_before": "C",
            "owner_after": "C",
            "left_chamber_id": PD.chambers[0].chamber_id,
            "right_chamber_id": PD.chambers[1].chamber_id,
        }
        for w in PD.witnesses
    ]
    return {
        "fixture_id": "p054-arf37",
        "model_schema": PD.model_schema,
        "wavelength_nm": FIXTURE["wavelength_nm"],
        "na": FIXTURE["na"],
        "sigma": FIXTURE["sigma"],
        "grid_shape": FIXTURE["grid_shape"],
        "pixel_size_nm": FIXTURE["pixel_size_nm"],
        "pupil_support_count": FIXTURE["pupil_support_count"],
        "boundary": FIXTURE["boundary"],
        "implementation_commit": PD.implementation_commit,
        "proof_level": "IMPORTED-QDM-CERTIFIED",
        "source_snapshot_sha256": SNAPSHOT_SHA,
        "coefficient_tensor_sha256": COEFF_SHA,
        "source_snapshot_file": "source_snapshot.npz",
        "coefficient_file": "coefficients.npz",
        "events": events,
        "chambers": chambers,
        "witnesses": witnesses,
    }


def _members() -> dict:
    return {
        "source_snapshot.npz": b"snapshot-bytes",
        "coefficients.npz": b"coefficient-bytes",
    }


def _healthy() -> tuple[dict, dict, dict]:
    replay = _healthy_replay()
    members = _members()
    members.setdefault("mask.bin", b"mask-bytes")
    replay["mask_sha256"] = hashlib.sha256(members["mask.bin"]).hexdigest()
    replay["mask_file"] = "mask.bin"
    # make the declared hashes real
    replay["source_snapshot_sha256"] = hashlib.sha256(members["source_snapshot.npz"]).hexdigest()
    replay["coefficient_tensor_sha256"] = hashlib.sha256(members["coefficients.npz"]).hexdigest()
    return replay, members, dict(FIXTURE)


def test_healthy_synthetic_bundle_passes():
    replay, members, fixture = _healthy()
    _verify_bundle_semantics(replay, members, PD, fixture, SHA)  # no raise


def test_wrong_fixture_identity_rejected():
    replay, members, fixture = _healthy()
    fixture["na"] = 0.9
    with pytest.raises(ValueError, match="na"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_duplicate_events_rejected():
    replay, members, fixture = _healthy()
    replay["events"].append(dict(replay["events"][0]))
    with pytest.raises(ValueError, match="sequence differs"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_permuted_events_rejected():
    replay, members, fixture = _healthy()
    replay["events"][0], replay["events"][1] = replay["events"][1], replay["events"][0]
    with pytest.raises(ValueError, match="sequence differs"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_bad_global_focus_order_rejected():
    replay, members, fixture = _healthy()
    # swap two adjacent intervals without touching the id order
    replay["events"][2]["focus_interval_nm"], replay["events"][3]["focus_interval_nm"] = (
        replay["events"][3]["focus_interval_nm"],
        replay["events"][2]["focus_interval_nm"],
    )
    with pytest.raises(ValueError, match="global focus ordering"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_nan_interval_rejected():
    replay, members, fixture = _healthy()
    replay["events"][0]["focus_interval_nm"] = [float("nan"), 10.0]
    with pytest.raises(ValueError, match="finite and ordered"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_duplicate_chambers_rejected():
    replay, members, fixture = _healthy()
    replay["chambers"].append(dict(replay["chambers"][0]))
    with pytest.raises(ValueError, match="chamber sequence differs"):
        # appending chamber-0 twice yields a sequence that no longer equals
        # the frozen chamber order (duplicate OR permutation both fail)
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_chamber_adjacency_mismatch_rejected():
    replay, members, fixture = _healthy()
    replay["chambers"][0]["bounded_by_events"] = ["zH"]
    with pytest.raises(ValueError, match="bounded_by_events mismatch"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_owner_mismatch_rejected():
    replay, members, fixture = _healthy()
    for e in replay["events"]:
        if e["event_id"] == "z3":
            e["owner_after"] = "D"
    with pytest.raises(ValueError, match="ownership mismatch"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_target_component_mismatch_rejected():
    replay, members, fixture = _healthy()
    for e in replay["events"]:
        if e["event_id"] == "zP":
            e["component_count_after"] = 6
    with pytest.raises(ValueError, match="component counts mismatch"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_inner_member_hash_mismatch_rejected():
    replay, members, fixture = _healthy()
    members["coefficients.npz"] = b"tampered"
    with pytest.raises(ValueError, match="does not match coefficient_tensor_sha256"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_exact_member_identity_required_not_substring():
    replay, members, fixture = _healthy()
    del replay["source_snapshot_file"]
    with pytest.raises(ValueError, match="exact member identity"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_duplicate_witness_rejected():
    replay, members, fixture = _healthy()
    replay["witnesses"].append(dict(replay["witnesses"][0]))
    with pytest.raises(ValueError, match="witness set differs"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _reject_duplicate_json_keys([("a", 1), ("a", 2)])


def test_member_and_count_limits():
    assert MAX_BUNDLE_MEMBERS == 64
    assert math.isfinite(float(MAX_BUNDLE_MEMBERS))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _reject_duplicate_json_keys([("replay_manifest", 1), ("replay_manifest", 2)])


def test_zip_duplicate_member_names_rejected_by_loader(tmp_path):
    from openlithohub.verify.phase_diagram import load_replay_bundle

    artifact = tmp_path / "bundle.zip"
    with zipfile.ZipFile(artifact, "w") as zf:
        zf.writestr("replay_manifest.json", json.dumps({"fixture_id": "p054-arf37"}))
        # duplicate member name: the second write appends another entry
        zf.writestr("source_snapshot.npz", b"x")
        zf.writestr("source_snapshot.npz", b"y")
    with pytest.raises(ValueError, match="duplicate ZIP member names"):
        load_replay_bundle(artifact)


# ---------------------------------------------------------------------------
# PR-5D3 — concrete ZIP-envelope hostiles (P1/P2) + chamber permutation (P3)
# ---------------------------------------------------------------------------


def _write_zip(path, member_payloads: dict[str, bytes]) -> None:
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in member_payloads.items():
            zf.writestr(name, payload)


def test_p1_member_count_over_limit_rejected(tmp_path):

    from openlithohub.verify import phase_diagram

    artifact = tmp_path / "many.zip"
    payloads = {f"m{i}.bin": b"x" for i in range(phase_diagram.MAX_BUNDLE_MEMBERS + 1)}
    _write_zip(artifact, payloads)
    with pytest.raises(ValueError, match="member limit"):
        phase_diagram.load_replay_bundle(artifact)


def test_p2_oversized_member_rejected(tmp_path, monkeypatch):
    from openlithohub.verify import phase_diagram

    monkeypatch.setattr(phase_diagram, "MAX_BUNDLE_MEMBER_BYTES", 16)
    artifact = tmp_path / "big.zip"
    _write_zip(artifact, {"replay_manifest.json": b"{}", "huge.bin": b"x" * 32})
    with pytest.raises(ValueError, match="per-member size limit"):
        phase_diagram.load_replay_bundle(artifact)


def test_p2b_total_uncompressed_cap_rejected(tmp_path, monkeypatch):
    from openlithohub.verify import phase_diagram

    monkeypatch.setattr(phase_diagram, "MAX_BUNDLE_TOTAL_UNCOMPRESSED_BYTES", 16)
    artifact = tmp_path / "total.zip"
    _write_zip(
        artifact,
        {"a.bin": b"x" * 8, "b.bin": b"y" * 8, "c.bin": b"z" * 8},
    )
    with pytest.raises(ValueError, match="total uncompressed limit"):
        phase_diagram.load_replay_bundle(artifact)


def test_p3_chamber_order_permutation_rejected():
    replay, members, fixture = _healthy()
    replay["chambers"][0], replay["chambers"][1] = (
        replay["chambers"][1],
        replay["chambers"][0],
    )
    with pytest.raises(ValueError, match="chamber sequence differs"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


# ---------------------------------------------------------------------------
# PR-5F2 — canonical mask member binding (audit third gap)
# ---------------------------------------------------------------------------


def _healthy_with_mask():
    replay, members, fixture = _healthy()
    mask_sha = hashlib.sha256(members.setdefault("mask.bin", b"mask-bytes")).hexdigest()
    replay["mask_sha256"] = mask_sha
    replay["mask_file"] = "mask.bin"
    return replay, members, fixture, mask_sha


def test_mask_member_binding_healthy_passes():
    replay, members, fixture, mask_sha = _healthy_with_mask()
    _verify_bundle_semantics(replay, members, PD, fixture, SHA)  # no raise
    assert mask_sha == hashlib.sha256(b"mask-bytes").hexdigest()


def test_mask_member_missing_rejected():
    replay, members, fixture, _ = _healthy_with_mask()
    del members["mask.bin"]
    with pytest.raises(ValueError, match="is missing"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_mask_bytes_altered_rejected():
    replay, members, fixture, _ = _healthy_with_mask()
    members["mask.bin"] = b"tampered-mask"
    with pytest.raises(ValueError, match="does not match mask_sha256"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_mask_wrong_declared_sha_rejected():
    replay, members, fixture, _ = _healthy_with_mask()
    replay["mask_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="does not match mask_sha256"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_mask_wrong_file_name_rejected():
    replay, members, fixture, _ = _healthy_with_mask()
    replay["mask_file"] = "not-the-mask.bin"
    with pytest.raises(ValueError, match="is missing"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)


def test_model_schema_mismatch_rejected():
    replay, members, fixture = _healthy()
    replay["model_schema"] = "P054.frozen-arf37.v0"
    with pytest.raises(ValueError, match="model_schema"):
        _verify_bundle_semantics(replay, members, PD, fixture, SHA)
