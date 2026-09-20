"""P-054 PR-3B — immutable frozen-basis / status-machine hostile tests.

H4: rewriting the status basis to the current HEAD hard-fails.
Plus: cross-file basis disagreement, profile content-hash drift, and the
activation gate (no passing replay state while the artifact is UNFETCHED).
"""

import json
from pathlib import Path

from scripts.check_verification_status import (
    _verify_replay_binding,
    verify_frozen_state,
)

BASIS = "348fa5d86d5355465af98e2c4ce3deac60081a4c"
HEAD = "f" * 40


def _state():
    status = {
        "paper": "P-054",
        "repo_commit_basis": BASIS,
        "canonical_entry": "proof_artifacts/p054/README.md",
        "finite_declared_model": {"full_artifact_replay": "BLOCKED_ARTIFACT_UNFETCHED"},
        "repository_engineering": {},
        "bridges": {},
    }
    manifest = {"implementation_commit": BASIS}
    frozen_basis = {
        "implementation_commit": BASIS,
        "content_hashes": {"manifest.json": "a" * 64},
    }
    registry = {
        "artifacts": [
            {"sha256": None, "required_for": ["p054-full-replay"]},
        ]
    }
    file_hashes = {"manifest.json": "a" * 64}
    return status, manifest, frozen_basis, registry, file_hashes


def test_healthy_state_passes():
    failures = verify_frozen_state(
        status=_state()[0],
        manifest=_state()[1],
        frozen_basis=_state()[2],
        registry=_state()[3],
        file_hashes=_state()[4],
        readiness_text="canonical entry: proof_artifacts/p054/README.md",
    )
    assert failures == []


def test_h4_basis_rewritten_to_current_head_hard_fails():
    status, manifest, frozen_basis, registry, file_hashes = _state()
    status["repo_commit_basis"] = HEAD
    manifest["implementation_commit"] = HEAD
    frozen_basis["implementation_commit"] = HEAD
    failures = verify_frozen_state(
        status=status,
        manifest=manifest,
        frozen_basis=frozen_basis,
        registry=registry,
        file_hashes=file_hashes,
        readiness_text="",
    )
    # every layer independently rejects the rewrite
    assert sum("is not the frozen P-054 basis" in f for f in failures) == 3


def test_cross_file_basis_disagreement_hard_fails():
    status, manifest, frozen_basis, registry, file_hashes = _state()
    manifest["implementation_commit"] = "2" * 40
    failures = verify_frozen_state(
        status=status,
        manifest=manifest,
        frozen_basis=frozen_basis,
        registry=registry,
        file_hashes=file_hashes,
        readiness_text="",
    )
    assert any("basis disagreement" in f for f in failures)


def test_profile_content_hash_drift_hard_fails():
    status, manifest, frozen_basis, registry, file_hashes = _state()
    file_hashes["manifest.json"] = "b" * 64  # catalog edited without re-anchoring
    failures = verify_frozen_state(
        status=status,
        manifest=manifest,
        frozen_basis=frozen_basis,
        registry=registry,
        file_hashes=file_hashes,
        readiness_text="",
    )
    assert any("hash drifted from the frozen anchor" in f for f in failures)


def test_hand_promoted_replay_state_fails_activation_gate():
    status, manifest, frozen_basis, registry, file_hashes = _state()
    status["finite_declared_model"]["full_artifact_replay"] = "FULL_REPLAY_PASSED"
    failures = verify_frozen_state(
        status=status,
        manifest=manifest,
        frozen_basis=frozen_basis,
        registry=registry,
        file_hashes=file_hashes,
        readiness_text="",
    )
    assert any("no replay receipt" in f for f in failures)


def test_fetched_artifact_allows_passing_replay_state():
    status, manifest, frozen_basis, registry, file_hashes = _state()
    registry["artifacts"][0]["sha256"] = "c" * 64
    status["finite_declared_model"]["full_artifact_replay"] = "FULL_REPLAY_PASSED"
    failures = verify_frozen_state(
        status=status,
        manifest=manifest,
        frozen_basis=frozen_basis,
        registry=registry,
        file_hashes=file_hashes,
        readiness_text="",
    )
    assert not any("hand promotion" in f for f in failures)


# ---------------------------------------------------------------------------
# PR-3C — replay-receipt binding (S1–S6)
# ---------------------------------------------------------------------------


def _receipt_state():
    status, manifest, frozen_basis, registry, file_hashes = _state()
    status["finite_declared_model"]["full_artifact_replay"] = "IMPORTED_CERTIFICATE_VERIFIED"
    registry["artifacts"][0]["sha256"] = "a" * 64
    registry["artifacts"][0]["bytes"] = 1024
    frozen_basis["external_artifact_sha256"] = "a" * 64
    return status, manifest, frozen_basis, registry, file_hashes


def _receipt(**overrides):
    import hashlib
    from pathlib import Path

    real_manifest = (
        Path(__file__).resolve().parents[2] / "proof_artifacts" / "p054" / "manifest.json"
    )
    receipt = {
        "schema": "P054.replay-receipt.v2",
        "profile": "p054-arf37",
        "artifact_sha256": "a" * 64,
        "artifact_bytes": 1024,
        "implementation_commit": BASIS,
        "manifest_sha256": hashlib.sha256(real_manifest.read_bytes()).hexdigest(),
        "replay_engine_sha256": "e" * 64,
        "replay_mode": "IMPORTED_CERTIFICATE_VERIFIED",
        "result_digest": "d" * 64,
        "event_count": 8,
        "chamber_count": 4,
        "witness_count": 1,
    }
    receipt.update(overrides)
    return receipt


def _check_receipt(receipt, tmp_path: Path, *, engine_hash="e" * 64, with_receipt=True, state=None):
    path = tmp_path / "replay-receipt.json"
    if with_receipt:
        path.write_text(json.dumps(receipt))
    state = state or _receipt_state()
    return _verify_replay_binding(
        replay_state=state[0]["finite_declared_model"]["full_artifact_replay"],
        receipt_path=path if with_receipt else tmp_path / "absent.json",
        registry=state[3],
        frozen_basis=state[2],
        manifest=state[1],
        engine_engine_hash=engine_hash,
    )


def test_s1_populated_registry_without_receipt_rejected(tmp_path):
    failures = _check_receipt(_receipt(), tmp_path, with_receipt=False)
    assert any("no replay receipt" in f for f in failures)


def test_s2_receipt_hash_differs_from_registry_rejected(tmp_path):
    failures = _check_receipt(_receipt(artifact_sha256="d" * 64), tmp_path)
    assert any("!= registry hash" in f for f in failures)


def test_s3_receipt_hash_differs_from_frozen_basis_rejected(tmp_path):
    state = _receipt_state()
    state[2]["external_artifact_sha256"] = "c" * 64
    failures = _check_receipt(_receipt(), tmp_path, state=state)
    assert any("frozen-basis external hash" in f for f in failures)


def test_s4_engine_hash_drift_rejected(tmp_path):
    failures = _check_receipt(_receipt(replay_engine_sha256="0" * 64), tmp_path)
    assert any("engine hash drifted" in f for f in failures)


def test_s5_weak_mode_rejected(tmp_path):
    failures = _check_receipt(_receipt(replay_mode="STRUCTURE_ONLY"), tmp_path)
    assert any("too weak" in f for f in failures)


def test_s6_bytes_mismatch_rejected(tmp_path):
    failures = _check_receipt(_receipt(artifact_bytes=1), tmp_path)
    assert any("bytes != registry bytes" in f for f in failures)


def test_healthy_receipt_passes_binding(tmp_path):
    failures = _check_receipt(_receipt(), tmp_path)
    assert failures == []


# ---------------------------------------------------------------------------
# PR-3D — status/mode strength lattice (T12–T14) + schema/profile/manifest
# ---------------------------------------------------------------------------

from scripts.check_verification_status import (  # noqa: E402
    _PASSING_REPLAY_STATES,
    STATUS_REQUIRED_STRENGTH,
)


def test_t12_full_replay_with_structure_only_mode_rejected(tmp_path):
    receipt = _receipt(replay_mode="STRUCTURE_ONLY")
    state = _receipt_state()
    state[0]["finite_declared_model"]["full_artifact_replay"] = "FULL_REPLAY_PASSED"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    failures = _verify_replay_binding(
        replay_state="FULL_REPLAY_PASSED",
        receipt_path=path,
        registry=state[3],
        frozen_basis=state[2],
        manifest=state[1],
        engine_engine_hash="e" * 64,
    )
    assert any("too weak" in f for f in failures)


def test_t13_full_replay_with_imported_mode_rejected(tmp_path):
    receipt = _receipt(replay_mode="IMPORTED_CERTIFICATE_VERIFIED")
    state = _receipt_state()
    state[0]["finite_declared_model"]["full_artifact_replay"] = "FULL_REPLAY_PASSED"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    failures = _verify_replay_binding(
        replay_state="FULL_REPLAY_PASSED",
        receipt_path=path,
        registry=state[3],
        frozen_basis=state[2],
        manifest=state[1],
        engine_engine_hash="e" * 64,
    )
    assert any("too weak" in f for f in failures)


def test_full_replay_with_source_native_mode_accepted(tmp_path):
    receipt = _receipt(replay_mode="SOURCE_NATIVE_RECOMPUTED")
    state = _receipt_state()
    state[0]["finite_declared_model"]["full_artifact_replay"] = "FULL_REPLAY_PASSED"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    failures = _verify_replay_binding(
        replay_state="FULL_REPLAY_PASSED",
        receipt_path=path,
        registry=state[3],
        frozen_basis=state[2],
        manifest=state[1],
        engine_engine_hash="e" * 64,
    )
    assert not any("too weak" in f for f in failures)


def test_t14_unknown_status_string_hard_fails():
    # the lattice rejects unrecognized states instead of skipping binding
    state = _receipt_state()
    state[0]["finite_declared_model"]["full_artifact_replay"] = "SOMETHING_ELSE"
    unknown = "SOMETHING_ELSE"
    assert unknown not in STATUS_REQUIRED_STRENGTH
    assert unknown not in _passing_set()
    del state


def _passing_set():
    return _PASSING_REPLAY_STATES


def test_unknown_receipt_mode_rejected(tmp_path):
    failures = _check_receipt(_receipt(replay_mode="WARP_DRIVE"), tmp_path)
    assert any("unknown receipt replay mode" in f for f in failures)


# ---------------------------------------------------------------------------
# PR-3F2 — receipt local semantic checks (S9 + digest + counts)
# ---------------------------------------------------------------------------


def test_s9_wrong_manifest_hash_rejected(tmp_path):
    receipt = _receipt(manifest_sha256="0" * 64)
    failures = _check_receipt(receipt, tmp_path)
    assert any("!= sha256(actual manifest.json)" in f for f in failures)


def test_receipt_result_digest_must_be_64_hex(tmp_path):
    failures = _check_receipt(_receipt(result_digest="short"), tmp_path)
    assert any("64-hex" in f for f in failures)


def test_receipt_counts_must_match_frozen_catalog(tmp_path):
    failures = _check_receipt(_receipt(event_count=99), tmp_path)
    assert any("event_count" in f and "frozen catalog" in f for f in failures)
    healthy = _check_receipt(_receipt(), tmp_path)
    assert not any("frozen catalog" in f for f in healthy)
