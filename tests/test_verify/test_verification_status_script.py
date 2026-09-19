"""P-054 PR-3B — immutable frozen-basis / status-machine hostile tests.

H4: rewriting the status basis to the current HEAD hard-fails.
Plus: cross-file basis disagreement, profile content-hash drift, and the
activation gate (no passing replay state while the artifact is UNFETCHED).
"""

from scripts.check_verification_status import verify_frozen_state

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
    assert any("hand promotion is not allowed" in f for f in failures)


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
