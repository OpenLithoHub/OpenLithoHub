"""P-054 frozen-certificate consistency replay — artifact-required shard (PR-5A).

Every test here is marked ``proof_artifact_required``: in normal unit CI a
missing artifact is an explicit skip; in the proof-replay workflow
(``B04_PROOF_REPLAY_STRICT=1``) a missing or mismatched artifact is a
HARD failure — the replay must never pass silently.

The shard closes the chain audited as open:

    frozen artifact -> registry identity -> verified local bytes
    -> imported-certificate verification -> receipt = IMPORTED_CERTIFICATE_VERIFIED
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from openlithohub.verify.phase_diagram import (
    PhaseArtifactNotAvailableError,
    ReplayState,
    load_frozen_phase_diagram,
    load_verified_frozen_phase_diagram,
)

ROOT = Path(__file__).resolve().parents[2]
P054 = ROOT / "proof_artifacts" / "p054"
ARTIFACT = P054 / "external" / "p054-arf37-frozen-artifact.zip"
REGISTRY = ROOT / "proof_artifacts" / "registry.json"
MANIFEST = json.loads((P054 / "manifest.json").read_text())
EVENT_CATALOG = json.loads((P054 / "event_catalog.json").read_text())
CHAMBER_CATALOG = json.loads((P054 / "chamber_catalog.json").read_text())

pytestmark = pytest.mark.proof_artifact_required

STRICT = os.environ.get("B04_PROOF_REPLAY_STRICT") == "1"
_registry_entry = next(
    a for a in json.loads(REGISTRY.read_text())["artifacts"] if a["name"] == ARTIFACT.name
)


def _require_artifact() -> None:
    """Skip in unit CI, hard-fail in the strict replay workflow."""
    if ARTIFACT.exists():
        return
    message = (
        "p054 frozen artifact not fetched; run "
        "scripts/fetch_proof_artifacts.py --profile p054-arf37"
    )
    if STRICT:
        pytest.fail(message)
    pytest.skip(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


MAX_MEMBERS = 64
MAX_MEMBER_BYTES = 1 << 30


def _no_duplicate_keys(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            pytest.fail(f"duplicate JSON key in bundle: {key!r}")
        seen[key] = value
    return seen


def _bundle() -> dict:
    """Open the verified artifact via the CANONICAL parser (PR-5D3).

    The shard deliberately has no parser of its own: every parse goes
    through load_replay_bundle so no weaker parallel verifier can drift
    into existence.
    """
    _require_artifact()
    return load_replay_bundle(ARTIFACT)


def test_artifact_exists():
    _require_artifact()


def test_outer_zip_sha256_matches_registry():
    _require_artifact()
    expected = _registry_entry.get("sha256")
    assert expected, "registry entry must carry a non-null sha256"
    assert _sha256(ARTIFACT) == expected


def test_byte_length_matches_registry():
    _require_artifact()
    expected = _registry_entry.get("bytes")
    assert expected, "registry entry must carry a non-null byte length"
    assert ARTIFACT.stat().st_size == expected


def test_embedded_manifest_identity_matches_frozen_repo_manifest():
    _require_artifact()
    replay, _members = _bundle()
    assert replay["fixture_id"] == MANIFEST["fixture"]["fixture_id"]
    assert replay["implementation_commit"] == MANIFEST["implementation_commit"]
    assert replay["model_schema"] == MANIFEST["model_schema"]


def test_fixture_and_schema_replay():
    _require_artifact()
    replay, _members = _bundle()
    fixture = json.loads((P054 / "fixture.json").read_text())
    assert replay["fixture_id"] == fixture["fixture_id"]
    assert replay["wavelength_nm"] == fixture["wavelength_nm"]
    assert replay["na"] == fixture["na"]
    assert replay["sigma"] == fixture["sigma"]
    assert replay["grid_shape"] == fixture["grid_shape"]
    assert replay["pixel_size_nm"] == fixture["pixel_size_nm"]
    assert replay["pupil_support_count"] == fixture["pupil_support_count"]
    assert replay["boundary"] == fixture["boundary"]


def test_source_snapshot_hash_matches_bundle_contents():
    _require_artifact()
    replay, members = _bundle()
    declared = replay["source_snapshot_sha256"]
    member_name = next(name for name in members if name.endswith("source_snapshot.npz"))
    assert hashlib.sha256(members[member_name]).hexdigest() == declared


def test_coefficient_tensor_hash_matches_bundle_contents():
    _require_artifact()
    replay, members = _bundle()
    declared = replay["coefficient_tensor_sha256"]
    member_name = next(name for name in members if name.endswith("coefficients.npz"))
    assert hashlib.sha256(members[member_name]).hexdigest() == declared


def test_event_focus_intervals_present_and_ordered():
    _require_artifact()
    replay, _members = _bundle()
    catalog_ids = [e["event_id"] for e in EVENT_CATALOG["events"]]
    events = replay["events"]
    assert [e["event_id"] for e in events] == catalog_ids
    intervals = []
    for event in events:
        interval = event["focus_interval_nm"]
        assert interval is not None, f"{event['event_id']} lacks its frozen interval"
        lo, hi = interval
        assert lo < hi, f"{event['event_id']} interval not ordered"
        intervals.append((lo, hi, event["event_id"]))
    assert intervals == sorted(intervals), "frozen event intervals not in focus order"


def test_event_multiplicity_and_types_replay():
    _require_artifact()
    replay, _members = _bundle()
    catalog = {e["event_id"]: e for e in EVENT_CATALOG["events"]}
    for event in replay["events"]:
        expected = catalog[event["event_id"]]
        assert event["layer"] == expected["layer"]
        assert event["kind"] == expected["kind"]
        assert event["multiplicity"] == expected["multiplicity"]


def test_z5_ownership_invisibility_witness_replays():
    _require_artifact()
    replay, _members = _bundle()
    witnesses = [w for w in replay["witnesses"] if w["critical_event_id"] == "z5"]
    assert len(witnesses) == 1, "exactly one z5 witness must exist"
    witness = witnesses[0]
    assert witness["owner_before"] is not None
    assert witness["owner_before"] == witness["owner_after"]
    chamber_ids = {c["chamber_id"] for c in replay["chambers"]}
    assert {witness["left_chamber_id"], witness["right_chamber_id"]} <= chamber_ids
    by_id = {c["chamber_id"]: c for c in replay["chambers"]}
    for key in ("left_chamber_id", "right_chamber_id"):
        assert "z5" in by_id[witness[key]]["bounded_by_events"]


def test_chamber_boundaries_replay():
    _require_artifact()
    replay, _members = _bundle()
    chambers = replay["chambers"]
    assert [c["chamber_id"] for c in chambers] == [
        c["chamber_id"] for c in CHAMBER_CATALOG["chambers"]
    ]
    for chamber in chambers:
        lo, hi = chamber["focus_interval_nm"]
        assert lo < hi
    # adjacency: consecutive chambers share their bounding event
    for left, right in zip(chambers, chambers[1:], strict=True):
        assert left["bounded_by_events"][-1] == right["bounded_by_events"][0]


def test_target_component_sequence_replays():
    _require_artifact()
    replay, _members = _bundle()
    sequence = [c["target_component_count"] for c in replay["chambers"]]
    assert sequence == CHAMBER_CATALOG["target_component_sequence"] == [1, 3, 5, 4]


def test_proof_level_not_downgraded():
    _require_artifact()
    replay, _members = _bundle()
    assert replay["proof_level"] == "IMPORTED-QDM-CERTIFIED"
    assert (
        MANIFEST["declared_proof_levels"]["finite_declared_model_phase_diagram"]
        == "IMPORTED-QDM-CERTIFIED"
    )


def test_numeric_query_only_after_imported_certificate_verified():
    _require_artifact()
    structure = load_frozen_phase_diagram("p054-arf37")
    assert structure.replay_state is ReplayState.STRUCTURE_ONLY
    with pytest.raises(PhaseArtifactNotAvailableError):
        structure.chamber_at_focus(40.0)

    # Contract B (re-audit PR-5B): verified imported-certificate replay,
    # not source-native recomputation
    numeric = load_verified_frozen_phase_diagram("p054-arf37", artifact_path=ARTIFACT, root=ROOT)
    assert numeric.replay_state is ReplayState.IMPORTED_CERTIFICATE_VERIFIED
    assert numeric.receipt is not None
    assert numeric.receipt.artifact_sha256 == _registry_entry["sha256"]
    assert numeric.receipt.artifact_bytes == ARTIFACT.stat().st_size
    mid = sum(numeric.chambers[1].focus_interval_nm) / 2
    assert numeric.chamber_at_focus(mid).target_component_count == 3
    assert numeric.target_component_count(mid) == 3


def test_hostile_mutated_artifact_byte_fails_hash():
    _require_artifact()
    expected = _registry_entry["sha256"]
    original = ARTIFACT.read_bytes()
    mutated = bytearray(original)
    mutated[len(mutated) // 2] ^= 0xFF
    assert hashlib.sha256(bytes(mutated)).hexdigest() != expected  # H5


def test_hostile_doi_landing_html_rejected_as_artifact():
    _require_artifact()
    expected = _registry_entry["sha256"]
    landing = b"<html>DOI landing page</html>"  # H6
    assert hashlib.sha256(landing).hexdigest() != expected
