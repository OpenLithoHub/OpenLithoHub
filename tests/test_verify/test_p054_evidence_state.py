"""P-054 PR-4C — replay state is evidence-owned, not caller-owned.

Q1 a receipt-backed numeric diagram validates its own binding;
Q2 numeric receipt + null event intervals -> reject;
Q3 numeric receipt + null z5 owners/chambers -> reject;
Q4 witness artifact hash differing from the receipt -> reject;
Q5 STRUCTURE_ONLY never answers a numeric query (no string can opt in).
"""

import pytest

from openlithohub.verify.phase_diagram import (
    ArtifactVerificationReceipt,
    CriticalSetEventCertificate,
    FocusChamber,
    FrozenPhaseDiagram,
    OwnershipEventCertificate,
    OwnershipInvisibilityWitness,
    PhaseArtifactNotAvailableError,
    ReplayState,
    TargetTopologyEventCertificate,
    load_frozen_phase_diagram,
)
from openlithohub.verify.types import ProofLevel

PD = load_frozen_phase_diagram("p054-arf37")


def _receipt(mode: ReplayState = ReplayState.IMPORTED_CERTIFICATE_VERIFIED):
    return ArtifactVerificationReceipt(
        profile="p054-arf37",
        artifact_sha256="a" * 64,
        artifact_bytes=1024,
        implementation_commit=PD.implementation_commit,
        manifest_sha256="b" * 64,
        mode=mode,
    )


def _numeric_diagram(receipt=None):
    """Diagram with all numeric fields filled (as the factory would)."""
    events = []
    for i, e in enumerate(PD.events):
        interval = (30.0 + i, 40.0 + i)
        if isinstance(e, CriticalSetEventCertificate):
            events.append(
                CriticalSetEventCertificate(
                    event_id=e.event_id,
                    kind=e.kind,
                    focus_interval_nm=interval,
                    multiplicity=e.multiplicity,
                    transversality_lower=e.transversality_lower,
                    morse_signature=e.morse_signature,
                    proof_level=e.proof_level,
                    artifact_sha256="a" * 64,
                )
            )
        elif isinstance(e, OwnershipEventCertificate):
            events.append(
                OwnershipEventCertificate(
                    event_id=e.event_id,
                    kind=e.kind,
                    focus_interval_nm=interval,
                    multiplicity=e.multiplicity,
                    owner_before=e.owner_before,
                    owner_after=e.owner_after,
                    invisible_to_lower_owner=e.invisible_to_lower_owner,
                    proof_level=e.proof_level,
                    artifact_sha256="a" * 64,
                )
            )
        else:
            events.append(
                TargetTopologyEventCertificate(
                    event_id=e.event_id,
                    kind=e.kind,
                    focus_interval_nm=interval,
                    multiplicity=e.multiplicity,
                    component_count_before=e.component_count_before,
                    component_count_after=e.component_count_after,
                    proof_level=e.proof_level,
                    artifact_sha256="a" * 64,
                )
            )
    chambers = tuple(
        FocusChamber(
            chamber_id=c.chamber_id,
            focus_interval_nm=(10.0 + i * 5, 15.0 + i * 5),
            lower_owner="C",
            upper_owner="E",
            target_component_count=c.target_component_count,
            bounded_by_events=c.bounded_by_events + ("z5",) if i < 2 else c.bounded_by_events,
        )
        for i, c in enumerate(PD.chambers)
    )
    witnesses = tuple(
        OwnershipInvisibilityWitness(
            critical_event_id=w.critical_event_id,
            owner_before="C",
            owner_after="C",
            left_chamber_id=PD.chambers[0].chamber_id,
            right_chamber_id=PD.chambers[1].chamber_id,
            proof_level=w.proof_level,
            artifact_sha256="a" * 64,
        )
        for w in PD.witnesses
    )
    return FrozenPhaseDiagram(
        fixture_id=PD.fixture_id,
        model_schema=PD.model_schema,
        implementation_commit=PD.implementation_commit,
        events=tuple(events),
        chambers=chambers,
        target_component_sequence=PD.target_component_sequence,
        witnesses=witnesses,
        manifest=PD.manifest,
        receipt=receipt if receipt is not None else _receipt(),
    )


def test_q1_receipt_backed_numeric_diagram_validates():
    numeric = _numeric_diagram()
    numeric.verify_manifest()  # binding consistent -> no raise
    assert numeric.replay_state is ReplayState.IMPORTED_CERTIFICATE_VERIFIED


def test_q1b_structurally_illegal_numeric_state_rejected_by_type():
    # Q1 (audit): a manually supplied string can no longer opt into the
    # numeric state — the state is derived from the receipt object.
    assert "replay_state" not in FrozenPhaseDiagram.__dataclass_fields__


def test_q2_null_event_interval_rejected_under_numeric_receipt():
    numeric = _numeric_diagram()
    broken = FrozenPhaseDiagram(
        fixture_id=numeric.fixture_id,
        model_schema=numeric.model_schema,
        implementation_commit=numeric.implementation_commit,
        events=[
            CriticalSetEventCertificate(
                event_id=e.event_id,
                kind=e.kind,
                focus_interval_nm=None,
                multiplicity=e.multiplicity,
                transversality_lower=e.transversality_lower,
                morse_signature=e.morse_signature,
                proof_level=e.proof_level,
            )
            if isinstance(e, CriticalSetEventCertificate)
            else e
            for e in numeric.events
        ],
        chambers=numeric.chambers,
        target_component_sequence=numeric.target_component_sequence,
        witnesses=numeric.witnesses,
        manifest=numeric.manifest,
        receipt=numeric.receipt,
    )
    with pytest.raises(ValueError, match="frozen interval"):
        broken.verify_manifest()


def test_q3_null_witness_owner_rejected_under_numeric_receipt():
    numeric = _numeric_diagram()
    broken = FrozenPhaseDiagram(
        fixture_id=numeric.fixture_id,
        model_schema=numeric.model_schema,
        implementation_commit=numeric.implementation_commit,
        events=numeric.events,
        chambers=numeric.chambers,
        target_component_sequence=numeric.target_component_sequence,
        witnesses=(
            type(numeric.witnesses[0])(
                critical_event_id=numeric.witnesses[0].critical_event_id,
                owner_before=None,
                owner_after=None,
                left_chamber_id=None,
                right_chamber_id=None,
                proof_level=numeric.witnesses[0].proof_level,
            ),
        ),
        manifest=numeric.manifest,
        receipt=numeric.receipt,
    )
    with pytest.raises(ValueError, match="non-null owners"):
        broken.verify_manifest()


def test_q4_witness_artifact_hash_mismatch_rejected():
    numeric = _numeric_diagram()
    broken = FrozenPhaseDiagram(
        fixture_id=numeric.fixture_id,
        model_schema=numeric.model_schema,
        implementation_commit=numeric.implementation_commit,
        events=numeric.events,
        chambers=numeric.chambers,
        target_component_sequence=numeric.target_component_sequence,
        witnesses=(
            type(numeric.witnesses[0])(
                critical_event_id=numeric.witnesses[0].critical_event_id,
                owner_before="C",
                owner_after="C",
                left_chamber_id=numeric.witnesses[0].left_chamber_id,
                right_chamber_id=numeric.witnesses[0].right_chamber_id,
                proof_level=numeric.witnesses[0].proof_level,
                artifact_sha256="d" * 64,
            ),
        ),
        manifest=numeric.manifest,
        receipt=numeric.receipt,
    )
    with pytest.raises(ValueError, match="differs from the verification receipt"):
        broken.verify_manifest()


def test_q5_structure_only_never_answers_numeric_query():
    assert PD.replay_state is ReplayState.STRUCTURE_ONLY
    with pytest.raises(PhaseArtifactNotAvailableError):
        PD.chamber_at_focus(40.0)


def test_receipt_profile_or_commit_mismatch_rejected():
    receipt = ArtifactVerificationReceipt(
        profile="other-profile",
        artifact_sha256="a" * 64,
        artifact_bytes=1024,
        implementation_commit="2" * 40,
        manifest_sha256="b" * 64,
        mode=ReplayState.IMPORTED_CERTIFICATE_VERIFIED,
    )
    numeric = _numeric_diagram(receipt=receipt)
    with pytest.raises(ValueError, match="does not match"):
        numeric.verify_manifest()


def test_source_native_recomputed_not_yet_producible():
    # Contract B: no factory produces SOURCE_NATIVE_RECOMPUTED; the enum
    # value exists as the Contract A seam only.
    assert ReplayState.SOURCE_NATIVE_RECOMPUTED.value == "SOURCE_NATIVE_RECOMPUTED"
    assert ProofLevel  # keep import honest


def test_pr4d_verified_type_answers_structural_type_refuses():
    from openlithohub.verify.phase_diagram import VerifiedFrozenPhaseDiagram

    numeric = _numeric_diagram()
    verified = VerifiedFrozenPhaseDiagram(
        fixture_id=numeric.fixture_id,
        model_schema=numeric.model_schema,
        implementation_commit=numeric.implementation_commit,
        events=numeric.events,
        chambers=numeric.chambers,
        target_component_sequence=numeric.target_component_sequence,
        witnesses=numeric.witnesses,
        manifest=numeric.manifest,
        receipt=numeric.receipt,
    )
    mid = sum(verified.chambers[1].focus_interval_nm) / 2
    assert verified.chamber_at_focus(mid).target_component_count == 3
    # the plain structural type refuses even with an identical receipt
    with pytest.raises(PhaseArtifactNotAvailableError):
        numeric.chamber_at_focus(mid)
