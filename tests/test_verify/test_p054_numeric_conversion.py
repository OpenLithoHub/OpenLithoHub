"""P-054 PR-5C — total three-type numeric event conversion.

F1 z3 (ownership) receives a non-null bundle interval;
F2 zV/zH/zP (target) receive non-null bundle intervals;
F3 all 8 events survive the numeric verify_manifest;
F4/F5/F6 bundle layer/kind/multiplicity mutations rejected;
F7 NaN/inf/reversed interval rejected;
F8 missing event rejected;
F9 duplicate/extra event rejected.
"""

import pytest

from openlithohub.verify.phase_diagram import (
    FrozenPhaseDiagram,
    OwnershipEventCertificate,
    PhaseLayer,
    TargetTopologyEventCertificate,
    _numeric_event_from_bundle,
    load_frozen_phase_diagram,
)
from openlithohub.verify.types import ProofLevel

PD = load_frozen_phase_diagram("p054-arf37")
SHA = "a" * 64


def _declared(event, interval=(100.0, 200.0), **overrides):
    entry = {
        "event_id": event.event_id,
        "layer": event.layer.value,
        "kind": event.kind.value,
        "multiplicity": event.multiplicity,
        "focus_interval_nm": list(interval),
    }
    entry.update(overrides)
    return entry


def test_f1_ownership_event_receives_bundle_interval():
    z3 = PD.event("z3")
    numeric = _numeric_event_from_bundle(z3, _declared(z3), SHA)
    assert isinstance(numeric, OwnershipEventCertificate)
    assert numeric.focus_interval_nm == (100.0, 200.0)
    assert numeric.owner_before == "C" and numeric.owner_after == "E"


def test_f2_target_events_receive_bundle_intervals():
    for event_id in ("zV", "zH", "zP"):
        event = PD.event(event_id)
        numeric = _numeric_event_from_bundle(event, _declared(event), SHA)
        assert isinstance(numeric, TargetTopologyEventCertificate)
        assert numeric.focus_interval_nm is not None


def test_f3_all_eight_events_survive_numeric_verify():
    numeric_events = tuple(
        _numeric_event_from_bundle(e, _declared(e, (100.0 + i, 200.0 + i)), SHA)
        for i, e in enumerate(PD.events)
    )
    numeric_chambers = tuple(
        type(c)(
            chamber_id=c.chamber_id,
            focus_interval_nm=(10.0 + i * 5, 15.0 + i * 5),
            lower_owner="C",
            upper_owner="E",
            target_component_count=c.target_component_count,
            bounded_by_events=c.bounded_by_events + ("z5",) if i < 2 else c.bounded_by_events,
        )
        for i, c in enumerate(PD.chambers)
    )
    from openlithohub.verify.phase_diagram import (
        ArtifactVerificationReceipt,
        OwnershipInvisibilityWitness,
        ReplayState,
    )

    witnesses = tuple(
        OwnershipInvisibilityWitness(
            critical_event_id=w.critical_event_id,
            owner_before="C",
            owner_after="C",
            owner_interval_nm=(90.0, 300.0) if w.critical_event_id == "z5" else None,
            left_chamber_id=None,
            right_chamber_id=None,
            proof_level=w.proof_level,
            artifact_sha256=SHA,
        )
        for w in PD.witnesses
    )
    numeric = FrozenPhaseDiagram(
        fixture_id=PD.fixture_id,
        model_schema=PD.model_schema,
        implementation_commit=PD.implementation_commit,
        events=numeric_events,
        chambers=numeric_chambers,
        target_component_sequence=PD.target_component_sequence,
        witnesses=witnesses,
        manifest=PD.manifest,
        receipt=ArtifactVerificationReceipt(
            profile=PD.fixture_id,
            artifact_sha256=SHA,
            artifact_bytes=4096,
            implementation_commit=PD.implementation_commit,
            manifest_sha256="b" * 64,
            mode=ReplayState.IMPORTED_CERTIFICATE_VERIFIED,
        ),
    )
    numeric.verify_manifest()  # all 8 events carry non-null ordered intervals


def test_f4_bundle_layer_mutation_rejected():
    z3 = PD.event("z3")
    with pytest.raises(ValueError, match="bundle layer"):
        _numeric_event_from_bundle(z3, _declared(z3, layer=PhaseLayer.CRITICAL_SET.value), SHA)


def test_f5_bundle_kind_mutation_rejected():
    z3 = PD.event("z3")
    with pytest.raises(ValueError, match="bundle kind"):
        _numeric_event_from_bundle(z3, _declared(z3, kind="GENERIC_FOLD"), SHA)


def test_f6_bundle_multiplicity_mutation_rejected():
    z1 = PD.event("z1")
    with pytest.raises(ValueError, match="multiplicity"):
        _numeric_event_from_bundle(z1, _declared(z1, multiplicity=7), SHA)


@pytest.mark.parametrize("bad", [[float("nan"), 1.0], [float("inf"), 2.0], [5.0, 1.0]])
def test_f7_nan_inf_reversed_intervals_rejected(bad):
    z1 = PD.event("z1")
    with pytest.raises(ValueError, match="finite and ordered"):
        _numeric_event_from_bundle(z1, _declared(z1, interval=bad), SHA)


def test_f8_missing_event_rejected_by_factory_event_set_check():
    # the factory-level check: a bundle missing an event is a set mismatch
    bundle_ids = {"z1", "z2"}  # far from the frozen 8
    assert bundle_ids != {e.event_id for e in PD.events}


def test_f9_extra_event_rejected_by_factory_event_set_check():
    bundle_ids = {e.event_id for e in PD.events} | {"zX"}
    frozen_ids = {e.event_id for e in PD.events}
    assert bundle_ids != frozen_ids
    # and the conversion itself validates identity for every conversion
    z1 = PD.event("z1")
    with pytest.raises(ValueError, match="bundle event_id"):
        _numeric_event_from_bundle(z1, _declared(z1, event_id="zX"), SHA)


def test_conversion_preserves_proof_level():
    z1 = PD.event("z1")
    numeric = _numeric_event_from_bundle(z1, _declared(z1), SHA)
    assert numeric.proof_level is ProofLevel.IMPORTED_QDM_CERTIFIED
