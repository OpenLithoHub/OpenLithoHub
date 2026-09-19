"""P-054 frozen phase-diagram replay — structure, layers, chambers.

Audit §10 event regression set, checkable offline: event order, layer
classification, multiplicity, owner before/after, ownership-invisible
pitchfork semantics, target-events-are-not-critical-events, target
component sequence 1 -> 3 -> 5 -> 4, and fail-closed chamber queries
without the external artifact.
"""

import pytest

from openlithohub.verify.phase_diagram import (
    CriticalSetEventCertificate,
    FrozenPhaseDiagram,
    OwnershipEventCertificate,
    OwnershipInvisibilityWitness,
    PhaseArtifactNotAvailableError,
    PhaseLayer,
    TargetTopologyEventCertificate,
    load_frozen_phase_diagram,
)

PD = load_frozen_phase_diagram("p054-arf37")


def test_frozen_profile_loads_with_imported_frozen_level():
    assert PD.fixture_id == "p054-arf37"
    assert PD.implementation_commit == "348fa5d86d5355465af98e2c4ce3deac60081a4c"
    assert all(event.proof_level.value == "IMPORTED-QDM-CERTIFIED" for event in PD.events)


def test_event_order_and_ids():
    assert [event.event_id for event in PD.events] == [
        "z1",
        "z2",
        "z3",
        "z4",
        "z5",
        "zV",
        "zH",
        "zP",
    ]


def test_layers_are_distinct_typed_objects():
    critical = PD.events_by_layer(PhaseLayer.CRITICAL_SET)
    ownership = PD.events_by_layer(PhaseLayer.OWNERSHIP)
    target = PD.events_by_layer(PhaseLayer.TARGET_TOPOLOGY)
    assert {e.event_id for e in critical} == {"z1", "z2", "z4", "z5"}
    assert {e.event_id for e in ownership} == {"z3"}
    assert {e.event_id for e in target} == {"zV", "zH", "zP"}
    assert all(isinstance(e, CriticalSetEventCertificate) for e in critical)
    assert all(isinstance(e, OwnershipEventCertificate) for e in ownership)
    assert all(isinstance(e, TargetTopologyEventCertificate) for e in target)


def test_z3_is_ownership_crossing_not_target_topology():
    z3 = PD.event("z3")
    assert isinstance(z3, OwnershipEventCertificate)
    assert z3.kind.value == "TRANSVERSE_OWNERSHIP_CROSSING"
    assert (z3.owner_before, z3.owner_after) == ("C", "E")


def test_z5_ownership_invisible_pitchfork_keeps_owner():
    z5 = PD.event("z5")
    assert z5.kind.value == "OWNERSHIP_INVISIBLE_PITCHFORK"
    # layered typing: a critical-set pitchfork carries no owner change
    assert isinstance(z5, CriticalSetEventCertificate)
    assert z5.layer is PhaseLayer.CRITICAL_SET


def test_target_events_are_not_critical_set_events():
    for event_id in ("zV", "zH", "zP"):
        event = PD.event(event_id)
        assert event.layer is PhaseLayer.TARGET_TOPOLOGY
        assert not isinstance(event, (CriticalSetEventCertificate, OwnershipEventCertificate))


def test_target_component_sequence_1_3_5_4():
    assert PD.target_component_sequence == (1, 3, 5, 4)
    counts = [c.target_component_count for c in sorted(PD.chambers, key=lambda c: c.chamber_id)]
    assert counts == [1, 3, 5, 4]


def test_chamber_query_fails_closed_without_external_artifact():
    with pytest.raises(PhaseArtifactNotAvailableError, match="fetch_proof_artifacts"):
        PD.chamber_at_focus(40.0)
    with pytest.raises(PhaseArtifactNotAvailableError):
        PD.target_component_count(279.0)


def test_verify_manifest_rejects_layer_mixing(tmp_path=None):
    from openlithohub.verify.phase_diagram import EventKind

    swapped = FrozenPhaseDiagram(
        fixture_id=PD.fixture_id,
        model_schema=PD.model_schema,
        implementation_commit=PD.implementation_commit,
        events=tuple(
            OwnershipEventCertificate(
                event_id="zP",
                kind=EventKind.TARGET_SADDLE_EVENT,
                focus_interval_nm=None,
                multiplicity=1,
                owner_before=None,
                owner_after=None,
                invisible_to_lower_owner=False,
                proof_level=z.proof_level,
            )
            for z in PD.events
            if z.event_id == "zP"
        ),
        chambers=(),
        target_component_sequence=(),
    )
    with pytest.raises(ValueError, match="belongs to layer TARGET_TOPOLOGY"):
        swapped.verify_manifest()


def test_unknown_profile_fails_closed():
    with pytest.raises(KeyError):
        load_frozen_phase_diagram("p054-made-up")


def test_z5_carries_exactly_one_cross_layer_witness():
    assert len(PD.witnesses) == 1
    witness = PD.witnesses[0]
    assert witness.critical_event_id == "z5"
    # structural values frozen by the external artifact are None until
    # numeric replay; the invariant itself is enforced below
    PD.verify_manifest()  # does not raise


def test_witness_owner_mutation_hard_fails():
    # H8: changing the owner across the cross-layer witness is a firewall
    # violation, not a catalog update
    mutated = OwnershipInvisibilityWitness(
        critical_event_id="z5", owner_before="C", owner_after="D"
    )
    broken = FrozenPhaseDiagram(
        fixture_id=PD.fixture_id,
        model_schema=PD.model_schema,
        implementation_commit=PD.implementation_commit,
        events=PD.events,
        chambers=PD.chambers,
        target_component_sequence=PD.target_component_sequence,
        witnesses=(mutated,),
        manifest=PD.manifest,
    )
    with pytest.raises(ValueError, match="ownership-invisibility broken"):
        broken.verify_manifest()


def test_witness_must_reference_invisible_pitchfork():
    stray = OwnershipInvisibilityWitness(critical_event_id="z3")
    broken = FrozenPhaseDiagram(
        fixture_id=PD.fixture_id,
        model_schema=PD.model_schema,
        implementation_commit=PD.implementation_commit,
        events=PD.events,
        chambers=PD.chambers,
        target_component_sequence=PD.target_component_sequence,
        witnesses=(stray,),
        manifest=PD.manifest,
    )
    with pytest.raises(ValueError, match="OWNERSHIP_INVISIBLE_PITCHFORK"):
        broken.verify_manifest()


def test_missing_witness_hard_fails():
    broken = FrozenPhaseDiagram(
        fixture_id=PD.fixture_id,
        model_schema=PD.model_schema,
        implementation_commit=PD.implementation_commit,
        events=PD.events,
        chambers=PD.chambers,
        target_component_sequence=PD.target_component_sequence,
        witnesses=(),
        manifest=PD.manifest,
    )
    with pytest.raises(ValueError, match="exactly one cross-layer witness"):
        broken.verify_manifest()
