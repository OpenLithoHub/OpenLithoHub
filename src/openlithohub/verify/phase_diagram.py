"""Typed phase-diagram verification API for frozen finite-model proofs.

P-054 repo integration (audit §4/§5, P1.1/P1.2/P1.7/P1.8): the paper's
central structural result is that three event layers are **distinct
objects** —

- ``CRITICAL_SET``: bifurcations of the spatial critical-point set;
- ``OWNERSHIP``: changes of the lower/upper critical-value owners;
- ``TARGET_TOPOLOGY``: topology changes of a fixed-threshold contour.

This module makes that separation a *type-level invariant*: each layer has
its own certificate type, cross-layer mixing is a constructor error, and
the public surface is a frozen-proof **query** API
(``load_frozen_phase_diagram``) — replaying what P-054 certified, not
re-certifying arbitrary inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .types import ProofLevel


class PhaseLayer(str, Enum):
    CRITICAL_SET = "CRITICAL_SET"
    OWNERSHIP = "OWNERSHIP"
    TARGET_TOPOLOGY = "TARGET_TOPOLOGY"


class EventKind(str, Enum):
    GENERIC_FOLD = "GENERIC_FOLD"
    TERMINAL_FOLD_CLIFF = "TERMINAL_FOLD_CLIFF"
    TRANSVERSE_OWNERSHIP_CROSSING = "TRANSVERSE_OWNERSHIP_CROSSING"
    REFLECTION_PITCHFORK = "REFLECTION_PITCHFORK"
    OWNERSHIP_INVISIBLE_PITCHFORK = "OWNERSHIP_INVISIBLE_PITCHFORK"
    TARGET_MAX_EVENT = "TARGET_MAX_EVENT"
    TARGET_SADDLE_EVENT = "TARGET_SADDLE_EVENT"


_LAYER_FOR_KIND: dict[EventKind, PhaseLayer] = {
    EventKind.GENERIC_FOLD: PhaseLayer.CRITICAL_SET,
    EventKind.REFLECTION_PITCHFORK: PhaseLayer.CRITICAL_SET,
    EventKind.OWNERSHIP_INVISIBLE_PITCHFORK: PhaseLayer.CRITICAL_SET,
    EventKind.TERMINAL_FOLD_CLIFF: PhaseLayer.OWNERSHIP,
    EventKind.TRANSVERSE_OWNERSHIP_CROSSING: PhaseLayer.OWNERSHIP,
    EventKind.TARGET_MAX_EVENT: PhaseLayer.TARGET_TOPOLOGY,
    EventKind.TARGET_SADDLE_EVENT: PhaseLayer.TARGET_TOPOLOGY,
}


class PhaseArtifactNotAvailableError(RuntimeError):
    """A frozen numeric value lives in the external artifact, which is absent."""


@dataclass(frozen=True)
class CriticalSetEventCertificate:
    """A bifurcation of the critical-point set itself."""

    event_id: str
    kind: EventKind
    focus_interval_nm: tuple[float, float] | None
    multiplicity: int
    transversality_lower: float | None
    morse_signature: str | None
    proof_level: ProofLevel
    artifact_sha256: str | None = None

    layer: PhaseLayer = PhaseLayer.CRITICAL_SET


@dataclass(frozen=True)
class OwnershipEventCertificate:
    """A change of critical-value ownership — never a topology event."""

    event_id: str
    kind: EventKind
    focus_interval_nm: tuple[float, float] | None
    multiplicity: int
    owner_before: str | None
    owner_after: str | None
    invisible_to_lower_owner: bool
    proof_level: ProofLevel
    artifact_sha256: str | None = None

    layer: PhaseLayer = PhaseLayer.OWNERSHIP


@dataclass(frozen=True)
class TargetTopologyEventCertificate:
    """A fixed-target contour surgery — never a critical-set bifurcation."""

    event_id: str
    kind: EventKind
    focus_interval_nm: tuple[float, float] | None
    multiplicity: int
    component_count_before: int | None
    component_count_after: int | None
    proof_level: ProofLevel
    artifact_sha256: str | None = None

    layer: PhaseLayer = PhaseLayer.TARGET_TOPOLOGY


@dataclass(frozen=True)
class FocusChamber:
    """An event-free focus interval of the frozen fixture."""

    chamber_id: str
    focus_interval_nm: tuple[float, float] | None
    lower_owner: str | None
    upper_owner: str | None
    target_component_count: int | None
    bounded_by_events: tuple[str, ...] = ()

    layer: PhaseLayer = PhaseLayer.OWNERSHIP


@dataclass(frozen=True)
class OwnershipInvisibilityWitness:
    """Cross-layer witness: a critical-set pitchfork keeps the owner.

    The correct ontology is ``critical-set event + witness => ownership
    invisibility`` — the critical-set event never becomes an ownership
    statement by itself.  ``owner_before == owner_after`` is the theorem;
    chamber adjacency across the event is the structural witness.  Fields
    frozen by the external artifact stay ``None`` until numeric replay.
    """

    critical_event_id: str
    owner_before: str | None = None
    owner_after: str | None = None
    left_chamber_id: str | None = None
    right_chamber_id: str | None = None
    proof_level: ProofLevel = ProofLevel.IMPORTED_QDM_CERTIFIED
    artifact_sha256: str | None = None


@dataclass(frozen=True)
class PhaseDiagramCertificate:
    """Frozen phase-diagram certificate: proof level + artifact identity."""

    fixture_id: str
    model_schema: str
    implementation_commit: str
    proof_level: ProofLevel
    artifact_sha256: str | None = None


@dataclass(frozen=True)
class FrozenPhaseDiagram:
    """Query surface over the frozen P-054 stratification.

    ``replay_state`` separates two facts that must never be conflated
    (P-054 re-audit, PR-4B): the paper's declared claim level stays
    ``IMPORTED-QDM-CERTIFIED`` regardless, but the *repository runtime*
    has either only structurally replayed the catalogs (``STRUCTURE_ONLY``)
    or verified the numeric frozen artifact (``ARTIFACT_VERIFIED_NUMERIC``).
    Numeric chamber queries answer only in the latter state.
    """

    fixture_id: str
    model_schema: str
    implementation_commit: str
    events: tuple[
        CriticalSetEventCertificate | OwnershipEventCertificate | TargetTopologyEventCertificate,
        ...,
    ]
    chambers: tuple[FocusChamber, ...]
    target_component_sequence: tuple[int, ...]
    witnesses: tuple[OwnershipInvisibilityWitness, ...] = ()
    manifest: dict[str, Any] = field(default_factory=dict)
    replay_state: str = "STRUCTURE_ONLY"

    def events_by_layer(
        self, layer: PhaseLayer
    ) -> tuple[
        CriticalSetEventCertificate | OwnershipEventCertificate | TargetTopologyEventCertificate,
        ...,
    ]:
        return tuple(event for event in self.events if event.layer is layer)

    def event(self, event_id: str) -> Any:
        for event in self.events:
            if event.event_id == event_id:
                return event
        raise KeyError(f"unknown event {event_id!r}")

    def chamber_at_focus(self, z_nm: float) -> FocusChamber:
        """The event-free chamber containing ``z_nm``.

        Requires the frozen numeric chamber boundaries from the external
        artifact; without it this is a fail-closed error, never a guess.
        A STRUCTURE_ONLY diagram must never answer a numeric query even if
        someone hand-injects intervals into a copy — use the
        ``ARTIFACT_VERIFIED_NUMERIC`` state produced by the replay shard.
        """
        if self.replay_state != "ARTIFACT_VERIFIED_NUMERIC":
            raise PhaseArtifactNotAvailableError(
                "numeric chamber queries require the verified frozen artifact "
                "(replay_state=STRUCTURE_ONLY); run "
                "scripts/fetch_proof_artifacts.py --profile p054-arf37"
            )
        for chamber in self.chambers:
            interval = chamber.focus_interval_nm
            if interval is not None and interval[0] <= z_nm < interval[1]:
                return chamber
        raise PhaseArtifactNotAvailableError(
            "chamber focus boundaries live in the external frozen artifact; "
            "run scripts/fetch_proof_artifacts.py (proof replay hard-fails "
            "without it)"
        )

    def target_component_count(self, z_nm: float) -> int:
        return self.chamber_at_focus(z_nm).target_component_count or 0

    def verify_manifest(self) -> None:
        """Structural replay: ordering, layers, ownership invariants."""
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate event ids in the frozen catalog")
        for event in self.events:
            expected_layer = _LAYER_FOR_KIND[event.kind]
            if event.layer is not expected_layer:
                raise ValueError(
                    f"event {event.event_id}: kind {event.kind.value} belongs to "
                    f"layer {expected_layer.value}, catalog says {event.layer.value}"
                )
        # ownership-invisible pitchfork must not change owners
        for event in self.events:
            if (
                event.kind is EventKind.OWNERSHIP_INVISIBLE_PITCHFORK
                and isinstance(event, OwnershipEventCertificate)
                and event.owner_before != event.owner_after
            ):
                raise ValueError(
                    f"event {event.event_id}: an ownership-invisible pitchfork must keep the owner"
                )
        # target component sequence (P-054: 1 -> 3 -> 5 -> 4)
        sequence = tuple(
            chamber.target_component_count or 0
            for chamber in sorted(self.chambers, key=lambda c: c.chamber_id)
        )
        if any(sequence) and sequence != self.target_component_sequence:
            raise ValueError(
                f"target component sequence {sequence} contradicts the frozen "
                f"sequence {self.target_component_sequence}"
            )
        self._verify_ownership_invisibility_witnesses()

    def _verify_ownership_invisibility_witnesses(self) -> None:
        """Cross-layer witness firewall (PR-4B).

        Ontology: ``critical-set event + cross-layer witness => ownership
        invisibility`` — the critical-set event itself never carries
        ownership fields.  Every ``OWNERSHIP_INVISIBLE_PITCHFORK`` must have
        exactly one witness; the witness must keep the owner, reference
        existing chambers adjacent across that event, and carry artifact
        identity matching the replay bundle.
        """
        invisible_events = [
            event for event in self.events if event.kind is EventKind.OWNERSHIP_INVISIBLE_PITCHFORK
        ]
        invisible_ids = {event.event_id for event in invisible_events}
        # every witness must point at an OWNERSHIP_INVISIBLE_PITCHFORK
        for witness in self.witnesses:
            if witness.critical_event_id not in invisible_ids:
                raise ValueError(
                    f"witness {witness.critical_event_id} does not refer to an "
                    "OWNERSHIP_INVISIBLE_PITCHFORK event"
                )
        by_event: dict[str, list[OwnershipInvisibilityWitness]] = {}
        for witness in self.witnesses:
            by_event.setdefault(witness.critical_event_id, []).append(witness)
        chamber_ids = {chamber.chamber_id for chamber in self.chambers}
        for event in invisible_events:
            witnesses = by_event.get(event.event_id, [])
            if len(witnesses) != 1:
                raise ValueError(
                    f"event {event.event_id}: an ownership-invisible pitchfork "
                    f"requires exactly one cross-layer witness, found {len(witnesses)}"
                )
            witness = witnesses[0]
            if (
                witness.owner_before is not None
                and witness.owner_after is not None
                and witness.owner_before != witness.owner_after
            ):
                raise ValueError(
                    f"witness {witness.critical_event_id}: ownership-invisibility "
                    f"broken ({witness.owner_before} -> {witness.owner_after})"
                )
            if witness.left_chamber_id is not None and witness.right_chamber_id is not None:
                for chamber_id in (witness.left_chamber_id, witness.right_chamber_id):
                    if chamber_id not in chamber_ids:
                        raise ValueError(
                            f"witness {witness.critical_event_id} references "
                            f"unknown chamber {chamber_id!r}"
                        )
                    chamber = next(c for c in self.chambers if c.chamber_id == chamber_id)
                    if event.event_id not in chamber.bounded_by_events:
                        raise ValueError(
                            f"witness {witness.critical_event_id}: chamber "
                            f"{chamber_id!r} is not adjacent across the event"
                        )


def _event_from_catalog(entry: dict[str, Any], proof_level: ProofLevel) -> Any:
    layer = PhaseLayer(entry["layer"])
    kind = EventKind(entry["kind"])
    if _LAYER_FOR_KIND[kind] is not layer:
        raise ValueError(
            f"event {entry.get('event_id')}: layer/kind mismatch ({layer.value}/{kind.value})"
        )
    interval = entry.get("focus_interval_nm")
    focus = (
        (interval[0], interval[1]) if isinstance(interval, list) and len(interval) == 2 else None
    )
    common = dict(
        event_id=entry["event_id"],
        kind=kind,
        focus_interval_nm=focus,
        multiplicity=int(entry.get("multiplicity", 1)),
        proof_level=proof_level,
        artifact_sha256=entry.get("artifact_sha256"),
    )
    if layer is PhaseLayer.CRITICAL_SET:
        return CriticalSetEventCertificate(
            transversality_lower=entry.get("transversality_lower"),
            morse_signature=entry.get("morse_signature"),
            **common,
        )
    if layer is PhaseLayer.OWNERSHIP:
        return OwnershipEventCertificate(
            owner_before=entry.get("owner_before"),
            owner_after=entry.get("owner_after"),
            invisible_to_lower_owner=kind is EventKind.OWNERSHIP_INVISIBLE_PITCHFORK,
            **common,
        )
    return TargetTopologyEventCertificate(
        component_count_before=entry.get("component_count_before"),
        component_count_after=entry.get("component_count_after"),
        **common,
    )


def load_frozen_phase_diagram(
    profile: str = "p054-arf37",
    *,
    root: Path | None = None,
) -> FrozenPhaseDiagram:
    """Load and structurally verify a frozen phase-diagram profile.

    Offline replay: validates the manifest, fixture honesty flags, event
    layer/type invariants and the frozen target component sequence.
    Numeric focus intervals live in the external artifact and surface as
    :class:`PhaseArtifactNotAvailableError` on chamber queries until it is
    fetched.
    """
    base = (root or Path(__file__).resolve().parents[3]) / "proof_artifacts" / "p054"
    if profile != "p054-arf37":
        raise KeyError(f"unknown frozen phase-diagram profile {profile!r}")

    manifest = json.loads((base / "manifest.json").read_text())
    fixture = json.loads((base / "fixture.json").read_text())
    catalog = json.loads((base / "event_catalog.json").read_text())
    chambers_doc = json.loads((base / "chamber_catalog.json").read_text())

    if fixture.get("fixture_id") != profile:
        raise ValueError("fixture id does not match the requested profile")
    honesty = fixture.get("honesty", {})
    if honesty.get("foundry_calibrated") or honesty.get("wafer_process_qualified"):
        raise ValueError(
            "the frozen profile must not claim foundry calibration or wafer qualification"
        )
    if honesty.get("continuous_hopkins_equivalent"):
        raise ValueError("the frozen profile must not claim continuous Hopkins equivalence")

    proof_level = ProofLevel(
        manifest["declared_proof_levels"]["finite_declared_model_phase_diagram"]
    )
    if proof_level is not ProofLevel.IMPORTED_QDM_CERTIFIED:
        raise ValueError("a frozen phase diagram replays as an imported frozen certificate")

    events = tuple(_event_from_catalog(entry, proof_level) for entry in catalog.get("events", []))
    chambers = tuple(
        FocusChamber(
            chamber_id=entry["chamber_id"],
            focus_interval_nm=(
                (entry["focus_interval_nm"][0], entry["focus_interval_nm"][1])
                if entry.get("focus_interval_nm")
                else None
            ),
            lower_owner=entry.get("lower_owner"),
            upper_owner=entry.get("upper_owner"),
            target_component_count=entry.get("target_component_count"),
            bounded_by_events=tuple(entry.get("bounded_by_events", ())),
        )
        for entry in chambers_doc.get("chambers", [])
    )
    witnesses = tuple(
        OwnershipInvisibilityWitness(
            critical_event_id=w["critical_event_id"],
            owner_before=w.get("owner_before"),
            owner_after=w.get("owner_after"),
            left_chamber_id=w.get("left_chamber_id"),
            right_chamber_id=w.get("right_chamber_id"),
            artifact_sha256=w.get("artifact_sha256"),
        )
        for w in chambers_doc.get("witnesses", [])
    )
    diagram = FrozenPhaseDiagram(
        fixture_id=profile,
        model_schema=manifest["model_schema"],
        implementation_commit=manifest["implementation_commit"],
        events=events,
        chambers=chambers,
        target_component_sequence=tuple(chambers_doc.get("target_component_sequence", ())),
        witnesses=witnesses,
        manifest=manifest,
    )
    diagram.verify_manifest()
    return diagram
