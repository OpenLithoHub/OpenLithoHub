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

import hashlib
import json
import math
import zipfile
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


class ReplayState(str, Enum):
    """Evidence-owned replay state (P-054 re-audit PR-5B/PR-4C).

    The states are produced by *evidence*, never by caller-supplied
    strings:

    - ``STRUCTURE_ONLY`` — catalogs structurally replayed; numeric
      queries are refused;
    - ``IMPORTED_CERTIFICATE_VERIFIED`` — the frozen external artifact
      was fetched, hash/length verified, and its declarations replayed
      against the catalogs (Contract B semantics);
    - ``SOURCE_NATIVE_RECOMPUTED`` — the declarations were *recomputed*
      from the source snapshot / coefficient tensor (Contract A; requires
      a pinned recomputation engine and is not producible yet).
    """

    STRUCTURE_ONLY = "STRUCTURE_ONLY"
    IMPORTED_CERTIFICATE_VERIFIED = "IMPORTED_CERTIFICATE_VERIFIED"
    SOURCE_NATIVE_RECOMPUTED = "SOURCE_NATIVE_RECOMPUTED"


_NUMERIC_REPLAY_STATES = (
    ReplayState.IMPORTED_CERTIFICATE_VERIFIED,
    ReplayState.SOURCE_NATIVE_RECOMPUTED,
)


@dataclass(frozen=True)
class ArtifactVerificationReceipt:
    """Evidence that a specific frozen artifact was verified and replayed.

    A :class:`FrozenPhaseDiagram` may answer numeric queries only when it
    carries a receipt whose identity binds the artifact bytes, the frozen
    model commit and the replay mode.  A manually supplied string can
    never substitute for the receipt.
    """

    profile: str
    artifact_sha256: str
    artifact_bytes: int
    implementation_commit: str
    manifest_sha256: str
    mode: ReplayState
    replay_engine_sha256: str | None = None


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
    statement by itself.  The load-bearing witness is
    ``owner_before == owner_after`` on a certified focus interval
    (``owner_interval_nm``) containing the critical event (PR-5F1): the
    target-topology chamber catalog is *not* an ownership-chamber catalog
    and must not be reused as one.  ``left_chamber_id`` /
    ``right_chamber_id`` remain only for backwards API compatibility.
    Fields frozen by the external artifact stay ``None`` until numeric
    replay.
    """

    critical_event_id: str
    owner_before: str | None = None
    owner_after: str | None = None
    owner_interval_nm: tuple[float, float] | None = None
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
    or verified the frozen artifact's declarations (``IMPORTED_CERTIFICATE_VERIFIED``).
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
    receipt: ArtifactVerificationReceipt | None = None

    @property
    def replay_state(self) -> ReplayState:
        """Derived from the verification receipt — never caller-owned."""
        return self.receipt.mode if self.receipt is not None else ReplayState.STRUCTURE_ONLY

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
        """Structural surface: numeric queries are refused, always.

        Numeric answers live on :class:`VerifiedFrozenPhaseDiagram`,
        produced only by ``load_verified_frozen_phase_diagram()``.  A
        receipt on a plain structural diagram does not upgrade it.
        """
        del z_nm
        raise PhaseArtifactNotAvailableError(
            "this is a structural phase diagram; numeric chamber queries "
            "require the verified frozen artifact — use "
            "load_verified_frozen_phase_diagram()"
        )

    def target_component_count(self, z_nm: float) -> int:
        del z_nm
        raise PhaseArtifactNotAvailableError(
            "this is a structural phase diagram; numeric queries require "
            "load_verified_frozen_phase_diagram()"
        )

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

    def _verify_numeric_replay(self) -> None:
        """Numeric-state obligations (PR-4C): the receipt must bind evidence."""
        receipt = self.receipt
        if receipt is None:  # pragma: no cover — gated by the caller
            raise ValueError("numeric verification requires a receipt")
        if receipt.profile != self.fixture_id:
            raise ValueError(
                f"receipt profile {receipt.profile!r} does not match the "
                f"diagram {self.fixture_id!r}"
            )
        if receipt.implementation_commit != self.implementation_commit:
            raise ValueError(
                "receipt implementation_commit does not match the frozen diagram basis"
            )
        for event in self.events:
            if event.focus_interval_nm is None:
                raise ValueError(f"numeric replay requires a frozen interval for {event.event_id}")
        for chamber in self.chambers:
            if chamber.focus_interval_nm is None:
                raise ValueError(
                    f"numeric replay requires a frozen boundary for {chamber.chamber_id}"
                )
        for witness in self.witnesses:
            if witness.critical_event_id and (
                witness.owner_before is None or witness.owner_after is None
            ):
                raise ValueError(
                    "numeric replay requires non-null owners on witness "
                    f"{witness.critical_event_id}"
                )
            # PR-5F1: the load-bearing ownership witness is the certified
            # focus interval — NOT the target-topology chamber catalog.
            interval = witness.owner_interval_nm
            if (
                interval is None
                or len(interval) != 2
                or not all(math.isfinite(v) for v in interval)
                or interval[0] >= interval[1]
            ):
                raise ValueError(
                    "numeric replay requires a finite ordered ownership "
                    f"interval on witness {witness.critical_event_id}"
                )
            if (
                witness.artifact_sha256 is not None
                and witness.artifact_sha256 != receipt.artifact_sha256
            ):
                raise ValueError(
                    f"witness {witness.critical_event_id} artifact hash differs "
                    "from the verification receipt"
                )

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
        if self.receipt is not None and self.receipt.mode in _NUMERIC_REPLAY_STATES:
            self._verify_numeric_replay()
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
            # PR-5F1: when both the witness ownership interval and the
            # critical event's frozen interval are known, the event must lie
            # INSIDE the ownership-certified interval.  Chamber membership is
            # no longer an ownership criterion (the chamber catalog is
            # target-topology, not ownership).
            if witness.owner_interval_nm is not None and event.focus_interval_nm is not None:
                wlo, whi = witness.owner_interval_nm
                elo, ehi = event.focus_interval_nm
                if not (wlo <= elo and ehi <= whi):
                    raise ValueError(
                        f"witness {witness.critical_event_id}: event interval is "
                        "not contained in the ownership-certified interval"
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


def _tuple_or_none(values: Any) -> tuple[float, float] | None:
    if isinstance(values, (list, tuple)) and len(values) == 2:
        return (float(values[0]), float(values[1]))
    return None


class VerifiedFrozenPhaseDiagram(FrozenPhaseDiagram):
    """Numeric query surface, produced only by the verified factory.

    The base class refuses numeric queries outright; only this subclass —
    which ``load_verified_frozen_phase_diagram()`` returns, receipt-bound —
    can answer them, and still only in an evidence-backed numeric state.
    """

    def chamber_at_focus(self, z_nm: float) -> FocusChamber:
        if self.receipt is None or self.receipt.mode not in _NUMERIC_REPLAY_STATES:
            raise PhaseArtifactNotAvailableError(
                "numeric chamber queries require a verified frozen artifact; "
                "run scripts/fetch_proof_artifacts.py --profile p054-arf37"
            )
        for chamber in self.chambers:
            interval = chamber.focus_interval_nm
            if interval is not None and interval[0] <= z_nm < interval[1]:
                return chamber
        raise PhaseArtifactNotAvailableError("focus does not fall inside any frozen chamber")

    def target_component_count(self, z_nm: float) -> int:
        return self.chamber_at_focus(z_nm).target_component_count or 0


def _numeric_event_from_bundle(
    structural_event: Any,
    declared: dict[str, Any],
    artifact_sha256: str,
) -> Any:
    """Total structural+bundle -> numeric event conversion (PR-5C).

    Covers all three layers.  The bundle declaration must agree with the
    structural catalog on identity (event_id, layer, kind, multiplicity)
    and carry a finite, ordered focus interval — any mismatch is a
    corrupted proof package, not a partial replay.
    """
    if declared.get("event_id") != structural_event.event_id:
        raise ValueError(
            f"bundle event_id {declared.get('event_id')!r} != structural "
            f"{structural_event.event_id!r}"
        )
    if declared.get("layer") != structural_event.layer.value:
        raise ValueError(
            f"event {structural_event.event_id}: bundle layer "
            f"{declared.get('layer')!r} != structural "
            f"{structural_event.layer.value}"
        )
    if declared.get("kind") != structural_event.kind.value:
        raise ValueError(
            f"event {structural_event.event_id}: bundle kind "
            f"{declared.get('kind')!r} != structural {structural_event.kind.value}"
        )
    if int(declared.get("multiplicity", -1)) != structural_event.multiplicity:
        raise ValueError(
            f"event {structural_event.event_id}: bundle multiplicity "
            f"{declared.get('multiplicity')!r} != structural "
            f"{structural_event.multiplicity}"
        )
    interval = declared.get("focus_interval_nm")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in interval)
        or interval[0] >= interval[1]
    ):
        raise ValueError(
            f"event {structural_event.event_id}: bundle focus interval "
            f"{interval!r} must be finite and ordered"
        )
    common = dict(
        event_id=structural_event.event_id,
        kind=structural_event.kind,
        focus_interval_nm=(float(interval[0]), float(interval[1])),
        multiplicity=structural_event.multiplicity,
        proof_level=structural_event.proof_level,
        artifact_sha256=artifact_sha256,
    )
    if structural_event.layer is PhaseLayer.CRITICAL_SET:
        if not isinstance(structural_event, CriticalSetEventCertificate):
            raise ValueError("critical-set layer with mismatched certificate type")
        return CriticalSetEventCertificate(
            transversality_lower=structural_event.transversality_lower,
            morse_signature=structural_event.morse_signature,
            **common,
        )
    if structural_event.layer is PhaseLayer.OWNERSHIP:
        if not isinstance(structural_event, OwnershipEventCertificate):
            raise ValueError("ownership layer with mismatched certificate type")
        return OwnershipEventCertificate(
            owner_before=structural_event.owner_before,
            owner_after=structural_event.owner_after,
            invisible_to_lower_owner=structural_event.invisible_to_lower_owner,
            **common,
        )
    if not isinstance(structural_event, TargetTopologyEventCertificate):
        raise ValueError("target-topology layer with mismatched certificate type")
    return TargetTopologyEventCertificate(
        component_count_before=structural_event.component_count_before,
        component_count_after=structural_event.component_count_after,
        **common,
    )


MAX_BUNDLE_MEMBERS = 64
MAX_BUNDLE_MEMBER_BYTES = 1 << 30
# Chosen from the frozen artifact envelope: the P-054 bundle is a ~100 MB
# ZIP whose uncompressed members stay well under 2 GiB.  Rejected before
# any member body is read.
MAX_BUNDLE_TOTAL_UNCOMPRESSED_BYTES = 2 << 30


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key in bundle: {key!r}")
        seen[key] = value
    return seen


def load_replay_bundle(artifact: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Canonical frozen-bundle parser (PR-5D2) — the ONLY bundle entry.

    Hardening: duplicate ZIP member names, member-count and per-member
    uncompressed-size caps, and duplicate JSON keys are rejected before
    anything is trusted.  Producer, shard and factory all parse through
    this function so no parallel, weaker parser can exist.
    """
    with zipfile.ZipFile(artifact) as zf:
        names = zf.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member names in the frozen bundle")
        if len(names) > MAX_BUNDLE_MEMBERS:
            raise ValueError(
                f"frozen bundle exceeds the member limit: {len(names)} > {MAX_BUNDLE_MEMBERS}"
            )
        total_uncompressed = sum(info.file_size for info in zf.infolist())
        if total_uncompressed > MAX_BUNDLE_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError(
                "frozen bundle exceeds the total uncompressed limit: "
                f"{total_uncompressed} > {MAX_BUNDLE_TOTAL_UNCOMPRESSED_BYTES}"
            )
        for info in zf.infolist():
            if info.file_size > MAX_BUNDLE_MEMBER_BYTES:
                raise ValueError(
                    f"bundle member {info.filename!r} exceeds the per-member "
                    f"size limit ({info.file_size} > {MAX_BUNDLE_MEMBER_BYTES})"
                )
        if "replay_manifest.json" not in names:
            raise ValueError("frozen bundle lacks replay_manifest.json")
        replay = json.loads(
            zf.read("replay_manifest.json"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        members = {name: zf.read(name) for name in names if name != "replay_manifest.json"}
    return replay, members


def _verify_bundle_semantics(
    replay: dict[str, Any],
    members: dict[str, bytes],
    structure: FrozenPhaseDiagram,
    fixture: dict[str, Any],
    artifact_sha256: str,
) -> None:
    """Canonical Contract-B semantic checks (PR-5D).

    Both the release producer and the scheduled shard depend on this
    function, so a passing factory implies the shard's semantic gates:
    complete fixture identity, member hashes declared by the bundle
    manifest, collection closure against the frozen catalogs, and
    layer-specific declarations.
    """
    identity_fields = (
        ("wavelength_nm", "wavelength_nm"),
        ("na", "na"),
        ("sigma", "sigma"),
        ("grid_shape", "grid_shape"),
        ("pixel_size_nm", "pixel_size_nm"),
        ("pupil_support_count", "pupil_support_count"),
        ("boundary", "boundary"),
    )
    for replay_key, fixture_key in identity_fields:
        if replay.get(replay_key) != fixture.get(fixture_key):
            raise ValueError(
                f"bundle {replay_key} {replay.get(replay_key)!r} != frozen "
                f"fixture {fixture.get(fixture_key)!r}"
            )
    # PR-5D2: exact canonical member identity — the bundle manifest names
    # the one member carrying each payload, and exactly that member is
    # hashed (no substring guessing).
    if replay.get("model_schema") != structure.model_schema:
        raise ValueError(
            f"bundle model_schema {replay.get('model_schema')!r} "
            f"!= frozen model schema {structure.model_schema!r}"
        )
    for hash_key, file_key in (
        ("source_snapshot_sha256", "source_snapshot_file"),
        ("coefficient_tensor_sha256", "coefficient_file"),
        ("mask_sha256", "mask_file"),
    ):
        declared = replay.get(hash_key)
        member_name = replay.get(file_key)
        if not declared or not member_name:
            raise ValueError(
                f"bundle manifest lacks {hash_key}/{file_key}: exact member identity is required"
            )
        if member_name not in members:
            raise ValueError(f"bundle member {member_name!r} is missing")
        if hashlib.sha256(members[member_name]).hexdigest() != declared:
            raise ValueError(f"bundle member {member_name!r} hash does not match {hash_key}")
    # PR-5D2: exact event SEQUENCE equality and globally ordered focus
    # intervals — a permuted or globally inconsistent bundle cannot pass.
    catalog = structure.events
    raw_events = replay.get("events", [])
    raw_ids = [e.get("event_id") for e in raw_events]
    if raw_ids != [e.event_id for e in catalog]:
        raise ValueError(
            f"bundle event sequence differs from the frozen catalog (bundle: {raw_ids})"
        )
    catalog_by_id = {e.event_id: e for e in catalog}
    previous_hi: float | None = None
    for event_id, declared in zip(raw_ids, raw_events, strict=True):
        structural = catalog_by_id[event_id]
        if declared.get("layer") != structural.layer.value:
            raise ValueError(f"bundle layer mismatch for {event_id}")
        if declared.get("kind") != structural.kind.value:
            raise ValueError(f"bundle kind mismatch for {event_id}")
        if declared.get("multiplicity") != structural.multiplicity:
            raise ValueError(f"bundle multiplicity mismatch for {event_id}")
        if isinstance(structural, OwnershipEventCertificate) and (
            declared.get("owner_before") != structural.owner_before
            or declared.get("owner_after") != structural.owner_after
        ):
            raise ValueError(f"bundle ownership mismatch for {event_id}")
        if isinstance(structural, TargetTopologyEventCertificate) and (
            declared.get("component_count_before") != structural.component_count_before
            or declared.get("component_count_after") != structural.component_count_after
        ):
            raise ValueError(f"bundle component counts mismatch for {event_id}")
        interval = declared.get("focus_interval_nm")
        interval_bad = (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in interval)
            or interval[0] >= interval[1]
        )
        if interval_bad:
            raise ValueError(f"event {event_id}: bundle focus interval must be finite and ordered")
        if previous_hi is not None and interval[0] < previous_hi:
            raise ValueError(
                f"event {event_id}: focus interval starts before the previous "
                "event ends (global focus ordering violated)"
            )
        previous_hi = interval[1]
    raw_chambers: list[dict[str, Any]] = list(replay.get("chambers", []))
    raw_chamber_ids = [c.get("chamber_id") for c in raw_chambers]
    if raw_chamber_ids != [c.chamber_id for c in structure.chambers]:
        raise ValueError(
            f"bundle chamber sequence differs from the frozen catalog (bundle: {raw_chamber_ids})"
        )
    catalog_chambers = {c.chamber_id: c for c in structure.chambers}
    for declared in raw_chambers:
        structural_chamber = catalog_chambers[declared["chamber_id"]]
        interval = declared.get("focus_interval_nm")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in interval)
            or interval[0] >= interval[1]
        ):
            raise ValueError(
                f"bundle chamber {declared['chamber_id']!r} interval must be finite and ordered"
            )
        if list(declared.get("bounded_by_events", ())) != list(
            structural_chamber.bounded_by_events
        ):
            raise ValueError(f"bundle bounded_by_events mismatch for {declared['chamber_id']!r}")
        if declared.get("target_component_count") != structural_chamber.target_component_count:
            raise ValueError(
                f"bundle target component count mismatch for {declared['chamber_id']!r}"
            )
    raw_witness_ids = [w.get("critical_event_id") for w in replay.get("witnesses", [])]
    catalog_witness_ids = [w.critical_event_id for w in structure.witnesses]
    if sorted(raw_witness_ids) != sorted(catalog_witness_ids):
        raise ValueError("bundle witness set differs from the frozen catalog")


def load_verified_frozen_phase_diagram(
    profile: str = "p054-arf37",
    *,
    artifact_path: Path | None = None,
    root: Path | None = None,
) -> FrozenPhaseDiagram:
    """Verify the frozen artifact and return a numeric-queryable diagram.

    Contract B (re-audit PR-5B): this performs **imported frozen
    certificate verification** — the artifact's identity and declarations
    are verified against the repository catalogs.  It is *not*
    source-native recomputation; producing
    ``ReplayState.SOURCE_NATIVE_RECOMPUTED`` requires a pinned
    recomputation engine and is intentionally unavailable.

    Evidence chain: registry identity -> artifact sha256 + byte length ->
    bundle manifest vs repo frozen manifest -> numeric fields -> receipt.
    """
    base = (root or Path(__file__).resolve().parents[3]) / "proof_artifacts"
    registry = json.loads((base / "registry.json").read_text())
    entry = next(
        (a for a in registry.get("artifacts", []) if profile in a.get("profiles", ["p054-arf37"])),
        None,
    )
    if entry is None:
        raise PhaseArtifactNotAvailableError(f"no registry entry for {profile!r}")
    expected_sha = entry.get("sha256")
    expected_bytes = entry.get("bytes")
    if not expected_sha or not expected_bytes:
        raise PhaseArtifactNotAvailableError(
            f"{profile} artifact is UNFETCHED: the frozen publication has not "
            "provided its content identity (run "
            "scripts/fetch_proof_artifacts.py once it has)"
        )
    artifact = artifact_path or (base / entry.get("destination", "p054/external") / entry["name"])
    if not artifact.exists():
        raise PhaseArtifactNotAvailableError(f"frozen artifact missing at {artifact}")
    raw = artifact.read_bytes()
    if len(raw) != expected_bytes:
        raise ValueError(f"artifact byte length {len(raw)} != registry {expected_bytes}")
    artifact_sha = hashlib.sha256(raw).hexdigest()
    if artifact_sha != expected_sha:
        raise ValueError(f"artifact sha256 {artifact_sha} != registry {expected_sha}")

    structure = load_frozen_phase_diagram(profile, root=root)
    fixture = json.loads((base / "p054" / "fixture.json").read_text())
    replay, members = load_replay_bundle(artifact)
    if replay.get("fixture_id") != profile:
        raise ValueError("bundle fixture_id mismatch")
    if replay.get("model_schema") != structure.model_schema:
        raise ValueError(
            f"bundle model_schema {replay.get('model_schema')!r} "
            f"!= frozen model schema {structure.model_schema!r}"
        )
    if replay.get("implementation_commit") != structure.implementation_commit:
        raise ValueError("bundle implementation_commit mismatch")
    if replay.get("proof_level") != ProofLevel.IMPORTED_QDM_CERTIFIED.value:
        raise ValueError("bundle proof level is downgraded")
    # PR-5D: complete frozen fixture identity, verified here (not only in
    # the shard) so a passing factory implies a passing shard fixture gate
    _verify_bundle_semantics(replay, members, structure, fixture, artifact_sha)

    bundle_events = {e["event_id"]: e for e in replay.get("events", [])}
    if set(bundle_events) != {e.event_id for e in structure.events}:
        raise ValueError(
            f"bundle event set differs from the frozen catalog (bundle: {sorted(bundle_events)})"
        )
    numeric_events = [
        _numeric_event_from_bundle(event, bundle_events[event.event_id], artifact_sha)
        for event in structure.events
    ]

    bundle_chambers = {c["chamber_id"]: c for c in replay.get("chambers", [])}
    numeric_chambers = []
    for chamber in structure.chambers:
        declared = bundle_chambers.get(chamber.chamber_id)
        if declared is None or declared.get("focus_interval_nm") is None:
            raise ValueError(f"numeric replay requires a frozen boundary for {chamber.chamber_id}")
        lo, hi = declared["focus_interval_nm"]
        numeric_chambers.append(
            FocusChamber(
                chamber_id=chamber.chamber_id,
                focus_interval_nm=(lo, hi),
                lower_owner=declared.get("lower_owner"),
                upper_owner=declared.get("upper_owner"),
                target_component_count=declared.get(
                    "target_component_count", chamber.target_component_count
                ),
                bounded_by_events=chamber.bounded_by_events,
            )
        )

    bundle_witnesses = {w["critical_event_id"]: w for w in replay.get("witnesses", [])}
    numeric_witnesses = tuple(
        OwnershipInvisibilityWitness(
            critical_event_id=w.critical_event_id,
            owner_before=bundle_witnesses.get(w.critical_event_id, {}).get("owner_before"),
            owner_after=bundle_witnesses.get(w.critical_event_id, {}).get("owner_after"),
            owner_interval_nm=_tuple_or_none(
                bundle_witnesses.get(w.critical_event_id, {}).get("owner_interval_nm")
            ),
            left_chamber_id=bundle_witnesses.get(w.critical_event_id, {}).get("left_chamber_id"),
            right_chamber_id=bundle_witnesses.get(w.critical_event_id, {}).get("right_chamber_id"),
            proof_level=w.proof_level,
            artifact_sha256=artifact_sha,
        )
        for w in structure.witnesses
    )

    receipt = ArtifactVerificationReceipt(
        profile=profile,
        artifact_sha256=artifact_sha,
        artifact_bytes=len(raw),
        implementation_commit=structure.implementation_commit,
        # real provenance hashes, from bytes (PR-5C): the repo manifest
        # file and the live replay engine (this module)
        manifest_sha256=hashlib.sha256((base / "p054" / "manifest.json").read_bytes()).hexdigest(),
        replay_engine_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mode=ReplayState.IMPORTED_CERTIFICATE_VERIFIED,
    )
    numeric = VerifiedFrozenPhaseDiagram(
        fixture_id=structure.fixture_id,
        model_schema=structure.model_schema,
        implementation_commit=structure.implementation_commit,
        events=tuple(numeric_events),
        chambers=tuple(numeric_chambers),
        target_component_sequence=structure.target_component_sequence,
        witnesses=numeric_witnesses,
        manifest=structure.manifest,
        receipt=receipt,
    )
    numeric.verify_manifest()
    return numeric


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
            owner_interval_nm=_tuple_or_none(w.get("owner_interval_nm")),
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
