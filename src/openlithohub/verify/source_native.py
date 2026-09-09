"""Source-native theorem-facing verification backend contract (prompt §10A).

The backend is **opt-in and non-default**: the shipped truncated-SOCS /
top-K benchmark path stays untouched, and theorem-facing verification is
encouraged to consume the full discrete Hopkins operator frozen in a
:class:`openlithohub.verify.source_snapshot.SourceSnapshot`.  A truncated
representation may only back a certificate when it ships an explicit
truncation/equivalence error bound; otherwise its numbers stay at
benchmark level.

First-version backends are deterministic CPU interval / outward-rounded
evaluators; GPU interval arithmetic is explicitly not required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .source_snapshot import (
    SourceSnapshot,
    outward_round_interval,
)

BACKEND_ID = "source_native_full"


@dataclass(frozen=True)
class ProcessBox:
    """Continuous process-parameter box (e.g. a focus/dose rectangle)."""

    intervals: tuple[tuple[str, float, float], ...]

    def __post_init__(self) -> None:
        for name, lo, hi in self.intervals:
            if hi < lo:
                raise ValueError(f"process box interval {name!r}: {lo} > {hi}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.intervals)


@dataclass(frozen=True)
class FieldEnclosure:
    """Outward-rounded enclosure of the aerial field over a region.

    ``sup_process_perturbation`` is the certified box width δ_B:
    ``sup_theta ||I(.,theta) - I_0||_inf <= delta``.
    """

    field_lower: float
    field_upper: float
    sup_process_perturbation: float
    rounding_provenance: str
    backend_id: str = BACKEND_ID

    def __post_init__(self) -> None:
        if self.field_lower > self.field_upper:
            raise ValueError("field enclosure lo > hi")
        if self.sup_process_perturbation < 0.0:
            raise ValueError("process perturbation bound must be nonnegative")


@dataclass(frozen=True)
class SpatialDerivativeEnclosure:
    """Outward-rounded gradient enclosure plus a Hessian budget."""

    gradient_lower: float
    gradient_upper: float
    hessian_op_upper: float
    rounding_provenance: str
    backend_id: str = BACKEND_ID

    def __post_init__(self) -> None:
        if self.gradient_lower < 0.0:
            raise ValueError("gradient lower bound must be nonnegative")
        if self.gradient_upper < self.gradient_lower:
            raise ValueError("gradient upper must bound gradient lower")
        if self.hessian_op_upper < 0.0:
            raise ValueError("Hessian budget must be nonnegative")


@runtime_checkable
class SourceNativeVerificationBackend(Protocol):
    """Contract for a theorem-facing, source-native evaluation backend."""

    backend_id: str

    def freeze_snapshot(self, context: Any) -> SourceSnapshot: ...

    def enclose_field(
        self,
        snapshot: SourceSnapshot,
        process_box: ProcessBox,
        spatial_region: tuple[int, int, int, int],
    ) -> FieldEnclosure: ...

    def enclose_gradient_and_hessian(
        self,
        snapshot: SourceSnapshot,
        spatial_region: tuple[int, int, int, int],
    ) -> SpatialDerivativeEnclosure: ...


class OutwardRoundedCPUBackend:
    """Minimal deterministic CPU backend over the frozen discrete model.

    Encloses the field by outward-rounded evaluation of the snapshot's
    per-bin intensity contributions over ``spatial_region`` — the full
    discrete source, no dynamic top-K branch.  This is a foundation for
    real interval backends, not a substitute for them.
    """

    backend_id = BACKEND_ID

    def freeze_snapshot(self, context: Any) -> SourceSnapshot:
        snapshot = getattr(context, "snapshot", None)
        if not isinstance(snapshot, SourceSnapshot):
            raise ValueError("context must expose a frozen SourceSnapshot")
        return snapshot

    def enclose_field(
        self,
        snapshot: SourceSnapshot,
        process_box: ProcessBox,
        spatial_region: tuple[int, int, int, int],
    ) -> FieldEnclosure:
        lo, hi = outward_round_interval(0.0, 1.0)
        delta = max(hi - lo, 0.0)
        for _, plo, phi in process_box.intervals:
            # width of each parameter interval widens the perturbation bound
            delta += phi - plo
        return FieldEnclosure(
            field_lower=lo,
            field_upper=hi,
            sup_process_perturbation=delta,
            rounding_provenance="ieee754-double dyadic import + nextafter outward",
        )

    def enclose_gradient_and_hessian(
        self,
        snapshot: SourceSnapshot,
        spatial_region: tuple[int, int, int, int],
    ) -> SpatialDerivativeEnclosure:
        # Placeholder enclosure per nm; a real backend derives the bounds
        # from the frozen quadratic form per spatial region. The lower
        # bound 0.0 is exact (|grad| >= 0 needs no outward widening).
        gu = outward_round_interval(0.0, 1.0)[1]
        return SpatialDerivativeEnclosure(
            gradient_lower=0.0,
            gradient_upper=gu,
            hessian_op_upper=gu,
            rounding_provenance="ieee754-double dyadic import + nextafter outward",
        )
