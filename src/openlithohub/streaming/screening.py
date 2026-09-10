"""Fail-closed pre-forward tile screening for streaming verification.

A screening policy runs before ``TileSource.read_window`` and before the
expensive forward model.  It may skip a tile only when it returns a certified
``SCREENED_OUT`` decision with an exact trusted-core fill value.

The built-in :class:`ExactEmptyContextScreeningPolicy` is deliberately
conservative.  It screens an exact-vector source only when:

* the current core+halo read window contains no physical mask runs;
* the caller explicitly declares that the context/halo is certified for the
  forward model; and
* the caller explicitly declares that all-zero context has the stated constant
  trusted-core response.

This is an architecture hook, not a universal Hopkins empty-tile theorem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .core_halo import TileRequest

ScreenStatus = Literal["ACTIVE", "SCREENED_OUT", "INCONCLUSIVE"]


@dataclass(frozen=True)
class TileScreenDecision:
    status: ScreenStatus
    reason: str
    certified: bool = False
    fill_value: float | None = None
    certifies_verifiers: bool = False
    verification_upper_bound: float | None = None
    certificate_ref: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status == "SCREENED_OUT":
            if not self.certified:
                raise ValueError("SCREENED_OUT requires a certified decision")
            if self.fill_value is None:
                raise ValueError("SCREENED_OUT requires an exact fill_value")
            if self.certifies_verifiers and self.verification_upper_bound is None:
                raise ValueError("verifier-certified screening requires verification_upper_bound")


@runtime_checkable
class TileScreeningPolicy(Protocol):
    name: str

    def screen(self, source: Any, request: TileRequest) -> TileScreenDecision: ...


class ExactEmptyContextScreeningPolicy:
    """Screen exact-vector tiles whose complete certified read context is empty."""

    name = "exact-empty-context"

    def __init__(
        self,
        *,
        context_certified: bool,
        zero_response_certified: bool,
        fill_value: float = 0.0,
        model_provenance: str,
        certifies_verifiers: bool = False,
        verification_upper_bound: float | None = None,
    ) -> None:
        self.context_certified = bool(context_certified)
        self.zero_response_certified = bool(zero_response_certified)
        self.fill_value = float(fill_value)
        self.model_provenance = str(model_provenance)
        self.certifies_verifiers = bool(certifies_verifiers)
        self.verification_upper_bound = verification_upper_bound

    def screen(self, source: Any, request: TileRequest) -> TileScreenDecision:
        if not self.context_certified:
            return TileScreenDecision(
                status="INCONCLUSIVE",
                reason="context/halo is not certified for pre-forward screening",
            )
        if not self.zero_response_certified:
            return TileScreenDecision(
                status="INCONCLUSIVE",
                reason="zero-input trusted-core response is not certified",
            )

        iter_runs = getattr(source, "iter_runs_for_bbox", None)
        if not callable(iter_runs):
            return TileScreenDecision(
                status="INCONCLUSIVE",
                reason="source has no exact run-query interface",
            )

        bbox = request.read_bbox
        runs = iter_runs((bbox.x0, bbox.y0, bbox.x1, bbox.y1))
        first = next(iter(runs), None)
        if first is not None:
            return TileScreenDecision(
                status="ACTIVE",
                reason="physical geometry intersects certified read context",
                metrics={"screen_geometry_present": 1.0},
            )

        cert_ref = (
            f"screen:{self.name}:{self.model_provenance}:{bbox.x0},{bbox.y0},{bbox.x1},{bbox.y1}"
        )
        return TileScreenDecision(
            status="SCREENED_OUT",
            reason=(
                "certified empty exact-vector read context and certified "
                "constant zero-context response"
            ),
            certified=True,
            fill_value=self.fill_value,
            certifies_verifiers=self.certifies_verifiers,
            verification_upper_bound=self.verification_upper_bound,
            certificate_ref=cert_ref,
            metrics={
                "screen_geometry_present": 0.0,
                "screened_core_pixels": float(request.core_bbox.area),
            },
        )
