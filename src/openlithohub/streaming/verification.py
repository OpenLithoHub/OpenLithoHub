"""Generic verification-plugin API with streaming reduction (RFC 0008).

Any third-party verifier — a future QDM continuous verifier, an interval
verifier, a DRC-like checker, or a research plugin — integrates through
:class:`VerificationPlugin`.  The API is deliberately QDM-agnostic:

- ``required_halo`` lets a plugin *request* context;
- ``verify_tile`` consumes one tile at a time;
- ``reduce`` folds tile results into a global result incrementally, so no
  stage ever needs the full raster or an in-memory list of all tiles;
- ``RefinementRequest`` lets an inconclusive tile ask for more halo or a
  subdivision without the scheduler learning any verifier maths.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .geometry import BoundingBox, HaloSpec
from .halo_policy import HaloRequirement, HaloStatus


@dataclass(frozen=True)
class VerificationContext:
    """Everything a verifier may need before tiles start flowing.

    Reserved fields (``parameter_box``, ``forward_model_id``, hashes) keep
    the door open for QDM-style continuous process-parameter verification
    without hard-coding anything QDM-specific into the core.
    """

    model: Any
    pixel_nm: float
    forward_model_id: str = "unknown"
    process_parameters: dict[str, float] = field(default_factory=dict)
    parameter_box: tuple[tuple[str, float, float], ...] = ()
    metric: str = "epe"
    input_hash: str | None = None
    model_hash: str | None = None
    git_commit: str | None = None


@dataclass(frozen=True)
class TileContext:
    tile_id: str
    core_bbox: BoundingBox
    read_bbox: BoundingBox
    halo: HaloSpec
    tensor: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TileVerificationResult:
    tile_id: str
    core_bbox: BoundingBox
    status: Literal["PASS", "FAIL", "INCONCLUSIVE"]
    upper_bound: float | None = None
    lower_bound: float | None = None
    halo_px: int = 0
    halo_provenance: str = "none"
    certificate_ref: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    # B04 §10C — complete-contour-coverage contract fields.  ``None`` means
    # "not part of this verifier's theorem", which must not demote results.
    component_coverage: Literal[
        "ALL_COMPONENTS_COVERED", "PARTIAL_COVER", "INCONCLUSIVE", "NOT_APPLICABLE"
    ] = "NOT_APPLICABLE"
    root_brackets: tuple[tuple[float, float], ...] = ()
    kappa_lower: float | None = None
    hessian_upper: float | None = None
    spatial_extraction_error: float | None = None


@dataclass(frozen=True)
class RefinementRequest:
    """A verifier's advice to re-run a tile with more context."""

    tile_id: str
    action: Literal["increase_halo", "subdivide"] = "increase_halo"
    extra_halo_px: int = 0
    subdivision: int = 2

    def __post_init__(self) -> None:
        if self.action not in ("increase_halo", "subdivide"):
            raise ValueError(f"unknown refinement action: {self.action!r}")
        if self.action == "increase_halo" and self.extra_halo_px <= 0:
            raise ValueError("increase_halo refinement needs extra_halo_px > 0")
        if self.action == "subdivide" and self.subdivision < 2:
            raise ValueError("subdivide refinement needs subdivision >= 2")


@dataclass(frozen=True)
class GlobalVerificationResult:
    status: Literal["PASS", "FAIL", "INCONCLUSIVE"]
    worst_upper_bound: float | None
    worst_lower_bound: float | None
    n_tiles: int
    n_pass: int
    n_fail: int
    n_inconclusive: int
    coverage: Literal["ALL_COMPONENTS_COVERED", "PARTIAL_COVER", "INCONCLUSIVE", "NOT_APPLICABLE"]
    halo_error_budget: float
    error_budget: dict[str, float]
    certificate_refs: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)


class StreamingVerificationReducer:
    """Fold :class:`TileVerificationResult`\\s one at a time (prompt §9/10B).

    The reducer keeps only the *latest* verdict per tile (a refined pass
    replaces the earlier inconclusive one) — never the full history, and
    never any raster.  Error budget composition follows the B04
    decomposition::

        E_total <= E_process + E_spatial + E_halo + E_raster_bridge + E_other

    ``E_halo`` comes from each tile's ``HaloRequirement.error_bound``;
    halo errors certified as entering the theorem must keep the global
    result INCONCLUSIVE when they exceed the budget.
    """

    def __init__(
        self,
        *,
        process_error: float = 0.0,
        spatial_error: float = 0.0,
        raster_bridge_error: float = 0.0,
        other_error: float = 0.0,
        tolerance: float | None = None,
    ) -> None:
        self.error_budget: dict[str, float] = {
            "process": process_error,
            "spatial": spatial_error,
            "halo": 0.0,
            "raster_bridge": raster_bridge_error,
            "other": other_error,
        }
        self._tolerance = tolerance
        self._latest: dict[str, TileVerificationResult] = {}

    def add(self, result: TileVerificationResult) -> None:
        """Record one verdict; a later verdict for the same tile replaces it."""
        self._latest[result.tile_id] = result

    def retire(self, tile_id: str) -> None:
        """Remove a tile from final completeness (R17 C1a parent retirement).

        Subdivision retires the parent: its result must cease to participate
        in finalization.  This is a different operation from replacement —
        a halo refinement *replaces* a leaf's latest result, while
        subdivision *retires* the parent and introduces new child leaves.
        """
        self._latest.pop(tile_id, None)

    @property
    def n_tiles(self) -> int:
        return len(self._latest)

    def total_error(self) -> float:
        # Recompute from the latest per-tile verdicts each time so a
        # refined pass replaces its earlier halo bound.
        halo = 0.0
        for result in self._latest.values():
            bound = result.metrics.get("halo_error_bound")
            if result.halo_provenance != "none" and bound is not None:
                halo = max(halo, float(bound))
        self.error_budget["halo"] = halo
        return sum(self.error_budget.values())

    def finalize(self) -> GlobalVerificationResult:
        n = len(self._latest)
        n_pass = sum(r.status == "PASS" for r in self._latest.values())
        n_fail = sum(r.status == "FAIL" for r in self._latest.values())
        n_inconclusive = sum(r.status == "INCONCLUSIVE" for r in self._latest.values())
        if n == 0:
            status: Literal["PASS", "FAIL", "INCONCLUSIVE"] = "INCONCLUSIVE"
        elif n_fail:
            status = "FAIL"
        elif n_inconclusive:
            status = "INCONCLUSIVE"
        else:
            status = "PASS"

        worst_upper: float | None = None
        worst_lower: float | None = None
        refs: list[str] = []
        metrics: dict[str, float] = {}
        coverage_components = 0
        coverage_covered = 0
        for result in self._latest.values():
            if result.upper_bound is not None:
                worst_upper = max(worst_upper or float("-inf"), result.upper_bound)
            if result.lower_bound is not None:
                worst_lower = max(worst_lower or float("-inf"), result.lower_bound)
            if result.certificate_ref:
                refs.append(result.certificate_ref)
            for key, value in result.metrics.items():
                metrics[f"max_{key}"] = max(metrics.get(f"max_{key}", float("-inf")), float(value))
            if result.component_coverage != "NOT_APPLICABLE":
                coverage_components += 1
                if result.component_coverage == "ALL_COMPONENTS_COVERED":
                    coverage_covered += 1

        if coverage_components == 0:
            coverage: Literal[
                "ALL_COMPONENTS_COVERED",
                "PARTIAL_COVER",
                "INCONCLUSIVE",
                "NOT_APPLICABLE",
            ] = "NOT_APPLICABLE"
        elif coverage_covered == coverage_components:
            coverage = "ALL_COMPONENTS_COVERED"
        elif coverage_covered > 0:
            coverage = "PARTIAL_COVER"
        else:
            coverage = "INCONCLUSIVE"

        # §10K — PARTIAL_COVER != PASS.
        if status == "PASS" and coverage == "PARTIAL_COVER":
            status = "INCONCLUSIVE"
        # §10B — a breached error budget also blocks PASS. Compute
        # total_error() first so the halo budget lands in the breakdown.
        total_error = self.total_error()
        if status == "PASS" and self._tolerance is not None and total_error > self._tolerance:
            status = "INCONCLUSIVE"

        return GlobalVerificationResult(
            status=status,
            worst_upper_bound=worst_upper,
            worst_lower_bound=worst_lower,
            n_tiles=n,
            n_pass=n_pass,
            n_fail=n_fail,
            n_inconclusive=n_inconclusive,
            coverage=coverage,
            halo_error_budget=self.error_budget["halo"],
            error_budget=dict(self.error_budget),
            certificate_refs=tuple(refs),
            metrics=dict(metrics),
        )


@dataclass(frozen=True, order=True)
class VerifierIdentity:
    """Stable per-instance verifier identity (R17 C1a).

    Two instances of the same verifier class are distinct verifiers and must
    never share reduction state; ``instance_key`` separates them.
    """

    name: str
    version: str
    instance_key: str = "default"


@dataclass(frozen=True)
class MultiVerifierSummary:
    """Cross-verifier facade for ``report.verification`` when n > 1 (R17 C1a).

    Deliberately carries only cross-metric-safe data.  There is **no**
    ``worst_upper_bound`` here on purpose: a generic numeric upper bound
    across heterogeneous metrics is not a theorem.
    """

    status: Literal["PASS", "FAIL", "INCONCLUSIVE"]
    n_verifiers: int
    n_pass: int
    n_fail: int
    n_inconclusive: int


class VerifierSession:
    """One verifier instance → exactly one session → exactly one reducer.

    The session owns this verifier's per-tile results.  A halo refinement
    replaces a leaf's latest result; ``retire`` implements parent removal
    after subdivision.  Results of different verifier instances never mix.
    """

    def __init__(
        self,
        plugin: VerificationPlugin,
        identity: VerifierIdentity,
        reducer: StreamingVerificationReducer,
    ) -> None:
        self.identity = identity
        self.plugin = plugin
        self.reducer = reducer

    def add(self, result: TileVerificationResult) -> None:
        self.reducer.add(result)

    def retire(self, tile_id: str) -> None:
        self.reducer.retire(tile_id)

    def finalize(self) -> GlobalVerificationResult:
        return self.reducer.finalize()

    def results(self) -> list[TileVerificationResult]:
        """Latest result per non-retired tile (legacy batch-reduce input)."""
        return list(self.reducer._latest.values())


def make_verifier_session(
    plugin: VerificationPlugin,
    context: VerificationContext,
) -> VerifierSession:
    """Create the session for one verifier instance (R17 C1a).

    Verifiers may provide ``make_reducer(context, identity)`` to own their
    reducer construction; the stock per-session
    :class:`StreamingVerificationReducer` is the fallback.  The legacy batch
    ``reduce(results)`` method is *not* given new streaming semantics.
    """
    identity = VerifierIdentity(
        name=str(plugin.name),
        version=str(getattr(plugin, "version", "0")),
        instance_key=str(getattr(plugin, "instance_key", "default")),
    )
    factory = getattr(plugin, "make_reducer", None)
    if callable(factory):
        reducer = factory(context, identity)
        if not isinstance(reducer, StreamingVerificationReducer):
            raise TypeError(
                f"verifier {identity!r} make_reducer must return a StreamingVerificationReducer"
            )
    else:
        reducer = StreamingVerificationReducer()
    return VerifierSession(plugin=plugin, identity=identity, reducer=reducer)


def fold_global(
    results: dict[VerifierIdentity, GlobalVerificationResult],
) -> MultiVerifierSummary:
    """Apply the PASS/FAIL/INCONCLUSIVE firewall across verifier outcomes."""
    n_pass = sum(r.status == "PASS" for r in results.values())
    n_fail = sum(r.status == "FAIL" for r in results.values())
    n_inconclusive = sum(r.status == "INCONCLUSIVE" for r in results.values())
    if n_fail:
        status: Literal["PASS", "FAIL", "INCONCLUSIVE"] = "FAIL"
    elif n_inconclusive or not results:
        status = "INCONCLUSIVE"
    else:
        status = "PASS"
    return MultiVerifierSummary(
        status=status,
        n_verifiers=len(results),
        n_pass=n_pass,
        n_fail=n_fail,
        n_inconclusive=n_inconclusive,
    )


class VerificationRegistry:
    """Lightweight name → verifier-factory registry (prompt §8)."""

    def __init__(self) -> None:
        self._factories: dict[str, Any] = {}

    def register(self, name: str, factory: Any) -> None:
        if name in self._factories:
            raise ValueError(f"verifier {name!r} already registered")
        self._factories[name] = factory

    def get(self, name: str, **kwargs: Any) -> Any:
        if name not in self._factories:
            raise KeyError(f"unknown verifier {name!r}; available: {sorted(self._factories)}")
        factory = self._factories[name]
        if isinstance(factory, type) or callable(factory):
            return factory(**kwargs)
        return factory

    def list(self) -> list[str]:
        return sorted(self._factories)


verification_registry = VerificationRegistry()


class DummyVerifier:
    """Reference implementation proving the plugin contract end-to-end.

    Marks a tile PASS unless its id is listed in ``fail_tiles`` /
    ``inconclusive_tiles``.  Requests a fixed halo and emits a refinement
    advice for inconclusive tiles, demonstrating the full data flow a real
    (e.g. QDM) verifier would use.
    """

    name = "dummy"
    version = "1.0"

    def __init__(
        self,
        fail_tiles: set[str] | None = None,
        inconclusive_tiles: set[str] | None = None,
        halo_px: int = 24,
    ) -> None:
        self.fail_tiles = set(fail_tiles or ())
        self.inconclusive_tiles = set(inconclusive_tiles or ())
        self._halo_px = halo_px
        self.prepared = False
        self._inconclusive_seen: set[str] = set()

    def required_halo(self, context: VerificationContext) -> HaloRequirement | None:
        return HaloRequirement(
            halo_px=self._halo_px,
            provenance="dummy_plugin",
            error_bound=None,
            certified=False,
            reason="dummy verifier requests a fixed 24 px context",
            status=HaloStatus.FIXED,
        )

    def prepare(self, context: VerificationContext) -> None:
        self.prepared = True
        self._inconclusive_seen.clear()

    def verify_tile(self, tile: TileContext) -> TileVerificationResult:
        if tile.tile_id in self.fail_tiles:
            status: Literal["PASS", "FAIL", "INCONCLUSIVE"] = "FAIL"
        elif tile.tile_id in self.inconclusive_tiles:
            status = "INCONCLUSIVE"
            self._inconclusive_seen.add(tile.tile_id)
        else:
            status = "PASS"
        return TileVerificationResult(
            tile_id=tile.tile_id,
            core_bbox=tile.core_bbox,
            status=status,
            upper_bound=0.0 if status == "PASS" else None,
            halo_px=tile.halo.left,
            halo_provenance="dummy_plugin",
        )

    def refine(self, tile_id: str) -> RefinementRequest:
        return RefinementRequest(tile_id=tile_id, extra_halo_px=8)

    def reduce(self, results: Iterable[TileVerificationResult]) -> StreamingVerificationReducer:
        reducer = StreamingVerificationReducer()
        for result in results:
            reducer.add(result)
        return reducer

    def finalize(self) -> GlobalVerificationResult | None:
        return None


verification_registry.register("dummy", DummyVerifier)


@runtime_checkable
class VerificationPlugin(Protocol):
    name: str
    version: str

    def required_halo(self, context: VerificationContext) -> HaloRequirement | None: ...

    def prepare(self, context: VerificationContext) -> None: ...

    def verify_tile(self, tile: TileContext) -> TileVerificationResult: ...

    def reduce(self, results: Iterable[TileVerificationResult]) -> Any: ...

    def finalize(self) -> Any: ...
