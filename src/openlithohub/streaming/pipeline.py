"""Streaming full-chip pipeline (RFC 0008, prompt §11).

Runs the per-tile loop::

    optional certified pre-forward screen
        -> TileSource.read_window(core+halo) only for survivors
        -> forward model
        -> optional VerificationPlugin(s)
        -> TileSink.write_core(trusted core only)
        -> discard tile tensors

Peak memory is O(tile area + active batch), never O(full-chip raster).
Every core is written exactly once; an inconclusive verdict re-runs the
same core with a larger halo (bounded by ``max_requeues``) before the
best-effort result is committed.

Increment 28 integrates work accounting into the actual pipeline and adds a
fail-closed screening hook.  Screening may skip expensive work only when its
decision is certified; with verification plugins present, the screen must also
explicitly certify those verifiers or the pipeline falls back to the ordinary
active path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from .core_halo import TileRequest, plan_tile_requests, tiling_overhead
from .geometry import BoundingBox, HaloSpec, halo_actual, read_bbox_for
from .halo_policy import HaloContext, HaloPolicy, HaloRequirement, LegacyFixedHaloPolicy
from .ownership import (
    TERMINAL_ACTIVE,
    TERMINAL_EXACT_SKIP,
    TERMINAL_VERIFY_SKIP,
    OwnershipTree,
    partition_core,
)
from .screening import TileScreeningPolicy, screen_decision_to_facts
from .sinks import TileSink
from .sources import TileSource
from .verification import (
    GlobalVerificationResult,
    RefinementRequest,
    TileContext,
    TileVerificationResult,
    VerificationContext,
    VerificationPlugin,
    VerifierSession,
    fold_global,
    make_verifier_session,
)
from .work_accounting import WorkAccounting


@dataclass
class StreamingRunReport:
    n_tiles: int = 0
    n_requeued: int = 0
    halo_requirement: HaloRequirement | None = None
    overhead: dict[str, float] = field(default_factory=dict)
    verification: Any = None
    verification_results: dict[Any, GlobalVerificationResult] = field(default_factory=dict)
    metric_descriptors: dict[Any, Any] = field(default_factory=dict)
    ownership: Any = None
    work_accounting: dict[str, float | int] = field(default_factory=dict)


def _default_refinement(
    request: TileRequest, plugin: VerificationPlugin | None
) -> RefinementRequest:
    """Ask the plugin for refinement advice; fall back to a bigger halo."""
    refine = getattr(plugin, "refine", None)
    if callable(refine):
        advice: RefinementRequest | None = refine(request.tile_id)
        if advice is not None:
            return advice
    extra = max(request.halo.left, request.halo.top, request.halo.right, request.halo.bottom)
    return RefinementRequest(
        tile_id=request.tile_id,
        action="increase_halo",
        extra_halo_px=max(8, extra),
    )


def _grown_request(
    request: TileRequest, refinement: RefinementRequest, domain: tuple[int, int]
) -> TileRequest:
    """Re-plan a tile with more halo (clipped to the real global domain)."""
    max_px = max(1, refinement.extra_halo_px)
    current = max(request.halo.left, request.halo.top, request.halo.right, request.halo.bottom)
    grown = HaloSpec.uniform(current + max_px)
    read = read_bbox_for(
        request.core_bbox,
        grown,
        width=domain[1],
        height=domain[0],
    )
    return TileRequest(
        core_bbox=request.core_bbox,
        read_bbox=read,
        halo=halo_actual(request.core_bbox, read),
        tile_id=request.tile_id,
    )


def _core_slices(request: TileRequest) -> tuple[slice, slice]:
    """Location of the core inside the read-region tensor."""
    y0 = request.core_bbox.y0 - request.read_bbox.y0
    x0 = request.core_bbox.x0 - request.read_bbox.x0
    return (
        slice(y0, y0 + request.core_bbox.height),
        slice(x0, x0 + request.core_bbox.width),
    )


def run_streaming(
    source: TileSource,
    sink: TileSink,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    core_size: int,
    halo_policy: HaloPolicy | None = None,
    verifiers: Iterable[VerificationPlugin] = (),
    screening_policy: TileScreeningPolicy | None = None,
    work_accounting: WorkAccounting | None = None,
    max_halo_px: int = 1024,
    pixel_nm: float = 1.0,
    max_requeues: int = 4,
    tolerance_nm: float | None = None,
) -> StreamingRunReport:
    """Process a full chip tile-by-tile under core/halo ownership.

    ``tolerance_nm`` (R17 C5) is a legacy convenience: it flows down only
    when there is at most one verifier.  With several verifiers the error
    budget gate requires per-verifier tolerances on each verifier's own
    :class:`MetricDescriptor` — a scalar cannot be a shared tolerance for
    heterogeneous metrics.
    """
    policy = halo_policy or LegacyFixedHaloPolicy()
    verifier_list = list(verifiers)

    vctx = VerificationContext(model=forward_fn, pixel_nm=pixel_nm)
    plugin_requirements = []
    for verifier in verifier_list:
        requirement = verifier.required_halo(vctx)
        if requirement is not None:
            plugin_requirements.append(requirement)

    physical = policy.required_halo(HaloContext(pixel_nm=pixel_nm, tile_core_px=core_size))
    if plugin_requirements:
        from .halo_policy import combine_requirements

        requirement = combine_requirements(physical, *plugin_requirements)
    else:
        requirement = physical
    halo_px = min(requirement.halo_px, max_halo_px)

    # R17 C1a: one verifier instance → one session → one reducer.  Verifier
    # results never share reduction state; the pipeline only schedules and
    # folds the per-verifier outcomes.
    if tolerance_nm is not None and len(verifier_list) > 1:
        raise ValueError(
            "tolerance_nm is a single-verifier legacy convenience; attach a "
            "MetricDescriptor with its own tolerance to each verifier instead"
        )
    sessions: list[VerifierSession] = [
        make_verifier_session(verifier, vctx, fallback_tolerance=tolerance_nm)
        for verifier in verifier_list
    ]
    report = StreamingRunReport(halo_requirement=requirement)
    accounting = work_accounting or WorkAccounting()
    full_pixels = int(source.shape[0] * source.shape[1])
    if accounting.full_chip_pixels == 0:
        accounting.full_chip_pixels = full_pixels
    elif accounting.full_chip_pixels != full_pixels:
        raise ValueError("WorkAccounting.full_chip_pixels does not match the source domain")

    for verifier in verifier_list:
        verifier.prepare(vctx)

    ownership = OwnershipTree()
    for base in plan_tile_requests(source.shape, core_size, halo_px):
        ownership.add_root(base.tile_id, base.core_bbox)
        # R17 C3a: explicit work stack.  increase_halo re-plans one leaf;
        # subdivide retires the leaf and pushes a true area-conserving child
        # partition whose read windows are reconstructed against the GLOBAL
        # domain (never the parent's read bbox).
        pending: list[TileRequest] = [base]
        refinements_left = max_requeues
        while pending:
            current = pending.pop()
            screen_decision = None
            if screening_policy is not None:
                accounting.record_screen_query()
                screen_decision = screening_policy.screen(source, current)

            tile_meta_rejected: dict[str, Any] = {}
            if screen_decision is not None and screen_decision.status == "SCREENED_OUT":
                # R17 C1b: the screen produces typed proof facts; it has no
                # authority over verifiers.  A tile may skip the expensive
                # path only when (a) there are no verifiers to discharge, or
                # (b) every attached verifier accepts the facts itself.
                exact_output, proof_facts, deprecated_authority = screen_decision_to_facts(
                    screen_decision
                )

                accepted: list[TileVerificationResult] = []
                discharged = True
                for session in sessions:
                    accept = getattr(session.plugin, "accept_screen_facts", None)
                    if not callable(accept):
                        discharged = False
                        break
                    fact_ctx = TileContext(
                        tile_id=current.tile_id,
                        core_bbox=current.core_bbox,
                        read_bbox=current.read_bbox,
                        halo=current.halo,
                        tensor=None,
                        metadata={
                            "screening_status": screen_decision.status,
                            "screening_reason": screen_decision.reason,
                        },
                    )
                    verdict = accept(fact_ctx, proof_facts)
                    if verdict is None or verdict.status != "PASS":
                        discharged = False
                        break
                    accepted.append(verdict)

                if discharged:
                    tile_meta: dict[str, Any] = {
                        "screen_disposition": "EXACT_OUTPUT_SKIP",
                        "screening_status": "SCREENED_OUT",
                        "screening_policy": getattr(
                            screening_policy, "name", type(screening_policy).__name__
                        ),
                        "screening_reason": screen_decision.reason,
                        "screening_certified": True,
                        "screening_certificate_ref": screen_decision.certificate_ref,
                        **screen_decision.metrics,
                    }
                    if deprecated_authority:
                        tile_meta["screen_authority"] = (
                            "deprecated-blanket-input-recorded-not-honored"
                        )

                    for session, verdict in zip(sessions, accepted, strict=True):
                        session.add(verdict)

                    # R17 C2: prefer the tensor-free certified commit; legacy
                    # sinks without the capability fall back to a synthesized
                    # exact-fill write_core.
                    recorder = getattr(sink, "record_certified_core", None)
                    if callable(recorder):
                        recorder(
                            current.tile_id,
                            current.core_bbox,
                            exact_fill=exact_output.fill_value,
                            metadata=tile_meta,
                        )
                    else:
                        core_result = torch.full(
                            (current.core_bbox.height, current.core_bbox.width),
                            exact_output.fill_value,
                            dtype=torch.float32,
                        )
                        sink.write_core(current.tile_id, current.core_bbox, core_result, tile_meta)
                    accounting.record_screened_out(current.tile_id, current.core_bbox.area)
                    ownership.set_terminal(current.tile_id, TERMINAL_EXACT_SKIP)
                    report.n_tiles += 1
                    continue

                # Not discharged: fall through to the ordinary active path,
                # recording the rejected legacy authority as provenance.
                tile_meta_rejected = {
                    "screening_status": "SCREENED_OUT_NOT_DISCHARGED",
                    "screening_reason": screen_decision.reason,
                }
                if deprecated_authority:
                    tile_meta_rejected["screen_authority"] = (
                        "deprecated-blanket-input-recorded-not-honored"
                    )

            accounting.record_active_core(current.tile_id, current.core_bbox)
            tile = source.read_window(current.read_bbox)
            accounting.record_read_window(current.read_bbox.area)

            source_meta: dict[str, Any] = {}
            metadata_fn = getattr(source, "verification_metadata", None)
            if callable(metadata_fn):
                source_meta = dict(
                    metadata_fn(
                        current.read_bbox,
                        tile_id=current.tile_id,
                        core_bbox=current.core_bbox,
                    )
                )
            if screen_decision is not None:
                source_meta["screening_status"] = screen_decision.status
                source_meta["screening_reason"] = screen_decision.reason

            result = forward_fn(tile)
            accounting.record_forward(current.read_bbox.area)
            accounting.record_forward_core(current.core_bbox)
            ownership.mark_forward(current.tile_id)
            ys, xs = _core_slices(current)
            core_result = result[ys, xs]

            # Screen-fact provenance (e.g. a rejected SCREENED_OUT) rides on
            # the committed core metadata.
            tile_meta = dict(tile_meta_rejected)
            refinement: RefinementRequest | None = None
            for session in sessions:
                verifier = session.plugin
                tctx = TileContext(
                    tile_id=current.tile_id,
                    core_bbox=current.core_bbox,
                    read_bbox=current.read_bbox,
                    halo=current.halo,
                    tensor=tile,
                    metadata=source_meta,
                )
                verdict = verifier.verify_tile(tctx)
                session.add(verdict)
                tile_meta[f"{verifier.name}_status"] = verdict.status
                if verdict.status == "INCONCLUSIVE" and refinement is None:
                    if refinements_left > 0:
                        refinement = _default_refinement(current, verifier)
                    else:
                        tile_meta[f"{verifier.name}_note"] = (
                            "inconclusive; refinement budget exhausted"
                        )

            if refinement is None:
                ownership.set_terminal(current.tile_id, TERMINAL_ACTIVE)
                sink.write_core(
                    current.tile_id,
                    current.core_bbox,
                    core_result,
                    tile_meta,
                )
                report.n_tiles += 1
                continue

            refinements_left -= 1
            report.n_requeued += 1
            accounting.record_refinement(
                current.tile_id,
                current.tile_id,
                current.core_bbox.area,
            )
            if refinement.action == "subdivide":
                # R17 C3a: retire the parent everywhere and push a true
                # area-conserving partition.  Child read windows are
                # reconstructed against the GLOBAL domain (F3/§10.2), never
                # clipped to the parent read bbox.
                child_cores = [
                    (f"{current.tile_id}#q{index}", box)
                    for index, box in enumerate(
                        partition_core(current.core_bbox, refinement.subdivision)
                    )
                ]
                ownership.subdivide(current.tile_id, child_cores)
                accounting.record_subdivision(current.core_bbox)
                for session in sessions:
                    # R17 C3b: no automatic restriction theorem — every
                    # verifier must reach a fresh verdict on every child,
                    # including verifiers that had PASSed the parent.
                    session.retire(current.tile_id)
                for child_id, child_core in child_cores:
                    child_read = read_bbox_for(
                        child_core,
                        current.halo,
                        width=source.shape[1],
                        height=source.shape[0],
                    )
                    pending.append(
                        TileRequest(
                            core_bbox=child_core,
                            read_bbox=child_read,
                            halo=halo_actual(child_core, child_read),
                            tile_id=child_id,
                        )
                    )
            else:
                pending.append(_grown_request(current, refinement, source.shape))
            del tile, result, core_result

    # R17 C4a: the terminal coverage ledger must partition the chip exactly
    # — gap, overlap, unterminated or double-disposition leaves fail closed.
    terminal = ownership.verify_total_coverage(int(source.shape[0] * source.shape[1]))
    report.ownership = ownership
    report.overhead = tiling_overhead(plan_tile_requests(source.shape, core_size, halo_px))
    if sessions:
        report.verification_results = {session.identity: session.finalize() for session in sessions}
        report.metric_descriptors = {
            session.identity: session.metric_descriptor for session in sessions
        }
        if len(sessions) == 1:
            report.verification = next(iter(report.verification_results.values()))
        else:
            # R17 C1a: multi-verifier facade carries cross-metric-safe data
            # only — never a generic numeric upper bound.
            report.verification = fold_global(report.verification_results)
    summary = accounting.summary()
    summary.update(
        {
            "terminal_active_pixels": terminal[TERMINAL_ACTIVE],
            "terminal_exact_skip_pixels": terminal[TERMINAL_EXACT_SKIP],
            "terminal_verification_skip_pixels": terminal[TERMINAL_VERIFY_SKIP],
        }
    )
    report.work_accounting = summary
    return report


__all__ = [
    "BoundingBox",
    "HaloSpec",
    "StreamingRunReport",
    "run_streaming",
]
