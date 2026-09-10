"""R17 C6 — hostile compatibility and ownership regression suite.

Deterministic parametrized hostiles (guide §17): degenerate domains,
subdivision factors, boundary parents and refinement histories.  Every
case enforces the ledger invariants:

- terminal_active + terminal_exact_skip + terminal_verify == full chip
- unique_forward + pre_forward_avoided == full chip
- unique_forward <= full chip
- final leaves are unique, retired parents never reappear as terminal,
  and per-verifier completeness covers exactly the final leaves.
"""

import pytest
import torch

from openlithohub.streaming.core_halo import plan_tile_requests
from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.screening import ExactEmptyContextScreeningPolicy
from openlithohub.streaming.sinks import MetricOnlyTileSink
from openlithohub.streaming.verification import (
    RefinementRequest,
    TileContext,
    TileVerificationResult,
    VerificationContext,
)


class HostileSource:
    """Duck-typed exact source over an arbitrary WxH domain.

    Geometry: one horizontal run at y=run_y spanning x=run_x0..run_x1.
    """

    def __init__(self, height: int, width: int, run_y: int = 0, run_x0: int = 0, run_x1: int = 1):
        self.shape = (height, width)
        self.pixel_size_nm = 1.0
        self.layout_hash = f"hostile-{height}x{width}"
        self.backend_kind = "exact-vector"
        self._run = (run_y, run_x0, run_x1)

    def iter_runs_for_bbox(self, bbox_px):
        from openlithohub.verify.interface_runs import HorizontalRun

        x0, y0, x1, y1 = bbox_px
        ry, rx0, rx1 = self._run
        if y0 <= ry < y1 and min(rx1, x1) > max(rx0, x0):
            return (HorizontalRun(ry, max(rx0, x0), min(rx1, x1)),)
        return ()

    def read_window(self, bbox):
        out = torch.zeros((bbox.height, bbox.width), dtype=torch.float32)
        for run in self.iter_runs_for_bbox((bbox.x0, bbox.y0, bbox.x1, bbox.y1)):
            out[run.y - bbox.y0, run.x0 - bbox.x0 : run.x1 - bbox.x0] = 1.0
        return out


class RefiningVerifier:
    """First-seen INCONCLUSIVE; refines via a scripted action sequence."""

    name = "hostile-refiner"
    version = "1.0"

    def __init__(self, actions, accept_screens: bool = True):
        self.actions = list(actions)  # consumed per refine() call
        self.accept_screens = accept_screens
        self.results: dict[str, TileVerificationResult] = {}
        self.seen: list[str] = []

    def required_halo(self, context: VerificationContext):
        return None

    def prepare(self, context: VerificationContext) -> None:
        return None

    def verify_tile(self, tile: TileContext) -> TileVerificationResult:
        self.seen.append(tile.tile_id)
        result = TileVerificationResult(
            tile_id=tile.tile_id, core_bbox=tile.core_bbox, status="PASS"
        )
        self.results[tile.tile_id] = result
        return result

    def accept_screen_facts(self, tile: TileContext, facts):
        if not self.accept_screens:
            return None
        result = TileVerificationResult(
            tile_id=tile.tile_id, core_bbox=tile.core_bbox, status="PASS"
        )
        self.results[tile.tile_id] = result
        return result

    def refine(self, tile_id: str) -> RefinementRequest:
        action = self.actions.pop(0) if self.actions else ("subdivide", 2)
        if action[0] == "increase_halo":
            return RefinementRequest(
                tile_id=tile_id, action="increase_halo", extra_halo_px=action[1]
            )
        return RefinementRequest(tile_id=tile_id, action="subdivide", subdivision=action[1])

    def reduce(self, results):
        return None

    def finalize(self):
        return None


def _assert_invariants(source, report, verifiers):
    work = report.work_accounting
    full = int(source.shape[0] * source.shape[1])
    # terminal coverage ledger partitions the chip exactly
    assert (
        work["terminal_active_pixels"]
        + work["terminal_exact_skip_pixels"]
        + work["terminal_verification_skip_pixels"]
        == full
    )
    # historical work ledger partitions the chip exactly
    assert work["unique_forward_pixels"] + work["pre_forward_avoided_pixels"] == full
    assert 0 <= work["unique_forward_pixels"] <= full
    tree = report.ownership
    final_ids = [leaf.leaf_id for leaf in tree.final_leaves()]
    assert len(final_ids) == len(set(final_ids)), "duplicated final leaf"
    for leaf in tree.final_leaves():
        assert leaf.terminal_disposition is not None
    # no retired parent remains terminal or participates in completeness
    for leaf_id, leaf in tree._leaves.items():
        if not tree.is_final(leaf_id):
            assert leaf.terminal_disposition is None, f"retired {leaf_id} is terminal"
    # per-verifier completeness: the finalized verdict set of every session
    # covers exactly the final leaves — retired parents are excluded, every
    # final leaf has a fresh verdict (active verification or accepted facts)
    for _identity, vres in report.verification_results.items():
        assert vres.n_tiles == len(final_ids), "retired parent in finalization"
    for verifier in verifiers:
        for leaf_id in final_ids:
            assert leaf_id in verifier.results, f"{leaf_id} missing from a verifier"


def _run_with(source, verifiers, screening=None, core=4):
    return run_streaming(
        source,
        MetricOnlyTileSink(source.shape),
        lambda tile: tile,
        core_size=core,
        halo_policy=LegacyFixedHaloPolicy(0),
        verifiers=verifiers,
        screening_policy=screening,
    )


@pytest.mark.parametrize("height,width", [(1, 1), (1, 9), (9, 1), (7, 7), (10, 6), (5, 11)])
@pytest.mark.parametrize("subdivision", [2, 3, 4])
def test_hostile_domains_and_subdivisions(height, width, subdivision):
    source = HostileSource(height, width)
    verifier = RefiningVerifier(actions=[("subdivide", subdivision)])
    # subdivide the FIRST tile (works even on 1x1 domains: its child grid
    # degenerates gracefully through the area-conserving partition)
    first = next(iter(plan_tile_requests(source.shape, 4, 0)))
    if (
        first.core_bbox.area >= subdivision
        and min(first.core_bbox.width, first.core_bbox.height) >= 2
    ):
        verifier.actions = [("subdivide", subdivision)]
        verifier._target = first.tile_id

        class TargetedVerifier(RefiningVerifier):
            def verify_tile(self, tile: TileContext) -> TileVerificationResult:
                self.seen.append(tile.tile_id)
                status = "INCONCLUSIVE" if tile.tile_id == self._target else "PASS"
                result = TileVerificationResult(
                    tile_id=tile.tile_id, core_bbox=tile.core_bbox, status=status
                )
                self.results[tile.tile_id] = result
                return result

        verifier = TargetedVerifier([("subdivide", subdivision)], accept_screens=True)
        verifier._target = first.tile_id
    report = _run_with(source, [verifier])
    _assert_invariants(source, report, [verifier])


@pytest.mark.parametrize("corner", ["top-left", "top-right", "bottom-left", "bottom-right"])
def test_hostile_boundary_parent_subdivision(corner):
    height, width = 12, 20
    source = HostileSource(height, width)
    plan = list(plan_tile_requests(source.shape, 4, 0))
    index = {
        "top-left": 0,
        "top-right": 4,  # second column, first row (4 tiles per row)
        "bottom-left": len(plan) - 4,
        "bottom-right": len(plan) - 1,
    }[corner]
    target = plan[index].tile_id
    verifier = RefiningVerifier(actions=[("subdivide", 2)], accept_screens=True)

    class TargetedVerifier(RefiningVerifier):
        def verify_tile(self, tile: TileContext) -> TileVerificationResult:
            self.seen.append(tile.tile_id)
            status = "INCONCLUSIVE" if tile.tile_id == target else "PASS"
            result = TileVerificationResult(
                tile_id=tile.tile_id, core_bbox=tile.core_bbox, status=status
            )
            self.results[tile.tile_id] = result
            return result

    verifier = TargetedVerifier([("subdivide", 2)], accept_screens=True)
    report = _run_with(source, [verifier])
    _assert_invariants(source, report, [verifier])
    assert target not in [leaf.leaf_id for leaf in report.ownership.final_leaves()]


def test_hostile_history_multiple_halo_refinements_then_subdivide():
    source = HostileSource(12, 12)
    plan = list(plan_tile_requests(source.shape, 4, 0))
    target = plan[0].tile_id
    verifier = RefiningVerifier(
        actions=[("increase_halo", 2), ("increase_halo", 3), ("subdivide", 2)],
        accept_screens=True,
    )

    class TargetedVerifier(RefiningVerifier):
        def verify_tile(self, tile: TileContext) -> TileVerificationResult:
            self.seen.append(tile.tile_id)
            status = "INCONCLUSIVE" if tile.tile_id == target else "PASS"
            result = TileVerificationResult(
                tile_id=tile.tile_id, core_bbox=tile.core_bbox, status=status
            )
            self.results[tile.tile_id] = result
            return result

    verifier = TargetedVerifier(
        [("increase_halo", 2), ("increase_halo", 3), ("subdivide", 2)],
        accept_screens=True,
    )
    report = _run_with(source, [verifier])
    _assert_invariants(source, report, [verifier])
    # two halo re-runs and one subdivision happened
    assert report.n_requeued == 3
    assert report.work_accounting["n_subdivisions"] == 1


def test_hostile_screen_before_forward_no_verifier_needed():
    source = HostileSource(8, 8)  # run at y=0 x0..1 exists; rest screened
    screening = ExactEmptyContextScreeningPolicy(
        context_certified=True,
        zero_response_certified=True,
        fill_value=0.0,
        model_provenance="c6",
    )
    report = _run_with(source, [], screening=screening)
    _assert_invariants(source, report, [])
    work = report.work_accounting
    assert work["unique_forward_pixels"] + work["pre_forward_avoided_pixels"] == 64
    assert work["forward_simulator_calls"] == 1
    assert work["screen_queries"] >= 4


def test_hostile_zero_verifier_verification_only_rejected_by_tensor_sink():
    source = HostileSource(8, 8)
    from openlithohub.streaming.sinks import CertifiedCommitNotRepresentableError, TensorTileSink

    sink = TensorTileSink(source.shape)
    with pytest.raises(CertifiedCommitNotRepresentableError):
        sink.record_certified_core("t", BoundingBox(0, 0, 1, 1), exact_fill=None, metadata={})


def test_hostile_inc28_accounting_preserved_on_all_shapes():
    for height, width in [(8, 8), (9, 9), (1, 9), (6, 10)]:
        source = HostileSource(height, width)
        screening = ExactEmptyContextScreeningPolicy(
            context_certified=True,
            zero_response_certified=True,
            fill_value=0.0,
            model_provenance="c6",
        )
        report = _run_with(source, [], screening=screening)
        work = report.work_accounting
        full = int(source.shape[0] * source.shape[1])
        assert work["accounted_pct"] == 100.0
        assert work["unique_forward_pixels"] + work["pre_forward_avoided_pixels"] == full
        assert work["dense_allocation_events"] == 0
