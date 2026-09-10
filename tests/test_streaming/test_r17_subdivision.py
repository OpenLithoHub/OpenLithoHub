"""R17 C3a/C3b — ownership tree, true subdivision, final-leaf completeness.

Covers the required matrix items:
7.  halo refinement replaces a leaf's latest result;
8.  parent result retirement under subdivision;
9.  no parent PASS inheritance (no automatic restriction theorem);
10. each child × each verifier reaches a final result;
11. area-conserving subdivision (parametrized hostile geometries);
12. child read windows are reconstructed against the GLOBAL domain;
13. forward history is inherited by children (F(Cj) ← F(P));
14. split-before-forward is genuine work avoidance (children screened).
"""

import pytest
import torch

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.ownership import OwnershipTree, partition_core
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.sinks import MetricOnlyTileSink
from openlithohub.streaming.sources import TensorTileSource
from openlithohub.streaming.verification import (
    RefinementRequest,
    TileContext,
    TileVerificationResult,
    VerificationContext,
)


class SubdivideWhenInconclusive:
    """INCONCLUSIVE on scripted ids (then subdivide), PASS elsewhere."""

    name = "subdivider"
    version = "1.0"

    def __init__(self, subdivide_ids, subdivision: int = 2):
        self.subdivide_ids = set(subdivide_ids)
        self.subdivision = subdivision
        self.seen: list[str] = []

    def required_halo(self, context: VerificationContext):
        return None

    def prepare(self, context: VerificationContext) -> None:
        return None

    def verify_tile(self, tile: TileContext) -> TileVerificationResult:
        self.seen.append(tile.tile_id)
        status = "INCONCLUSIVE" if tile.tile_id in self.subdivide_ids else "PASS"
        return TileVerificationResult(tile_id=tile.tile_id, core_bbox=tile.core_bbox, status=status)

    def accept_screen_facts(self, tile: TileContext, facts):
        kinds = {fact.kind for fact in facts}
        if "EXACT_OUTPUT_SKIP" in kinds and "#" in tile.tile_id:
            # children of a subdivided parent accept the empty-screen fact;
            # parents never do (they must run and be judged by verify_tile)
            return TileVerificationResult(
                tile_id=tile.tile_id, core_bbox=tile.core_bbox, status="PASS"
            )
        return None

    def refine(self, tile_id: str) -> RefinementRequest:
        return RefinementRequest(tile_id=tile_id, action="subdivide", subdivision=self.subdivision)

    def reduce(self, results):
        return None

    def finalize(self):
        return None


class HaloThenPassVerifier:
    """INCONCLUSIVE on first sight of an id, PASS on any later re-run."""

    name = "halo-then-pass"
    version = "1.0"

    def __init__(self) -> None:
        self._seen_first: set[str] = set()

    def required_halo(self, context: VerificationContext):
        return None

    def prepare(self, context: VerificationContext) -> None:
        self._seen_first.clear()

    def verify_tile(self, tile: TileContext) -> TileVerificationResult:
        first = tile.tile_id not in self._seen_first
        self._seen_first.add(tile.tile_id)
        return TileVerificationResult(
            tile_id=tile.tile_id,
            core_bbox=tile.core_bbox,
            status="INCONCLUSIVE" if first else "PASS",
        )

    def refine(self, tile_id: str) -> RefinementRequest:
        return RefinementRequest(tile_id=tile_id, action="increase_halo", extra_halo_px=2)

    def reduce(self, results):
        return None

    def finalize(self):
        return None


def _source(size: int = 16) -> TensorTileSource:
    return TensorTileSource(torch.zeros((size, size), dtype=torch.float32))


def _run(source, verifiers, screening=None):
    sink = MetricOnlyTileSink(source.shape)
    report = run_streaming(
        source,
        sink,
        lambda tile: tile,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        verifiers=verifiers,
        screening_policy=screening,
    )
    return report, sink


def test_halo_refinement_replaces_latest_leaf_result():
    report, _sink = _run(_source(), [HaloThenPassVerifier()])
    vres = report.verification
    # every tile ends on its latest (PASS) verdict — no stale INCONCLUSIVE
    assert vres.status == "PASS"
    assert vres.n_tiles == 4
    assert vres.n_inconclusive == 0


def test_subdivision_retires_parent_and_verifies_children():
    verifier = SubdivideWhenInconclusive({"tile_0"})
    report, sink = _run(_source(), [verifier])
    # 3 untouched tiles + 4 children; the retired parent is gone everywhere
    assert report.verification.n_tiles == 7
    assert report.verification.status == "PASS"
    assert report.verification.n_inconclusive == 0
    assert "tile_0" not in sink.tiles
    # the work stack is LIFO, so child commit order is not the partition
    # order; membership is what completeness requires
    assert {t for t in sink.tiles if t.startswith("tile_0#")} == {
        "tile_0#q0",
        "tile_0#q1",
        "tile_0#q2",
        "tile_0#q3",
    }
    # every child was freshly verified (fresh verdicts, not inherited)
    for child in ("tile_0#q0", "tile_0#q1", "tile_0#q2", "tile_0#q3"):
        assert child in verifier.seen


def test_no_parent_pass_inheritance_across_verifiers():
    always_pass = SubdivideWhenInconclusive(set())
    always_pass.name = "always-pass"
    subdivider = SubdivideWhenInconclusive({"tile_0"})
    subdivider.name = "subdivider"
    report, _sink = _run(_source(), [always_pass, subdivider])
    assert set(report.verification_results) is not None
    results = list(report.verification_results.values())
    # BOTH verifiers retire the PASSed parent and verify all children fresh
    for res in results:
        assert res.n_tiles == 7
        assert res.status == "PASS"
    children_pass = [t for t in always_pass.seen if t.startswith("tile_0#")]
    assert len(children_pass) == 4


def test_child_read_windows_use_global_domain():
    captured: list = []

    class SpySource(TensorTileSource):
        def read_window(self, bbox):
            captured.append(bbox)
            return super().read_window(bbox)

    source = SpySource(torch.zeros((16, 16), dtype=torch.float32))
    run_streaming(
        source,
        MetricOnlyTileSink(source.shape),
        lambda tile: tile,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(4),
        verifiers=[SubdivideWhenInconclusive({"tile_0"})],
    )
    from openlithohub.streaming.geometry import HaloSpec, read_bbox_for

    assert captured, "spy recorded no reads"
    for bbox in captured:
        assert bbox.x0 >= 0 and bbox.y0 >= 0
        assert bbox.x1 <= 16 and bbox.y1 <= 16
    # corner child q0 of tile_0 inherits the parent's ACTUAL halo
    # (left/top clipped to 0 at the chip edge); its read window is exactly
    # the global-domain reconstruction for core (0,0,4,4).
    expected = read_bbox_for(
        BoundingBox(0, 0, 4, 4), HaloSpec(left=0, top=0, right=4, bottom=4), width=16, height=16
    )
    assert (expected.x0, expected.y0, expected.x1, expected.y1) == (0, 0, 8, 8)
    assert expected in captured


@pytest.mark.parametrize(
    "box,parts",
    [
        ((0, 0, 16, 16), 2),
        ((0, 0, 16, 16), 3),
        ((0, 0, 16, 16), 4),
        ((0, 0, 1, 7), 3),  # 1xN
        ((0, 0, 7, 1), 3),  # Nx1
        ((2, 3, 12, 13), 3),  # odd, non-divisible
        ((5, 5, 6, 6), 4),  # 1x1 parent: zero-area children are dropped
    ],
)
def test_partition_core_is_area_conserving(box, parts):
    parent = BoundingBox(*box)
    children = partition_core(parent, parts)
    assert sum(c.area for c in children) == parent.area
    keys = [(c.x0, c.y0, c.x1, c.y1) for c in children]
    assert len(keys) == len(set(keys))
    for c in children:
        assert c.x0 >= parent.x0 and c.y0 >= parent.y0
        assert c.x1 <= parent.x1 and c.y1 <= parent.y1
        assert c.width > 0 and c.height > 0
    # deterministic
    again = partition_core(parent, parts)
    assert [(c.x0, c.y0, c.x1, c.y1) for c in again] == keys


def test_children_inherit_forward_history():
    tree = OwnershipTree()
    tree.add_root("t", BoundingBox(0, 0, 8, 8))
    tree.mark_forward("t")
    children = tree.subdivide(
        "t",
        [
            ("t#q0", BoundingBox(0, 0, 4, 4)),
            ("t#q1", BoundingBox(4, 0, 8, 4)),
            ("t#q2", BoundingBox(0, 4, 4, 8)),
            ("t#q3", BoundingBox(4, 4, 8, 8)),
        ],
    )
    assert all(child.forward_seen for child in children)
    assert not tree.is_final("t")
    assert len(tree.final_leaves()) == 4

    fresh = OwnershipTree()
    fresh.add_root("u", BoundingBox(0, 0, 8, 8))
    fresh_children = fresh.subdivide(
        "u",
        [
            ("u#q0", BoundingBox(0, 0, 4, 4)),
            ("u#q1", BoundingBox(4, 0, 8, 4)),
            ("u#q2", BoundingBox(0, 4, 4, 8)),
            ("u#q3", BoundingBox(4, 4, 8, 8)),
        ],
    )
    assert not any(child.forward_seen for child in fresh_children)


def test_split_before_forward_is_genuine_work_avoidance(tmp_path=None):
    from test_screening_work_accounting_inc28 import CounterForward

    # a 32x16 chip: tile_1 (x 8..16) is geometrically empty but INCONCLUSIVE
    # at the parent level; after subdivision its empty children are screened
    # and never forwarded.
    source = SparseEmptyMiddleSource()
    forward = CounterForward()
    sink = MetricOnlyTileSink(source.shape)
    from test_screening_work_accounting_inc28 import ExactEmptyContextScreeningPolicy

    screening = ExactEmptyContextScreeningPolicy(
        context_certified=True,
        zero_response_certified=True,
        fill_value=0.0,
        model_provenance="c3",
    )
    verifier = SubdivideWhenInconclusive({"tile_1"})
    report = run_streaming(
        source,
        sink,
        forward,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        verifiers=[verifier],
        screening_policy=screening,
    )
    # the 32x16 chip has 8 parent tiles; every parent runs the forward
    # (verify drives subdivision, so the parent forwards first), but the
    # four empty tile_1 children are screened and NEVER forwarded.
    assert forward.calls == 8
    screened = [t for t in sink.tiles if t.startswith("tile_1#")]
    assert len(screened) == 4  # all four children reached a terminal commit
    assert all(sink.metadata[t]["screen_disposition"] == "EXACT_OUTPUT_SKIP" for t in screened)
    assert report.verification.status == "PASS"


class SparseEmptyMiddleSource:
    """32x16 chip, geometry only in the first tile column.

    Duck-typed exact source (same protocol as the Inc28 test fixture): the
    middle column (x 8..16) is geometrically empty, so subdividing an
    inconclusive parent there yields screenable empty children.
    """

    shape = (16, 32)
    pixel_size_nm = 1.0
    layout_hash = "sparse-empty-middle"
    backend_kind = "exact-vector"

    def iter_runs_for_bbox(self, bbox_px):
        from openlithohub.verify.interface_runs import HorizontalRun

        x0, y0, x1, y1 = bbox_px
        run = (2, 1, 3)
        if y0 <= run[0] < y1 and min(run[2], x1) > max(run[1], x0):
            return (HorizontalRun(run[0], max(run[1], x0), min(run[2], x1)),)
        return ()

    def read_window(self, bbox):
        out = torch.zeros((bbox.height, bbox.width), dtype=torch.float32)
        for run in self.iter_runs_for_bbox((bbox.x0, bbox.y0, bbox.x1, bbox.y1)):
            out[run.y - bbox.y0, run.x0 - bbox.x0 : run.x1 - bbox.x0] = 1.0
        return out
