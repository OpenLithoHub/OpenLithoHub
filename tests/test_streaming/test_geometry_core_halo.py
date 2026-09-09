"""Core/halo geometry and planning tests (RFC 0008, prompt §1/§13/§15)."""

import pytest

from openlithohub.streaming.core_halo import (
    RefinementRequest,
    TileScheduler,
    plan_tile_requests,
    plan_tiling,
    subdivide_request,
    tiling_overhead,
)
from openlithohub.streaming.geometry import (
    BoundingBox,
    HaloSpec,
    halo_actual,
    halo_overhead_stats,
    read_bbox_for,
)


class TestBoundingBox:
    def test_rejects_degenerate(self):
        with pytest.raises(ValueError, match="degenerate"):
            BoundingBox(x0=5, y0=0, x1=5, y1=10)

    def test_area_and_dims(self):
        b = BoundingBox(x0=2, y0=3, x1=7, y1=11)
        assert (b.width, b.height, b.area) == (5, 8, 40)

    def test_clip_and_contains(self):
        domain = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        inner = BoundingBox(x0=2, y0=2, x1=5, y1=5)
        assert domain.contains(inner)
        assert not inner.contains(domain)
        clipped = BoundingBox(x0=-3, y0=-1, x1=20, y1=4).clipped_to(10, 10)
        assert clipped == BoundingBox(x0=0, y0=0, x1=10, y1=4)


class TestHaloSpec:
    def test_uniform_and_total(self):
        h = HaloSpec.uniform(4)
        assert (h.left, h.top, h.right, h.bottom) == (4, 4, 4, 4)
        assert h.total == 16

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="nonnegative"):
            HaloSpec.uniform(-1)


class TestExactCoreCover:
    @pytest.mark.parametrize(
        "shape,core", [((64, 64), 16), ((100, 90), 32), ((17, 33), 8), ((7, 5), 16)]
    )
    def test_cores_exactly_cover_domain(self, shape, core):
        boxes = [(b.x0, b.y0, b.x1, b.y1) for b in core_grid_boxes_safe(shape, core)]
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes)
        assert area == shape[0] * shape[1], "cores must cover the domain exactly"

        seen = set()
        for box in boxes:
            for y in range(box[1], box[3]):
                for x in range(box[0], box[2]):
                    seen.add((y, x))
        assert len(seen) == shape[0] * shape[1], "cores must not overlap"


def core_grid_boxes_safe(shape, core):
    from openlithohub.streaming.geometry import core_grid_boxes

    return core_grid_boxes(shape, core)


class TestReadRegions:
    def test_read_expands_and_clips(self):
        core = BoundingBox(x0=0, y0=0, x1=32, y1=32)
        read = read_bbox_for(core, HaloSpec.uniform(8), 100, 100)
        assert read == BoundingBox(x0=0, y0=0, x1=40, y1=40)
        assert halo_actual(core, read) == HaloSpec(left=0, top=0, right=8, bottom=8)

    def test_interior_core_full_halo(self):
        core = BoundingBox(x0=32, y0=32, x1=64, y1=64)
        read = read_bbox_for(core, HaloSpec.uniform(8), 1000, 1000)
        assert halo_actual(core, read) == HaloSpec.uniform(8)


class TestRequestsAndScheduler:
    def test_plan_requests_shapes(self):
        requests = plan_tile_requests((100, 90), 32, 8)
        assert all(r.read_bbox.width >= r.core_bbox.width for r in requests)
        assert all(r.tile_id.startswith("tile_") for r in requests)

    def test_scheduler_iterates_once(self):
        requests = plan_tile_requests((64, 64), 32, 0)
        scheduler = TileScheduler(requests)
        pulled = list(scheduler)
        assert pulled == requests
        try:
            next(scheduler)
            raised = False
        except StopIteration:
            raised = True
        assert raised

    def test_refinement_validation(self):
        with pytest.raises(ValueError, match="unknown refinement"):
            RefinementRequest(tile_id="t", action="explode")
        with pytest.raises(ValueError, match="extra_halo_px"):
            RefinementRequest(tile_id="t", extra_halo_px=0)

    def test_enqueue_refinement_grows_halo(self):
        requests = plan_tile_requests((64, 64), 32, 4)
        scheduler = TileScheduler(requests, domain=(64, 64))
        first = next(scheduler)  # corner tile: left/top clipped to 0
        refined = scheduler.enqueue_refinement(
            first, RefinementRequest(tile_id=first.tile_id, extra_halo_px=4)
        )
        assert refined[0].halo.right == 8
        assert refined[0].halo.bottom == 8
        assert refined[0].halo.left == 0  # still domain-clipped
        assert scheduler._pending[-1].tile_id == first.tile_id

    def test_subdivide_splits_core(self):
        request = plan_tile_requests((64, 64), 32, 4)[0]
        subs = subdivide_request(request, 2)
        assert len(subs) == 4
        total = sum(s.core_bbox.area for s in subs)
        assert total == request.core_bbox.area


class TestOverhead:
    def test_overhead_stats_zero_halo(self):
        requests = plan_tile_requests((64, 64), 32, 0)
        stats = tiling_overhead(requests)
        assert stats["halo_overhead_pct"] == 0.0
        assert stats["n_tiles"] == 4.0

    def test_overhead_stats_with_halo(self):
        requests = plan_tile_requests((64, 64), 32, 8)
        stats = tiling_overhead(requests)
        assert stats["halo_pixels"] > 0
        assert 0 < stats["halo_overhead_pct"] < 100

    def test_halo_overhead_stats_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            halo_overhead_stats([BoundingBox(0, 0, 1, 1)], [])


class TestPlanTiling:
    def test_budget_limits_core_size(self):
        # tiny budget forces the minimum core size
        requests = plan_tiling((512, 512), memory_budget_bytes=64 * 64 * 4 * 4, min_core_size=32)
        assert all(r.core_bbox.width <= 128 for r in requests)

    def test_small_layout_single_tile(self):
        requests = plan_tiling((48, 48), memory_budget_bytes=10**9, min_core_size=32)
        assert len(requests) == 1
        assert requests[0].core_bbox == BoundingBox(0, 0, 48, 48)
        assert requests[0].halo.total == 0

    def test_invalid_budget_rejected(self):
        with pytest.raises(ValueError, match="memory_budget"):
            plan_tiling((64, 64), memory_budget_bytes=0)
