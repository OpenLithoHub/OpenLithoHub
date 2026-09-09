"""Verification-plugin API and streaming-reducer tests (RFC 0008, prompt §7-10)."""

import pytest

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.verification import (
    GlobalVerificationResult,
    RefinementRequest,
    StreamingVerificationReducer,
    TileContext,
    TileVerificationResult,
    VerificationRegistry,
    verification_registry,
)


def _tile(
    tile_id: str = "t0",
    status: str = "PASS",
    *,
    upper: float | None = None,
    lower: float | None = None,
    coverage: str = "NOT_APPLICABLE",
    halo_provenance: str = "none",
    metrics: dict | None = None,
    certificate_ref: str | None = None,
) -> TileVerificationResult:
    return TileVerificationResult(
        tile_id=tile_id,
        core_bbox=BoundingBox(0, 0, 32, 32),
        status=status,
        upper_bound=upper,
        lower_bound=lower,
        halo_px=8,
        halo_provenance=halo_provenance,
        metrics=metrics or {},
        component_coverage=coverage,
        certificate_ref=certificate_ref,
    )


class TestStreamingReducer:
    def test_streaming_reduce_all_pass(self):
        reducer = StreamingVerificationReducer()
        for i in range(1000):  # streaming: no list kept
            reducer.add(_tile(f"t{i}"))
        result = reducer.finalize()
        assert isinstance(result, GlobalVerificationResult)
        assert result.status == "PASS"
        assert result.n_tiles == 1000

    def test_fail_dominates_pass(self):
        reducer = StreamingVerificationReducer()
        reducer.add(_tile("a", "PASS"))
        reducer.add(_tile("b", "FAIL"))
        assert reducer.finalize().status == "FAIL"

    def test_inconclusive_blocks_pass(self):
        reducer = StreamingVerificationReducer()
        reducer.add(_tile("a", "PASS"))
        reducer.add(_tile("b", "INCONCLUSIVE"))
        assert reducer.finalize().status == "INCONCLUSIVE"

    def test_partial_cover_is_not_pass(self):
        reducer = StreamingVerificationReducer()
        reducer.add(_tile("a", "PASS", coverage="ALL_COMPONENTS_COVERED"))
        reducer.add(_tile("b", "PASS", coverage="PARTIAL_COVER"))
        result = reducer.finalize()
        assert result.coverage == "PARTIAL_COVER"
        assert result.status == "INCONCLUSIVE", "PARTIAL_COVER must not be PASS"

    def test_error_budget_composition(self):
        reducer = StreamingVerificationReducer(
            process_error=0.3, spatial_error=0.1, raster_bridge_error=0.05
        )
        reducer.add(
            _tile(
                "a",
                "PASS",
                halo_provenance="kernel_tail",
                metrics={"halo_error_bound": 0.02},
            )
        )
        reducer.add(
            _tile(
                "b",
                "PASS",
                halo_provenance="kernel_tail",
                metrics={"halo_error_bound": 0.07},
            )
        )
        total = reducer.total_error()
        assert reducer.error_budget["halo"] == 0.07  # worst halo bound folded in
        assert total == pytest.approx(0.3 + 0.1 + 0.07 + 0.05)

    def test_budget_breach_blocks_pass(self):
        # E_total = process 1.5 > tolerance 1.0 -> certified PASS impossible
        reducer = StreamingVerificationReducer(process_error=1.5, tolerance=1.0)
        reducer.add(_tile("a", "PASS"))
        assert reducer.finalize().status == "INCONCLUSIVE"

    def test_within_budget_stays_pass(self):
        reducer = StreamingVerificationReducer(process_error=0.3, spatial_error=0.2, tolerance=1.0)
        reducer.add(_tile("a", "PASS"))
        assert reducer.finalize().status == "PASS"

    def test_worst_bounds_and_refs(self):
        reducer = StreamingVerificationReducer()
        reducer.add(_tile("a", "PASS", upper=0.5, certificate_ref="cert-a"))
        reducer.add(_tile("b", "PASS", upper=0.9, certificate_ref="cert-b"))
        result = reducer.finalize()
        assert result.worst_upper_bound == 0.9
        assert result.certificate_refs == ("cert-a", "cert-b")

    def test_empty_is_inconclusive(self):
        assert StreamingVerificationReducer().finalize().status == "INCONCLUSIVE"


class TestRegistry:
    def test_register_get_list(self):
        registry = VerificationRegistry()

        class Dummy:
            name = "dummy"

        registry.register("dummy", Dummy)
        assert registry.list() == ["dummy"]
        instance = registry.get("dummy")
        assert isinstance(instance, Dummy)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("dummy", Dummy)
        with pytest.raises(KeyError, match="unknown verifier"):
            registry.get("nope")

    def test_global_registry_has_dummy(self):
        assert "dummy" in verification_registry.list()


class _DummyVerifier:
    """Minimal plugin used by the acceptance tests (prompt PR D/E)."""

    name = "dummy"
    version = "1.0"

    def __init__(self, fail_tiles: set[str] | None = None, inconclusive: set[str] | None = None):
        self.fail_tiles = fail_tiles or set()
        self.inconclusive = inconclusive or set()
        self.requested_halo = 0
        self.prepared = False
        self.tiles_seen: list[str] = []

    def required_halo(self, context):
        self.requested_halo = 24
        from openlithohub.streaming.halo_policy import HaloRequirement

        return HaloRequirement(
            halo_px=24,
            provenance="dummy_plugin",
            error_bound=None,
            certified=False,
            reason="dummy wants 24 px of context",
            status="PHYSICS_ESTIMATED",
        )

    def prepare(self, context):
        self.prepared = True

    def verify_tile(self, tile: TileContext) -> TileVerificationResult:
        self.tiles_seen.append(tile.tile_id)
        if tile.tile_id in self.fail_tiles:
            status = "FAIL"
        elif tile.tile_id in self.inconclusive:
            status = "INCONCLUSIVE"
        else:
            status = "PASS"
        return TileVerificationResult(
            tile_id=tile.tile_id,
            core_bbox=tile.core_bbox,
            status=status,
            halo_px=tile.halo.left,
            halo_provenance="dummy_plugin",
        )

    def reduce(self, results):
        reducer = StreamingVerificationReducer()
        for r in results:
            reducer.add(r)
        return reducer

    def finalize(self):
        return None


class TestRefinementRequest:
    def test_subdivide_validation(self):
        with pytest.raises(ValueError, match="subdivision"):
            RefinementRequest(tile_id="t", action="subdivide", subdivision=1)

    def test_refinement_defaults(self):
        r = RefinementRequest(tile_id="t", extra_halo_px=8)
        assert r.action == "increase_halo"
