from dataclasses import replace

from openlithohub.streaming.geometry import BoundingBox, HaloSpec
from openlithohub.streaming.vector_runs import (
    CellInstance,
    ExactVectorRunSource,
    VectorCell,
    rectangle,
)
from openlithohub.streaming.verification import (
    StreamingVerificationReducer,
    TileContext,
    VerificationContext,
)
from openlithohub.verify.run_ownership import RunOwnershipVerifier


def source():
    child = VectorCell("CHILD", polygons=(rectangle("c", 0, 0, 4, 3),))
    top = VectorCell(
        "TOP",
        polygons=(rectangle("long", 4, 6, 28, 10),),
        instances=(CellInstance("i", "CHILD", 15, 15),),
    )
    return ExactVectorRunSource(shape=(24, 32), cells={"TOP": top, "CHILD": child}, top="TOP")


def tile_context(src, bbox, tile_id):
    tensor = src.read_window(bbox)
    meta = src.verification_metadata(bbox, tile_id=tile_id, core_bbox=bbox)
    return TileContext(
        tile_id=tile_id,
        core_bbox=bbox,
        read_bbox=bbox,
        halo=HaloSpec.uniform(0),
        tensor=tensor,
        metadata=meta,
    )


def test_ownership_verifier_accepts_overlapping_views_without_new_parent():
    src = source()
    v = RunOwnershipVerifier()
    v.prepare(VerificationContext(model=None, pixel_nm=1.0))
    a = v.verify_tile(tile_context(src, BoundingBox(0, 6, 16, 7), "a"))
    b = v.verify_tile(tile_context(src, BoundingBox(12, 6, 32, 7), "b"))
    assert a.status == "PASS" and b.status == "PASS"
    # The same long parent run is seen through two tile views.
    assert v.state.seen_run_ids
    assert len(v.state.seen_run_ids) == 1


def test_analytic_periodic_copy_is_rejected_as_physical_input():
    src = source()
    bbox = BoundingBox(0, 6, 16, 7)
    t = tile_context(src, bbox, "a")
    runs = t.metadata["b04_owned_runs"]
    bad = replace(runs[0], analytic_period_shift=(0, 32))
    meta = dict(t.metadata)
    meta["b04_owned_runs"] = (bad,)
    badctx = replace(t, metadata=meta)
    v = RunOwnershipVerifier()
    v.prepare(VerificationContext(model=None, pixel_nm=1.0))
    assert v.verify_tile(badctx).status == "INCONCLUSIVE"


def test_existing_reducer_consumes_ownership_error_budget_without_sum_duplication():
    src = source()
    v = RunOwnershipVerifier()
    v.prepare(VerificationContext(model=None, pixel_nm=1.0))
    reducer = StreamingVerificationReducer(tolerance=1.0)
    for tile_id, bbox in [
        ("a", BoundingBox(0, 6, 16, 7)),
        ("b", BoundingBox(12, 6, 32, 7)),
    ]:
        ctx = tile_context(src, bbox, tile_id)
        # Assign the same physical parent a 0.125 charge in both views.
        charged = tuple(
            replace(r, error_budget_charge=0.125) for r in ctx.metadata["b04_owned_runs"]
        )
        meta = dict(ctx.metadata)
        meta["b04_owned_runs"] = charged
        verdict = v.verify_tile(replace(ctx, metadata=meta))
        reducer.add(verdict)
    final = reducer.finalize()
    assert final.status == "PASS"
    assert final.halo_error_budget == 0.125
