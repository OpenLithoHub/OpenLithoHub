from openlithohub.verify.boundary import Interval, isolate_roots_on_segment
from openlithohub.verify.spatial import hidden_loop_excluded, reduce_coverage
from openlithohub.verify.types import (
    BoundaryRootStatus,
    CellCertificate,
    CellDisposition,
    CoverageStatus,
)


def test_monotone_single_root_is_bracketed():
    # f(t)=t-0.4, f'=1.  Interval extension is exact.
    def f(a: float, b: float) -> Interval:
        return Interval(a - 0.4, b - 0.4)

    def df(a: float, b: float) -> Interval:
        return Interval(1.0, 1.0)

    r = isolate_roots_on_segment(f, df, root_width=1e-5)
    assert r.status is BoundaryRootStatus.ROOT_BRACKETS
    assert len(r.brackets) == 1
    assert r.brackets[0].hi - r.brackets[0].lo <= 1e-5


def test_root_free_segment_is_certified():
    def f(a: float, b: float) -> Interval:
        return Interval(a + 1.0, b + 1.0)

    def df(a: float, b: float) -> Interval:
        return Interval(1.0, 1.0)

    r = isolate_roots_on_segment(f, df)
    assert r.status is BoundaryRootStatus.ROOT_FREE_CERTIFIED


def test_curvature_scale_hidden_loop_gate():
    assert hidden_loop_excluded(
        cell_diameter_nm=11.3137085,
        kappa_lower_per_nm=0.005891396342472183,
        hessian_upper_per_nm2=0.0006701535065401148,
    )


def _cell(disposition):
    return CellCertificate(
        cell_id="c",
        cell_bbox_nm=(0, 0, 8, 8),
        kappa_lower_per_nm=0.005,
        hessian_upper_per_nm2=0.001,
        cell_diameter_nm=11.4,
        curvature_scale_nm=10.0,
        boundary_root_status=BoundaryRootStatus.INCONCLUSIVE,
        hidden_loop_excluded=False,
        root_brackets=(),
        eta_sp_nm=None,
        disposition=disposition,
    )


def test_partial_cover_never_becomes_all_components_covered():
    cells = (_cell(CellDisposition.SEEDED), _cell(CellDisposition.INCONCLUSIVE))
    assert reduce_coverage(cells) is CoverageStatus.PARTIAL_COVER
