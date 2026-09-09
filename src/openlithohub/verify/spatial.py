"""Coverage and curvature-scale completeness helpers for B04."""

from __future__ import annotations

from .types import CellCertificate, CellDisposition, CoverageStatus


def curvature_scale_nm(kappa_lower_per_nm: float, hessian_upper_per_nm2: float) -> float:
    if kappa_lower_per_nm <= 0.0:
        raise ValueError("kappa lower bound must be positive")
    if hessian_upper_per_nm2 < 0.0:
        raise ValueError("hessian upper bound must be nonnegative")
    if hessian_upper_per_nm2 == 0.0:
        return float("inf")
    return 2.0 * kappa_lower_per_nm / hessian_upper_per_nm2


def hidden_loop_excluded(
    *, cell_diameter_nm: float, kappa_lower_per_nm: float, hessian_upper_per_nm2: float
) -> bool:
    if cell_diameter_nm < 0.0:
        raise ValueError("cell diameter must be nonnegative")
    return cell_diameter_nm < curvature_scale_nm(kappa_lower_per_nm, hessian_upper_per_nm2)


def reduce_coverage(cells: tuple[CellCertificate, ...]) -> CoverageStatus:
    if not cells:
        return CoverageStatus.INCONCLUSIVE
    terminal_ok = {
        CellDisposition.SIGN_DEFINITE_EMPTY,
        CellDisposition.CURVATURE_CERTIFIED_EMPTY,
        CellDisposition.SEEDED,
    }
    ok = sum(c.disposition in terminal_ok for c in cells)
    if ok == len(cells):
        return CoverageStatus.ALL_COMPONENTS_COVERED
    if ok > 0:
        return CoverageStatus.PARTIAL_COVER
    return CoverageStatus.INCONCLUSIVE
