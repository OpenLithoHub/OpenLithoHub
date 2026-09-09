"""Proof gates for B04 continuous-focus verification.

The central firewall is intentionally asymmetric:
- certified upper bounds may prove PASS;
- FAIL requires an independently certified violation lower bound;
- everything else is INCONCLUSIVE.
"""

from __future__ import annotations

from .types import (
    CertificateStatus,
    CertificateTarget,
    ContinuousFocusCertificate,
    CoverageStatus,
    DependencyRecord,
    ParameterInterval,
    ProofLevel,
)

PINNED_UPSTREAM = "348fa5d86d5355465af98e2c4ce3deac60081a4c"

_CERTIFYING_LEVELS = {
    ProofLevel.INTERVAL_CERTIFIED,
    ProofLevel.IMPORTED_QDM_CERTIFIED,
}


def dependency_chain_is_certified(dependencies: tuple[DependencyRecord, ...]) -> bool:
    return bool(dependencies) and all(
        d.satisfied and d.level in _CERTIFYING_LEVELS for d in dependencies
    )


def assemble_continuous_focus_certificate(
    *,
    model_id: str,
    target: CertificateTarget = CertificateTarget.EXTRACTED_CONTOUR_EPE,
    input_sha256: str,
    parameters: tuple[ParameterInterval, ...],
    continuous_focus_contour_upper_nm: float,
    continuous_focus_pairwise_diameter_upper_nm: float | None,
    tolerance_nm: float,
    coverage_status: CoverageStatus,
    dependencies: tuple[DependencyRecord, ...],
    certified_violation_lower_nm: float | None = None,
    proof_artifact_sha256: str | None = None,
    nominal_reconstruction_upper_nm: float | None = None,
    note: str = "",
) -> ContinuousFocusCertificate:
    if continuous_focus_contour_upper_nm < 0.0:
        raise ValueError("continuous_focus_contour_upper_nm must be nonnegative")
    if tolerance_nm < 0.0:
        raise ValueError("tolerance_nm must be nonnegative")
    if certified_violation_lower_nm is not None and certified_violation_lower_nm < 0.0:
        raise ValueError("certified_violation_lower_nm must be nonnegative")

    deps_ok = dependency_chain_is_certified(dependencies)

    # Contradictory certified claims indicate a corrupt proof package, not a
    # choice between PASS and FAIL.
    if (
        deps_ok
        and certified_violation_lower_nm is not None
        and continuous_focus_contour_upper_nm <= tolerance_nm
        and certified_violation_lower_nm > tolerance_nm
    ):
        raise ValueError("certificate contains contradictory certified bounds")

    status = CertificateStatus.INCONCLUSIVE
    if (
        deps_ok
        and certified_violation_lower_nm is not None
        and certified_violation_lower_nm > tolerance_nm
    ):
        status = CertificateStatus.FAIL
    coverage_ok = (
        target is CertificateTarget.LEVEL_SET_STABILITY
        or coverage_status is CoverageStatus.ALL_COMPONENTS_COVERED
    )
    if (
        status is CertificateStatus.INCONCLUSIVE
        and deps_ok
        and coverage_ok
        and continuous_focus_contour_upper_nm <= tolerance_nm
    ):
        status = CertificateStatus.PASS

    return ContinuousFocusCertificate(
        target=target,
        upstream_commit=PINNED_UPSTREAM,
        model_id=model_id,
        input_sha256=input_sha256,
        parameters=parameters,
        continuous_focus_contour_upper_nm=continuous_focus_contour_upper_nm,
        continuous_focus_pairwise_diameter_upper_nm=(continuous_focus_pairwise_diameter_upper_nm),
        tolerance_nm=tolerance_nm,
        coverage_status=coverage_status,
        dependencies=dependencies,
        status=status,
        certified_violation_lower_nm=certified_violation_lower_nm,
        proof_artifact_sha256=proof_artifact_sha256,
        nominal_reconstruction_upper_nm=nominal_reconstruction_upper_nm,
        note=note,
    )
