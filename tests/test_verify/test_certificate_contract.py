from openlithohub.verify.certifier import assemble_continuous_focus_certificate
from openlithohub.verify.types import (
    CertificateStatus,
    CertificateTarget,
    CoverageStatus,
    DependencyRecord,
    ParameterInterval,
    ProofLevel,
)


def _dep(level=ProofLevel.INTERVAL_CERTIFIED, satisfied=True):
    return DependencyRecord("gate", level, satisfied, "synthetic")


def _cert(**kw):
    args = dict(
        model_id="b04-test",
        input_sha256="0" * 64,
        parameters=(ParameterInterval("focus_nm", -10.0, 10.0),),
        continuous_focus_contour_upper_nm=0.9,
        continuous_focus_pairwise_diameter_upper_nm=1.8,
        tolerance_nm=1.0,
        coverage_status=CoverageStatus.ALL_COMPONENTS_COVERED,
        dependencies=(_dep(),),
    )
    args.update(kw)
    return assemble_continuous_focus_certificate(**args)


def test_certified_upper_bound_can_pass():
    assert _cert().status is CertificateStatus.PASS


def test_numerical_diagnostic_cannot_pass():
    c = _cert(dependencies=(_dep(ProofLevel.NUMERICAL_DIAGNOSTIC),))
    assert c.status is CertificateStatus.INCONCLUSIVE


def test_partial_cover_cannot_pass_for_extracted_epe():
    c = _cert(coverage_status=CoverageStatus.PARTIAL_COVER)
    assert c.status is CertificateStatus.INCONCLUSIVE


def test_level_set_stability_does_not_require_explicit_contour_cover():
    c = _cert(
        target=CertificateTarget.LEVEL_SET_STABILITY,
        coverage_status=CoverageStatus.INCONCLUSIVE,
    )
    assert c.status is CertificateStatus.PASS


def test_upper_bound_above_tolerance_is_not_fail():
    c = _cert(continuous_focus_contour_upper_nm=1.1)
    assert c.status is CertificateStatus.INCONCLUSIVE


def test_fail_requires_certified_violation_lower_bound():
    c = _cert(
        continuous_focus_contour_upper_nm=1.2,
        certified_violation_lower_nm=1.1,
    )
    assert c.status is CertificateStatus.FAIL
