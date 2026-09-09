"""Continuous-metric firewall (prompt §B04-E / §22-I).

The shipped raster ``epe_max_nm`` semantics are the leaderboard contract.
Theorem-facing continuous results live in a separate schema and must never
overwrite or alias the raster metrics.
"""

import torch

from openlithohub.benchmark.metrics.epe import compute_epe
from openlithohub.verify.types import (
    CertificateStatus,
    CertificateTarget,
    ContinuousFocusCertificate,
    CoverageStatus,
)

# Metric keys owned by the shipped raster benchmark schema. The continuous
# certificate schema must not collide with any of them.
RASTER_METRIC_KEYS = {
    "epe_nm",
    "epe_max_nm",
    "epe_mean_nm",
    "epe_wafer_nm",
    "pvband_nm",
    "l2_error_nm2",
}

CONTINUOUS_CERTIFICATE_KEYS = {
    "continuous_focus_contour_upper_nm",
    "continuous_focus_pairwise_diameter_upper_nm",
    "nominal_reconstruction_upper_nm",
    "tolerance_nm",
    "certificate_status_alias_guard",
}


def _certificate() -> ContinuousFocusCertificate:
    return ContinuousFocusCertificate(
        target=CertificateTarget.LEVEL_SET_STABILITY,
        upstream_commit="0" * 40,
        model_id="firewall-test",
        input_sha256="0" * 64,
        parameters=(),
        continuous_focus_contour_upper_nm=0.5,
        continuous_focus_pairwise_diameter_upper_nm=None,
        tolerance_nm=1.0,
        coverage_status=CoverageStatus.INCONCLUSIVE,
        dependencies=(),
        status=CertificateStatus.INCONCLUSIVE,
    )


def test_continuous_schema_uses_independent_field_names():
    keys = set(_certificate().to_dict())
    assert keys & RASTER_METRIC_KEYS == set(), (
        "continuous certificate must not shadow raster metric names"
    )
    expected = CONTINUOUS_CERTIFICATE_KEYS - {"certificate_status_alias_guard"}
    assert expected <= keys


def test_continuous_upper_bound_is_reported_separately_from_raster_epe():
    """`continuous_focus_contour_upper_nm` exists only on the certificate."""
    layout = torch.zeros(16, 16)
    raster = compute_epe(layout, layout)
    assert not hasattr(raster, "status"), "raster EPE must not be a certificate"
    cert = _certificate()
    assert cert.continuous_focus_contour_upper_nm == 0.5
    assert cert.status is CertificateStatus.INCONCLUSIVE
