"""P-054 repo integration — backend capability / model-identity firewall.

Claim-firewall tests (audit §11):

- test_diagnostic_backend_cannot_certify_pass: a placeholder backend with
  synthetic "certified" dependency injection must NOT produce a PASS —
  the assembler fails closed;
- backend identity != capability identity: the canonical backend id may
  exist while its capability ceiling forbids certifying claims;
- a certifying dependency that omits its backend's capability fails
  closed once it names the backend;
- the frozen model commit of a certificate is whatever its ModelIdentity
  declares — never silently rewritten to the current HEAD or the legacy
  pin.
"""

import pytest

from openlithohub.verify.certifier import (
    LEGACY_PINNED_UPSTREAM,
    assemble_continuous_focus_certificate,
    replay_legacy_continuous_focus_certificate,
)
from openlithohub.verify.model_identity import ModelIdentity
from openlithohub.verify.source_native import (
    OutwardRoundedCPUBackend,
    assert_backend_supports,
)
from openlithohub.verify.types import (
    CertificateStatus,
    CertificateTarget,
    CertificationCapability,
    CoverageStatus,
    DependencyProvenance,
    DependencyRecord,
    ParameterInterval,
    ProofLevel,
)


def _identity() -> ModelIdentity:
    return ModelIdentity(
        repo="OpenLithoHub/OpenLithoHub",
        implementation_commit="1" * 40,
        model_schema="P054.frozen-arf37.v1",
        fixture_id="p054-arf37",
        fixture_manifest_sha256="c" * 64,
    )


def _certified_dependency(**kwargs) -> DependencyRecord:
    defaults = dict(
        name="field_enclosure",
        level=ProofLevel.INTERVAL_CERTIFIED,
        satisfied=True,
        method="interval evaluation",
        artifact_sha256="a" * 64,
        provenance=DependencyProvenance.IMPORTED_FROZEN,
    )
    defaults.update(kwargs)
    return DependencyRecord(**defaults)


def _assemble(**overrides):
    args = dict(
        model_id="openlithohub.source_native_full.hopkins.discrete",
        target=CertificateTarget.LEVEL_SET_STABILITY,
        input_sha256="b" * 64,
        parameters=(ParameterInterval("focus_nm", 0.0, 4.0),),
        continuous_focus_contour_upper_nm=1.0,
        continuous_focus_pairwise_diameter_upper_nm=None,
        tolerance_nm=3.0,
        coverage_status=CoverageStatus.ALL_COMPONENTS_COVERED,
        dependencies=(_certified_dependency(),),
        model_identity=_identity(),
    )
    args.update(overrides)
    return assemble_continuous_focus_certificate(**args)


def test_diagnostic_backend_cannot_certify_pass():
    # synthetic injection: a DIAGNOSTIC_ONLY backend claiming a
    # certifying-level dependency corrupts the proof package
    with pytest.raises(ValueError, match="ceilings at NUMERICAL-DIAGNOSTIC"):
        _assemble(
            dependencies=(
                _certified_dependency(
                    backend_id=OutwardRoundedCPUBackend.backend_id,
                    certification_capability=CertificationCapability.DIAGNOSTIC_ONLY,
                ),
            )
        )


def test_certifying_dependency_without_capability_fails_closed_when_backend_named():
    with pytest.raises(ValueError, match="without a certification capability"):
        _assemble(dependencies=(_certified_dependency(backend_id="some_interval_backend"),))


def test_legacy_dependency_without_backend_still_replays_via_explicit_api():
    # historical replays carry no backend attribution; they work ONLY
    # through the explicit legacy replay function
    cert = replay_legacy_continuous_focus_certificate(
        model_id="b04-legacy",
        target=CertificateTarget.LEVEL_SET_STABILITY,
        input_sha256="b" * 64,
        parameters=(ParameterInterval("focus_nm", 0.0, 4.0),),
        continuous_focus_contour_upper_nm=1.0,
        continuous_focus_pairwise_diameter_upper_nm=None,
        tolerance_nm=3.0,
        coverage_status=CoverageStatus.ALL_COMPONENTS_COVERED,
        dependencies=(_certified_dependency(provenance=None),),
    )
    assert cert.status is CertificateStatus.PASS
    assert cert.upstream_commit == LEGACY_PINNED_UPSTREAM
    assert "legacy replay without model identity" in cert.note


def test_rigorous_interval_capability_supports_certifying_level():
    dep = _certified_dependency(
        backend_id="real_interval_backend",
        certification_capability=CertificationCapability.RIGOROUS_INTERVAL,
        provenance=DependencyProvenance.BACKEND,
    )
    cert = _assemble(dependencies=(dep,))
    assert cert.status is CertificateStatus.PASS


def test_assert_backend_supports_fails_closed():
    backend = OutwardRoundedCPUBackend()
    assert backend.certification_capability is CertificationCapability.DIAGNOSTIC_ONLY
    assert backend.proof_level_ceiling is ProofLevel.NUMERICAL_DIAGNOSTIC
    # diagnostic ceiling: its own level is fine, certifying levels are not
    assert_backend_supports(backend, ProofLevel.NUMERICAL_DIAGNOSTIC)
    with pytest.raises(ValueError, match="cannot back INTERVAL-CERTIFIED"):
        assert_backend_supports(backend, ProofLevel.INTERVAL_CERTIFIED)


def test_frozen_model_commit_not_rewritten():
    identity = ModelIdentity(
        repo="OpenLithoHub/OpenLithoHub",
        implementation_commit="348fa5d86d5355465af98e2c4ce3deac60081a4c",
        model_schema="P054.frozen-arf37.v1",
        fixture_id="p054-arf37",
        fixture_manifest_sha256="c" * 64,
        paper_id="P-054",
        paper_doi="10.5281/zenodo.22843141",
    )
    cert = _assemble(model_identity=identity)
    # the certificate names exactly the declared frozen commit
    assert cert.upstream_commit == "348fa5d86d5355465af98e2c4ce3deac60081a4c"
    assert cert.model_identity is not None
    assert cert.model_identity.paper_id == "P-054"
    assert cert.model_identity.sha256() == identity.sha256()
    assert "legacy replay" not in cert.note


def test_model_identity_distinguishes_identity_from_capability():
    # the canonical backend id may exist with a non-certifying capability:
    # identity and capability are separate axes
    dep = _certified_dependency(
        backend_id="source_native_full",
        certification_capability=CertificationCapability.DIAGNOSTIC_ONLY,
        provenance=DependencyProvenance.BACKEND,
    )
    with pytest.raises(ValueError, match="ceilings"):
        _assemble(dependencies=(dep,))


def test_new_certificate_cannot_inherit_legacy_commit_by_omission():
    # H1: the normal assembler REQUIRES ModelIdentity — omission is a
    # TypeError, not a silent legacy fallback
    args = dict(
        model_id="b04-test",
        target=CertificateTarget.LEVEL_SET_STABILITY,
        input_sha256="b" * 64,
        parameters=(ParameterInterval("focus_nm", 0.0, 4.0),),
        continuous_focus_contour_upper_nm=1.0,
        continuous_focus_pairwise_diameter_upper_nm=None,
        tolerance_nm=3.0,
        coverage_status=CoverageStatus.ALL_COMPONENTS_COVERED,
        dependencies=(_certified_dependency(),),
    )
    with pytest.raises(TypeError, match="model_identity"):
        assemble_continuous_focus_certificate(**args)


def test_certifying_backend_dependency_requires_backend_id():
    with pytest.raises(ValueError, match="BACKEND provenance requires backend_id"):
        _assemble(
            dependencies=(
                _certified_dependency(
                    backend_id=None,
                    certification_capability=CertificationCapability.RIGOROUS_INTERVAL,
                    provenance=DependencyProvenance.BACKEND,
                ),
            )
        )


def test_certifying_backend_dependency_requires_capability():
    # the capability-ceiling check fires first: a BACKEND dependency without
    # a declared capability cannot state its ceiling, so it hard-fails
    with pytest.raises(ValueError, match="without a certification capability"):
        _assemble(
            dependencies=(
                _certified_dependency(
                    backend_id="some_interval_backend",
                    certification_capability=None,
                    provenance=DependencyProvenance.BACKEND,
                ),
            )
        )


def test_certifying_imported_dependency_requires_artifact_hash():
    with pytest.raises(ValueError, match="IMPORTED_FROZEN provenance requires"):
        _assemble(
            dependencies=(
                _certified_dependency(
                    artifact_sha256=None,
                    provenance=DependencyProvenance.IMPORTED_FROZEN,
                ),
            )
        )


def test_anonymous_certifying_dependency_rejected_on_new_path():
    with pytest.raises(ValueError, match="without provenance"):
        _assemble(dependencies=(_certified_dependency(provenance=None),))


def test_legacy_provenance_rejected_on_new_path():
    with pytest.raises(ValueError, match="LEGACY_REPLAY provenance is only"):
        _assemble(
            dependencies=(
                _certified_dependency(
                    provenance=DependencyProvenance.LEGACY_REPLAY,
                ),
            )
        )
