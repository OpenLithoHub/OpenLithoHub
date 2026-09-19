"""P-054 claim firewall — what the frozen profile may and may not claim.

Audit §11 checks that are decidable offline:
- the profile never claims continuous Hopkins equivalence or foundry
  calibration (fixture honesty flags + manifest does_not_prove);
- the frozen model commit is never rewritten to the current HEAD;
- proof-level downgrade is a hard error (capability ceiling, PR-1).
"""

import json
from pathlib import Path

import pytest

from openlithohub.verify.certifier import assemble_continuous_focus_certificate
from openlithohub.verify.model_identity import ModelIdentity
from openlithohub.verify.phase_diagram import load_frozen_phase_diagram
from openlithohub.verify.source_native import OutwardRoundedCPUBackend
from openlithohub.verify.types import (
    CertificateTarget,
    CertificationCapability,
    CoverageStatus,
    DependencyRecord,
    ParameterInterval,
    ProofLevel,
)

ROOT = Path(__file__).resolve().parents[2]


def _fixture():
    return json.loads((ROOT / "proof_artifacts" / "p054" / "fixture.json").read_text())


def _manifest():
    return json.loads((ROOT / "proof_artifacts" / "p054" / "manifest.json").read_text())


def test_p054_profile_does_not_claim_continuous_hopkins():
    fixture = _fixture()
    assert fixture["honesty"]["continuous_hopkins_equivalent"] is False
    assert fixture["honesty"]["continuous_source_integral_certified"] is False
    assert "continuous Hopkins operator equivalence" in _manifest()["does_not_prove"]


def test_p054_profile_does_not_claim_foundry_calibration():
    fixture = _fixture()
    assert fixture["honesty"]["foundry_calibrated"] is False
    assert fixture["honesty"]["wafer_process_qualified"] is False
    assert "foundry or process qualification" in _manifest()["does_not_prove"]


def test_manifest_pins_the_frozen_commit_not_head():
    manifest = _manifest()
    assert manifest["implementation_commit"] == "348fa5d86d5355465af98e2c4ce3deac60081a4c"
    pd = load_frozen_phase_diagram("p054-arf37")
    assert pd.implementation_commit == manifest["implementation_commit"]


def test_model_identity_carries_paper_doi():
    identity = ModelIdentity(
        repo="OpenLithoHub/OpenLithoHub",
        implementation_commit=_manifest()["implementation_commit"],
        model_schema=_manifest()["model_schema"],
        fixture_id="p054-arf37",
        fixture_manifest_sha256="d" * 64,
        paper_id="P-054",
        paper_doi="10.5281/zenodo.22843141",
    )
    assert identity.paper_doi == "10.5281/zenodo.22843141"


def test_diagnostic_backend_cannot_certify_pass():
    # PHASE-diagram certificates replay as imported frozen certs; a
    # diagnostic backend can never be the dependency that proves them.
    with pytest.raises(ValueError, match="ceilings"):
        assemble_continuous_focus_certificate(
            model_id="openlithohub.source_native_full.hopkins.discrete.p054-arf37",
            target=CertificateTarget.FIXED_TARGET_TOPOLOGY,
            input_sha256="e" * 64,
            parameters=(ParameterInterval("focus_nm", 30.0, 40.0),),
            continuous_focus_contour_upper_nm=1.0,
            continuous_focus_pairwise_diameter_upper_nm=None,
            tolerance_nm=3.0,
            coverage_status=CoverageStatus.ALL_COMPONENTS_COVERED,
            dependencies=(
                DependencyRecord(
                    name="phase_replay",
                    level=ProofLevel.INTERVAL_CERTIFIED,
                    satisfied=True,
                    method="frozen artifact replay",
                    backend_id=OutwardRoundedCPUBackend.backend_id,
                    certification_capability=CertificationCapability.DIAGNOSTIC_ONLY,
                ),
            ),
            model_identity=ModelIdentity(
                repo="OpenLithoHub/OpenLithoHub",
                implementation_commit="348fa5d86d5355465af98e2c4ce3deac60081a4c",
                model_schema="P054.frozen-arf37.v1",
                fixture_id="p054-arf37",
                fixture_manifest_sha256="c" * 64,
            ),
        )
