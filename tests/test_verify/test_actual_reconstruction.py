import json
from pathlib import Path

import pytest

from openlithohub.verify.mvp1 import certify_mvp1_manifest
from openlithohub.verify.reconstruction import replay_reconstruction_artifact
from openlithohub.verify.types import CertificateStatus, CertificateTarget

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / "proof_artifacts"
RECON = PROOF / "B04_Increment14_ContourReconstruction_Certificate_2026-09-09.json"
MANIFEST = PROOF / "B04_Increment12_MVP1_GoldenManifest.json"


def test_actual_reconstruction_artifact():
    r = replay_reconstruction_artifact(RECON)
    assert r.components == 1
    assert r.reconstructed_cells == 148
    assert r.polyline_vertices == 148
    assert r.polyline_segments == 148
    assert abs(r.nominal_reconstruction_upper_nm - 1.056042021172912) < 1e-12
    assert abs(r.continuous_focus_to_reconstruction_upper_nm - 2.0324780565696243) < 1e-12


def test_all_cells_have_positive_fixed_normal_margin():
    raw = json.loads(RECON.read_text(encoding="utf-8"))
    assert min(c["normal_derivative_lower_per_nm"] for c in raw["cells"]) > 0.0058
    assert max(c["graph_slope_upper"] for c in raw["cells"]) < 0.57


@pytest.mark.skipif(
    not (PROOF / "B04_ExactSource_ExpandedBand_Grid_2026-09-09.npz").exists(),
    reason="B04 expanded-band proof grid npz not installed",
)
def test_reconstruction_hash_is_bound_to_manifest(tmp_path):
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fake = tmp_path / "manifest.json"
    raw["proof_artifacts"]["reconstruction_sha256"] = "0" * 64
    fake.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="reconstruction artifact hash"):
        certify_mvp1_manifest(
            fake,
            artifact_dir=PROOF,
            tolerance_nm=2.1,
            target=CertificateTarget.EXTRACTED_CONTOUR_EPE,
        )


@pytest.mark.skipif(
    not (PROOF / "B04_ExactSource_ExpandedBand_Grid_2026-09-09.npz").exists(),
    reason="B04 expanded-band proof grid npz not installed",
)
def test_extracted_target_dependency_contains_reconstruction():
    c = certify_mvp1_manifest(
        MANIFEST,
        tolerance_nm=2.1,
        target=CertificateTarget.EXTRACTED_CONTOUR_EPE,
    )
    assert c.status is CertificateStatus.PASS
    assert any(d.name == "nominal contour reconstruction" for d in c.dependencies)
