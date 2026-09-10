import json
from pathlib import Path

import pytest

from openlithohub.verify.coverage import replay_boundary_coverage
from openlithohub.verify.mvp1 import certify_mvp1_manifest
from openlithohub.verify.types import (
    CertificateStatus,
    CertificateTarget,
    CoverageStatus,
)

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / "proof_artifacts"
BOUNDARY = PROOF / "B04_Increment13_BoundaryCoverage_Certificate_2026-09-09.json"
MANIFEST = PROOF / "B04_Increment12_MVP1_GoldenManifest.json"
GRID = PROOF / "B04_ExactSource_ExpandedBand_Grid_2026-09-09.npz"

# Tests that replay the expanded-band proof grid skip when the release
# artifact is not mounted (mirrors test_increment11_replay).
needs_grid = pytest.mark.skipif(
    not GRID.exists(), reason="B04 expanded-band proof grid npz not installed"
)


def test_actual_boundary_coverage_replay_is_complete():
    r = replay_boundary_coverage(BOUNDARY)
    assert r.status is CoverageStatus.ALL_COMPONENTS_COVERED
    assert len(r.cells) == 164
    assert r.seeded_cells == 148
    assert r.curvature_certified_empty_cells == 16
    assert r.root_face_incidences == 296
    assert r.root_free_face_incidences == 360


@needs_grid
def test_mvp1_level_set_certificate_carries_complete_coverage():
    cert = certify_mvp1_manifest(MANIFEST, tolerance_nm=1.0)
    assert cert.status is CertificateStatus.PASS
    assert cert.coverage_status is CoverageStatus.ALL_COMPONENTS_COVERED


@needs_grid
def test_extracted_contour_target_passes_at_2_1_nm():
    cert = certify_mvp1_manifest(
        MANIFEST,
        tolerance_nm=2.1,
        target=CertificateTarget.EXTRACTED_CONTOUR_EPE,
    )
    assert cert.status is CertificateStatus.PASS
    assert cert.nominal_reconstruction_upper_nm is not None
    assert cert.continuous_focus_contour_upper_nm < 2.1


@needs_grid
def test_extracted_contour_target_is_inconclusive_at_2_0_nm():
    cert = certify_mvp1_manifest(
        MANIFEST,
        tolerance_nm=2.0,
        target=CertificateTarget.EXTRACTED_CONTOUR_EPE,
    )
    assert cert.status is CertificateStatus.INCONCLUSIVE


def test_boundary_manifest_has_no_inconclusive_cells():
    raw = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    assert raw["summary"]["inconclusive_cells"] == 0
    assert raw["summary"]["unique_root_edges"] == 148
    assert raw["summary"]["unique_root_free_edges"] == 328
    assert raw["summary"]["max_root_bracket_width_nm"] <= 1e-4
    assert raw["summary"]["min_root_transverse_derivative_margin_per_nm"] > 0.003
