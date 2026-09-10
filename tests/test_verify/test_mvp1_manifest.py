from pathlib import Path

import pytest

from openlithohub.verify.mvp1 import certify_mvp1_manifest
from openlithohub.verify.types import CertificateStatus, CertificateTarget

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "proof_artifacts" / "B04_ExactSource_ExpandedBand_Grid_2026-09-09.npz"
MANIFEST = ROOT / "proof_artifacts" / "B04_Increment12_MVP1_GoldenManifest.json"

# The expanded-band proof grid may be absent in fresh clones that predate
# the proof-grid tracking commit; tests that replay it skip when unmounted.
needs_grid = pytest.mark.skipif(
    not GRID.exists(), reason="B04 expanded-band proof grid npz not installed"
)


@needs_grid
def test_increment12_mvp1_manifest_replays_and_passes_one_nm_gate():
    manifest = ROOT / "proof_artifacts" / "B04_Increment12_MVP1_GoldenManifest.json"
    cert = certify_mvp1_manifest(manifest, tolerance_nm=1.0)
    assert cert.target is CertificateTarget.LEVEL_SET_STABILITY
    assert cert.status is CertificateStatus.PASS
    assert cert.continuous_focus_contour_upper_nm < 1.0


@needs_grid
def test_increment12_mvp1_manifest_is_inconclusive_for_tighter_gate():
    manifest = ROOT / "proof_artifacts" / "B04_Increment12_MVP1_GoldenManifest.json"
    cert = certify_mvp1_manifest(manifest, tolerance_nm=0.9)
    assert cert.status is CertificateStatus.INCONCLUSIVE
