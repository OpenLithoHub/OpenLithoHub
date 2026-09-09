from pathlib import Path

from openlithohub.verify.mvp1 import certify_mvp1_manifest
from openlithohub.verify.types import CertificateStatus, CertificateTarget


def test_increment12_mvp1_manifest_replays_and_passes_one_nm_gate():
    root = Path(__file__).resolve().parents[2]
    manifest = root / "proof_artifacts" / "B04_Increment12_MVP1_GoldenManifest.json"
    cert = certify_mvp1_manifest(manifest, tolerance_nm=1.0)
    assert cert.target is CertificateTarget.LEVEL_SET_STABILITY
    assert cert.status is CertificateStatus.PASS
    assert cert.continuous_focus_contour_upper_nm < 1.0


def test_increment12_mvp1_manifest_is_inconclusive_for_tighter_gate():
    root = Path(__file__).resolve().parents[2]
    manifest = root / "proof_artifacts" / "B04_Increment12_MVP1_GoldenManifest.json"
    cert = certify_mvp1_manifest(manifest, tolerance_nm=0.9)
    assert cert.status is CertificateStatus.INCONCLUSIVE
