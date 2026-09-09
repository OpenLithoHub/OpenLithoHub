import json
from pathlib import Path

from openlithohub.verify.full_chip import TileProofStatus, aggregate_full_chip_status
from openlithohub.verify.halo import (
    CoreHaloGeometry,
    HaloStatus,
    HaloTailPoint,
    halo_duplicate_compute_ratio,
    minimal_halo_bracket,
    resolve_final_halo,
    socs_absolute_tail_upper,
    unrestricted_binary_tail_lower,
)
from openlithohub.verify.types import CertificateStatus

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "proof_artifacts" / "B04_Increment15_HaloTail_Certificate_2026-09-09.json"


def test_tail_upper_formula():
    u = socs_absolute_tail_upper([2.0], [3.0], [0.5])
    assert u == 2.0 * (2.0 * 3.0 * 0.5 + 0.25)


def test_unrestricted_binary_lower_is_positive():
    lower = unrestricted_binary_tail_lower([2.0], [0.5])
    assert lower > 0.0


def test_actual_halo_profile_brackets():
    raw = json.loads(ART.read_text(encoding="utf-8"))
    profile = tuple(
        HaloTailPoint(
            h_px=int(r["h_px"]),
            lower_intensity=float(r["unrestricted_binary_lower_intensity"]),
            upper_intensity=float(r["absolute_tail_upper_intensity"]),
        )
        for r in raw["profile"]
    )
    b = minimal_halo_bracket(profile, 0.002916392)
    assert b.necessary_h_min_px == 29
    assert b.certified_sufficient_h_px == 36
    assert b.status is HaloStatus.PARTIAL


def test_final_halo_keeps_physics_model_verifier_separate():
    raw = json.loads(ART.read_text(encoding="utf-8"))
    profile = tuple(
        HaloTailPoint(
            h_px=int(r["h_px"]),
            lower_intensity=float(r["unrestricted_binary_lower_intensity"]),
            upper_intensity=float(r["absolute_tail_upper_intensity"]),
        )
        for r in raw["profile"]
    )
    b = minimal_halo_bracket(profile, 0.002916392)
    final = resolve_final_halo(physics_h_px=56, model_h_px=64, verifier_bracket=b)
    assert final.final_h_px == 64
    assert final.verifier_h_px == 36


def test_core_geometry_does_not_infer_workflow_overlap():
    g = CoreHaloGeometry(
        core_width_px=512,
        core_height_px=512,
        loaded_width_px=584,
        loaded_height_px=584,
        margin_left_px=36,
        margin_right_px=36,
        margin_top_px=36,
        margin_bottom_px=36,
    )
    assert g.supports(36)
    assert not g.supports(37)


def test_duplicate_compute_ratio_matches_prompt_formula():
    assert abs(halo_duplicate_compute_ratio(512, 36) - 0.301025390625) < 1e-15


def test_full_chip_one_inconclusive_blocks_pass():
    records = [
        *(TileProofStatus(str(i), CertificateStatus.PASS) for i in range(9999)),
        TileProofStatus("9999", CertificateStatus.INCONCLUSIVE),
    ]
    agg = aggregate_full_chip_status(records)
    assert agg.status is CertificateStatus.INCONCLUSIVE
    assert agg.passed_tiles == 9999
    assert agg.inconclusive_tiles == 1


def test_full_chip_certified_violation_wins():
    records = [
        TileProofStatus("a", CertificateStatus.PASS),
        TileProofStatus("b", CertificateStatus.INCONCLUSIVE),
        TileProofStatus("c", CertificateStatus.FAIL),
    ]
    assert aggregate_full_chip_status(records).status is CertificateStatus.FAIL
