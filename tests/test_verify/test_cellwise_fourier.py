import json
from pathlib import Path

import pytest

from openlithohub.verify.cellwise_fourier import (
    LayoutFourierEnvelope,
    local_cell_geometry,
    transfer_center_gradient_to_cell,
)

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "proof_artifacts" / "B04_K24_RunInterface_AliasFreeIntensityFourier_2026-09-10.npz"
CERT = (
    ROOT / "proof_artifacts" / "B04_Increment18_CellwiseFourierEnvelope_Certificate_2026-09-10.json"
)

# The alias-free Fourier grid is a >500 KB release/CI artifact; tests that
# load it skip when it is not mounted (sha256-pinned in README_B04_PATCH.md).
needs_snapshot = pytest.mark.skipif(
    not ART.exists(), reason="B04 alias-free intensity Fourier npz not installed"
)


@needs_snapshot
def test_actual_layout_fourier_hessian_is_coercive_scale():
    f = LayoutFourierEnvelope.load(ART)
    h = f.hessian_frobenius_upper_per_nm2()
    assert 0.0011 < h < 0.0013


@needs_snapshot
def test_actual_layout_conditioned_l3_repairs_generic_failure():
    f = LayoutFourierEnvelope.load(ART)
    l3 = f.third_derivative_tensor_upper_per_nm3()
    raw = json.loads(CERT.read_text())
    generic = raw["firewall_update"]["increment17_generic_kernel_l1_L3_upper_per_nm3"]
    assert l3 < 6e-5
    assert generic / l3 > 100


def test_continuous_active_cell_transfer_passes():
    raw = json.loads(CERT.read_text())
    ext = raw["active_cell_extension"]
    tr = transfer_center_gradient_to_cell(
        center_gradient_lower_per_nm=ext["min_center_gradient_lower_per_nm"],
        hessian_upper_per_nm2=raw["layout_conditioned_majorants"][
            "hessian_frobenius_upper_per_nm2"
        ],
        cell_radius_nm=ext["cell_radius_nm"],
    )
    assert tr.passes
    assert tr.cell_gradient_lower_per_nm > 0.0023


def test_certificate_keeps_streamed_construction_open():
    raw = json.loads(CERT.read_text())
    assert raw["status"] == "PASS_CONTINUOUS_ACTIVE_CELL_EXTENSION"
    assert (
        raw["ledger"]["production_streamed_construction_without_full_grid"] == "OPEN_IMPLEMENTATION"
    )


def test_actual_local_hidden_loop_scale_passes():
    raw = json.loads(CERT.read_text())
    loc = raw["local_cell_geometry"]
    assert loc["status"] == "PASS"
    assert loc["min_cell_gradient_lower_per_nm"] > 0.0058
    assert loc["max_cell_hessian_upper_per_nm2"] < 0.00061
    assert loc["min_hidden_loop_curvature_scale_nm"] > loc["cell_diameter_nm"]


def test_generic_local_geometry_helper_reproduces_worst_cell():
    raw = json.loads(CERT.read_text())
    loc = raw["local_cell_geometry"]
    worst = min(loc["cells"], key=lambda r: r["cell_gradient_lower_per_nm"])
    g = local_cell_geometry(
        center_gradient_lower_per_nm=worst["center_gradient_lower_per_nm"],
        center_hessian_upper_per_nm2=worst["center_hessian_frobenius_upper_per_nm2"],
        third_derivative_upper_per_nm3=raw["layout_conditioned_majorants"][
            "third_derivative_tensor_upper_per_nm3"
        ],
        cell_radius_nm=raw["active_cell_extension"]["cell_radius_nm"],
    )
    assert g.passes_transversality
    assert g.excludes_hidden_loop(cell_diameter_nm=loc["cell_diameter_nm"])
    assert abs(g.cell_gradient_lower_per_nm - worst["cell_gradient_lower_per_nm"]) < 1e-14
