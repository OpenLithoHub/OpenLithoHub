import json
from pathlib import Path

import numpy as np
import pytest

from openlithohub.verify.derivative_runs import (
    DerivativeRunPrefixArtifact,
    JetComponent,
    center_jet_certificate_from_json,
)
from openlithohub.verify.interface_runs import encode_binary_row_runs

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "proof_artifacts" / "B04_K24_DerivativeRunPrefixIntervalSnapshot_2026-09-10.npz"
CERT = ROOT / "proof_artifacts" / "B04_Increment17_DerivativeRunPrefix_Certificate_2026-09-10.json"

# The interval snapshot is a >500 KB release/CI artifact; tests that load
# it skip when it is not mounted (sha256-pinned in README_B04_PATCH.md).
needs_snapshot = pytest.mark.skipif(
    not PREFIX.exists(), reason="B04 derivative run-prefix snapshot npz not installed"
)


def fixture_mask():
    m = np.zeros((72, 72), dtype=np.uint8)
    m[18:54, 18:54] = 1
    return m


@needs_snapshot
def test_derivative_prefix_artifact_loads_all_jet_components():
    art = DerivativeRunPrefixArtifact.load(PREFIX)
    assert art.grid_n == 72
    for comp in JetComponent:
        assert f"{comp.value}_real_lo" in art.arrays
        assert f"{comp.value}_imag_hi" in art.arrays


@needs_snapshot
def test_far_derivative_query_has_tiny_width():
    art = DerivativeRunPrefixArtifact.load(PREFIX)
    runs = encode_binary_row_runs(fixture_mask())
    q = art.sum_far_runs(
        component=JetComponent.DXY,
        runs=runs,
        center_y=17,
        center_x=22,
        halo_px=8,
    )
    assert float(np.max(q.real_hi - q.real_lo)) < 1e-12
    assert float(np.max(q.imag_hi - q.imag_lo)) < 1e-12


def test_actual_center_jet_certificate_is_pass_but_cell_extension_open():
    c = center_jet_certificate_from_json(CERT, halo_px=8)
    assert c.status == "PASS"
    assert c.min_center_gradient_lower_per_nm > 0.009
    assert c.continuous_cell_status == "OPEN"
    assert c.generic_l3_upper_per_nm3 > 0.005


def test_certificate_records_all_six_intensity_jet_widths():
    raw = json.loads(CERT.read_text())
    row = {int(r["h_px"]): r for r in raw["queries"]["halos"]}[8]
    widths = row["max_intensity_jet_interval_width"]
    assert set(widths) == {"I", "Ix", "Iy", "Ixx", "Ixy", "Iyy"}
    assert widths["I"] < 1e-10
    assert widths["Ixx"] < 1e-12
