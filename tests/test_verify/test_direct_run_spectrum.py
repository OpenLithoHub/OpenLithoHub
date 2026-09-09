import json
from pathlib import Path

import numpy as np
import pytest

from openlithohub.verify.interface_runs import HorizontalRun
from openlithohub.verify.run_spectrum import (
    RunSpectrumAccumulator,
    coherent_spectrum_from_layout,
)

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "proof_artifacts" / "B04_K24_DirectRunSpectrum_AliasFreeIntensity_2026-09-10.npz"
CERT = ROOT / "proof_artifacts" / "B04_Increment19_DirectRunSpectrum_Certificate_2026-09-10.json"

# The alias-free intensity grid is a >500 KB release/CI artifact; the test
# that loads it skips when it is not mounted (sha256-pinned in
# README_B04_PATCH.md).
needs_snapshot = pytest.mark.skipif(
    not ART.exists(), reason="B04 direct run-spectrum npz not installed"
)


def fixture_runs():
    return [HorizontalRun(y, 18, 54) for y in range(18, 54)]


def test_run_spectrum_matches_dense_fft_numerically():
    acc = RunSpectrumAccumulator.create(72)
    acc.extend(fixture_runs())
    got = acc.normalized_spectrum()
    mask = np.zeros((72, 72))
    mask[18:54, 18:54] = 1
    want = np.fft.fft2(mask) / (72 * 72)
    assert np.max(np.abs(got - want)) < 3e-15


@needs_snapshot
def test_coherent_spectrum_identity():
    z = np.load(ART, allow_pickle=False)
    got = coherent_spectrum_from_layout(
        normalized_kernel_spectrum=z["kernel_coeff_mid"],
        normalized_layout_spectrum=z["layout_coeff_mid"],
    )
    assert np.max(np.abs(got - z["coherent_coeff_mid"])) < 1e-14


def test_direct_artifact_removes_corrected_spatial_grid():
    raw = json.loads(CERT.read_text())
    assert raw["stream_contract"]["corrected_spatial_grid_materialized"] is False
    assert raw["status"] == "PASS_DIRECT_STREAMED_SPECTRAL_CONSTRUCTION"


def test_direct_spectrum_preserves_geometry():
    raw = json.loads(CERT.read_text())
    g = raw["continuous_geometry"]
    assert g["status"] == "PASS"
    assert g["global_cell_gradient_lower_per_nm"] > 0.0023
    assert g["local_cell_gradient_lower_per_nm"] > 0.0058
    assert g["min_hidden_loop_curvature_scale_nm"] > g["cell_diameter_nm"]


def test_direct_and_increment18_spectra_overlap():
    raw = json.loads(CERT.read_text())
    e = raw["equivalence_to_increment18"]
    assert e["coherent_disks_intersect"]
    assert e["intensity_disks_intersect"]
    assert e["coherent_coefficient_midpoint_max_difference"] < 1e-15
    assert e["intensity_coefficient_midpoint_max_difference"] < 1e-14
