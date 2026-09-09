import json
from pathlib import Path

import numpy as np
import pytest

from openlithohub.verify.interface_runs import (
    CyclicRowPrefix,
    encode_binary_row_runs,
    known_layout_far_coherent,
    run_compression_stats,
    split_runs_outside_chebyshev,
    zero_padded_horizontal_variation,
)

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "proof_artifacts" / "B04_K24_HaloKernelSnapshot_2026-09-09.npz"
CERT = ROOT / "proof_artifacts" / "B04_Increment16_RunInterface_Certificate_2026-09-09.json"

# The kernel snapshot is a >500 KB release/CI artifact (sha256-pinned in
# README_B04_PATCH.md); tests that need it skip when it is not mounted.
needs_snapshot = pytest.mark.skipif(
    not SNAP.exists(), reason="B04 kernel snapshot npz not installed"
)


def fixture_mask():
    m = np.zeros((72, 72), dtype=np.uint8)
    m[18:54, 18:54] = 1
    return m


def test_center_square_run_compression():
    runs = encode_binary_row_runs(fixture_mask())
    stats = run_compression_stats(runs)
    assert stats.occupied_pixels == 1296
    assert stats.horizontal_runs == 36
    assert stats.pixel_to_run_ratio == 36.0
    assert zero_padded_horizontal_variation(runs) == 72


@needs_snapshot
def test_actual_run_oracle_matches_pixel_sum_at_active_center():
    z = np.load(SNAP, allow_pickle=False)
    kernels = z["kernels_spatial"]
    prefix = CyclicRowPrefix.from_kernel_bank(kernels)
    runs = encode_binary_row_runs(fixture_mask())
    center = (17, 22)
    h = 8
    got = known_layout_far_coherent(prefix, runs, center_y=center[0], center_x=center[1], halo_px=h)
    naive = np.zeros(kernels.shape[0], dtype=np.complex128)
    c = 36
    yy, xx = np.nonzero(fixture_mask())
    for y, x in zip(yy, xx, strict=True):
        if max(abs(int(y) - center[0]), abs(int(x) - center[1])) <= h:
            continue
        yi = (c + center[0] - int(y)) % 72
        xi = (c + center[1] - int(x)) % 72
        naive += kernels[:, yi, xi].astype(np.complex128)
    assert np.max(np.abs(got - naive)) < 1e-15


def test_split_far_runs_is_interface_sized():
    runs = encode_binary_row_runs(fixture_mask())
    far = split_runs_outside_chebyshev(runs, center_y=17, center_x=22, halo_px=8)
    assert len(far) < 50


def test_frozen_certificate_interval_width_is_tiny():
    raw = json.loads(CERT.read_text())
    assert raw["mask"]["horizontal_runs"] == 36
    assert raw["mask"]["global_pixel_to_run_compression"] == 36.0
    assert max(r["max_exact_prefix_interval_width"] for r in raw["queries"]["rows"]) < 1e-14
