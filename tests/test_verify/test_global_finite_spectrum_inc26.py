from dataclasses import dataclass

import numpy as np
import pytest

from openlithohub.verify.global_finite_spectrum import (
    GlobalFiniteSpectralContract,
    PlaneWaveMode,
    SpectralNode,
    evaluate_intensity_jet,
    global_finite_spectral_summary,
)
from openlithohub.verify.interface_runs import HorizontalRun


@dataclass
class FakeSource:
    runs: tuple[HorizontalRun, ...]
    shape: tuple[int, int] = (10, 12)
    layout_hash: str = "fixture"
    pixel_size_nm: float = 2.0
    backend_kind: str = "exact-vector"

    def iter_runs_for_bbox(self, bbox_px):
        assert bbox_px == (0, 0, self.shape[1], self.shape[0])
        return self.runs


def explicit_moment(source: FakeSource, node: SpectralNode) -> complex:
    total = 0j
    p = source.pixel_size_nm
    for run in source.runs:
        for x in range(run.x0, run.x1):
            xn = (x + 0.5) * p
            yn = (run.y + 0.5) * p
            total += np.exp(-2j * np.pi * (node.fx_per_nm * xn + node.fy_per_nm * yn))
    return complex(total)


def test_arbitrary_frequency_run_moments_match_explicit_pixels():
    source = FakeSource(
        runs=(
            HorizontalRun(1, 2, 8),
            HorizontalRun(4, 1, 5),
            HorizontalRun(7, 6, 11),
        )
    )
    nodes = (
        SpectralNode(0.0, 0.0),
        SpectralNode(0.00317, -0.00241),
        SpectralNode(0.00611, 0.00193),
    )
    got = global_finite_spectral_summary(source, nodes=nodes)
    want = np.array([explicit_moment(source, n) for n in nodes])
    assert np.max(np.abs(got.moments - want)) < 2e-13
    assert got.occupied_pixels == 15
    assert got.contract.spatial_halo_error_upper == 0.0


def test_generic_physical_frequency_does_not_alias_576px_shift():
    node = SpectralNode(0.00317, 0.00121)
    a = FakeSource(
        runs=(HorizontalRun(2, 1, 5),),
        shape=(8, 600),
        pixel_size_nm=1.0,
    )
    b = FakeSource(
        runs=(HorizontalRun(2, 577, 581),),
        shape=(8, 1200),
        pixel_size_nm=1.0,
    )
    ma = global_finite_spectral_summary(a, nodes=(node,)).moments[0]
    mb = global_finite_spectral_summary(b, nodes=(node,)).moments[0]
    assert abs(ma - mb) > 1e-3


def test_zero_halo_contract_rejects_omitted_exterior():
    with pytest.raises(ValueError, match="omitted"):
        GlobalFiniteSpectralContract(
            layout_hash="x",
            backend_kind="x",
            source_shape_px=(8, 8),
            pixel_size_nm=1.0,
            omitted_spatial_exterior=True,
        )


def test_single_plane_wave_intensity_is_spatially_constant():
    source = FakeSource(runs=(HorizontalRun(1, 2, 8),))
    summary = global_finite_spectral_summary(source, nodes=(SpectralNode(0.004, 0.001),))
    mode = PlaneWaveMode(
        weight=2.0,
        amplitudes=np.array([0.25 + 0.5j]),
    )
    pts = np.array([[0.0, 0.0], [7.5, 2.0], [13.0, 11.0]])
    jet = evaluate_intensity_jet(summary, (mode,), points_nm=pts)
    assert np.max(np.abs(jet.value - jet.value[0])) < 1e-12
    assert np.max(np.abs(jet.dx)) < 1e-12
    assert np.max(np.abs(jet.dy)) < 1e-12
    assert np.max(np.abs(jet.dxx)) < 1e-12
    assert np.max(np.abs(jet.dxy)) < 1e-12
    assert np.max(np.abs(jet.dyy)) < 1e-12


def test_two_plane_wave_gradient_matches_centered_difference():
    source = FakeSource(runs=(HorizontalRun(1, 1, 7), HorizontalRun(5, 3, 9)))
    nodes = (SpectralNode(0.002, 0.001), SpectralNode(-0.003, 0.004))
    summary = global_finite_spectral_summary(source, nodes=nodes)
    mode = PlaneWaveMode(
        weight=1.0,
        amplitudes=np.array([0.4 + 0.1j, -0.2 + 0.3j]),
    )
    p = np.array([[6.0, 4.0]])
    jet = evaluate_intensity_jet(summary, (mode,), points_nm=p)
    h = 1e-4
    px = np.array([[6.0 + h, 4.0], [6.0 - h, 4.0]])
    vals = evaluate_intensity_jet(summary, (mode,), points_nm=px).value
    fd = (vals[0] - vals[1]) / (2 * h)
    assert abs(jet.dx[0] - fd) < 2e-7
