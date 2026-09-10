from dataclasses import dataclass

import numpy as np
import pytest

from openlithohub.verify.continuous_square_mask import (
    ContinuousSquareMaskContract,
    continuous_square_mask_transform,
    rectangle_fourier_transform,
)
from openlithohub.verify.global_finite_spectrum import (
    SpectralNode,
    global_finite_spectral_summary,
)
from openlithohub.verify.interface_runs import HorizontalRun


@dataclass
class FakeSource:
    runs: tuple[HorizontalRun, ...]
    shape: tuple[int, int] = (12, 16)
    pixel_size_nm: float = 2.0
    layout_hash: str = "continuous-fixture"
    backend_kind: str = "exact-vector"

    def iter_runs_for_bbox(self, bbox_px):
        assert bbox_px == (0, 0, self.shape[1], self.shape[0])
        return self.runs


def sinc_pi(x: float) -> float:
    if x == 0.0:
        return 1.0
    return float(np.sin(np.pi * x) / (np.pi * x))


def test_rectangle_transform_at_zero_is_area():
    got = rectangle_fourier_transform(
        x0_nm=2.0,
        x1_nm=10.0,
        y0_nm=4.0,
        y1_nm=7.0,
        fx_per_nm=0.0,
        fy_per_nm=0.0,
    )
    assert got == 24.0 + 0j


def test_square_aperture_lift_matches_point_moment_times_sinc():
    source = FakeSource(
        runs=(
            HorizontalRun(1, 2, 8),
            HorizontalRun(4, 1, 5),
            HorizontalRun(7, 6, 11),
        )
    )
    fx = 0.00317
    fy = -0.00241
    cont = continuous_square_mask_transform(source, fx_per_nm=fx, fy_per_nm=fy)
    summary = global_finite_spectral_summary(
        source,
        nodes=(SpectralNode(fx, fy),),
    )
    p = source.pixel_size_nm
    want = p * p * sinc_pi(p * fx) * sinc_pi(p * fy) * summary.moments[0]
    assert abs(cont - want) < 5e-13


def test_adjacent_pixels_equal_single_run_rectangle():
    source = FakeSource(
        runs=(HorizontalRun(3, 2, 7),),
        pixel_size_nm=4.0,
    )
    fx = 0.005
    fy = 0.003
    got = continuous_square_mask_transform(source, fx_per_nm=fx, fy_per_nm=fy)
    want = rectangle_fourier_transform(
        x0_nm=8.0,
        x1_nm=28.0,
        y0_nm=12.0,
        y1_nm=16.0,
        fx_per_nm=fx,
        fy_per_nm=fy,
    )
    assert abs(got - want) < 1e-13


def test_continuous_contract_rejects_periodic_or_omitted_geometry():
    with pytest.raises(ValueError, match="periodic"):
        ContinuousSquareMaskContract(
            layout_hash="x",
            backend_kind="x",
            source_shape_px=(8, 8),
            pixel_size_nm=1.0,
            periodic_mask_copies=True,
        )
    with pytest.raises(ValueError, match="complete"):
        ContinuousSquareMaskContract(
            layout_hash="x",
            backend_kind="x",
            source_shape_px=(8, 8),
            pixel_size_nm=1.0,
            omitted_spatial_exterior=True,
        )
