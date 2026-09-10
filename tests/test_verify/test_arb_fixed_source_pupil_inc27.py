from dataclasses import dataclass

import pytest

pytest.importorskip("flint")

from openlithohub.verify.arb_fixed_source_pupil import (
    FixedSourcePupilParams,
    certified_fixed_source_coherent_field,
)
from openlithohub.verify.interface_runs import HorizontalRun


@dataclass
class TinySource:
    runs: tuple[HorizontalRun, ...]
    shape: tuple[int, int] = (8, 8)
    pixel_size_nm: float = 8.0

    def iter_runs_for_bbox(self, bbox_px):
        assert bbox_px == (0, 0, 8, 8)
        return self.runs


def test_empty_mask_field_is_certified_zero():
    source = TinySource(runs=())
    cert = certified_fixed_source_coherent_field(
        source,
        params=FixedSourcePupilParams(
            wavelength_nm=193.0,
            na=1.35,
            source_fx_per_nm=0.0,
            source_fy_per_nm=0.0,
        ),
        observation_x_nm=16.0,
        observation_y_nm=16.0,
        precision_bits=80,
        abs_tol_decimal="1e-12",
        rel_tol_decimal="1e-12",
        eval_limit=5000,
        depth_limit=12,
    )
    assert cert.finite
    assert abs(cert.real_mid) <= cert.real_rad
    assert abs(cert.imag_mid) <= cert.imag_rad
    assert cert.pupil_integral_certified
    assert not cert.source_discretization_certified


def test_one_run_certificate_is_finite_and_precision_replays_overlap():
    source = TinySource(runs=(HorizontalRun(3, 2, 6),))
    params = FixedSourcePupilParams(
        wavelength_nm=193.0,
        na=1.35,
        source_fx_per_nm=0.0005,
        source_fy_per_nm=-0.0003,
        defocus_nm=5.0,
    )
    kwargs = dict(
        source=source,
        params=params,
        observation_x_nm=32.0,
        observation_y_nm=28.0,
        abs_tol_decimal="1e-8",
        rel_tol_decimal="1e-8",
        eval_limit=30000,
        depth_limit=20,
    )
    c80 = certified_fixed_source_coherent_field(precision_bits=80, **kwargs)
    c112 = certified_fixed_source_coherent_field(precision_bits=112, **kwargs)
    assert c80.finite and c112.finite
    assert c80.max_component_radius < 1e-5
    assert c112.max_component_radius < 1e-5
    assert abs(c80.real_mid - c112.real_mid) <= c80.real_rad + c112.real_rad + 1e-12
    assert abs(c80.imag_mid - c112.imag_mid) <= c80.imag_rad + c112.imag_rad + 1e-12
