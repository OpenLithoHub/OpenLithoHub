import pytest

from openlithohub.verify.optical_raster_bridge import (
    ProcessWindowBridgeBudget,
    SOCSModeBridge,
    contour_transfer_from_intensity_bridge,
    socs_intensity_bridge_upper,
)


def test_socs_single_mode_formula():
    b = socs_intensity_bridge_upper(
        [SOCSModeBridge(2.0, 3.0, 0.5)],
        provenance="unit-test",
    )
    assert b.intensity_upper == 2.0 * (2.0 * 3.0 * 0.5 + 0.25)


def test_socs_multi_mode_uses_positive_sum():
    modes = [
        SOCSModeBridge(1.0, 2.0, 0.1),
        SOCSModeBridge(0.5, 1.0, 0.2),
    ]
    got = socs_intensity_bridge_upper(modes, provenance="test")
    want = (2 * 2.0 * 0.1 + 0.1**2) + 0.5 * (2 * 1.0 * 0.2 + 0.2**2)
    assert abs(got.intensity_upper - want) < 1e-15


def test_contour_transfer_is_delta_over_kappa():
    c = contour_transfer_from_intensity_bridge(
        intensity_upper=0.003,
        kappa_lower_per_nm=0.006,
    )
    assert c.contour_hausdorff_upper_nm == 0.5
    assert c.epe_upper_nm == 0.5
    assert c.paired_cd_error_upper_nm == 1.0


def test_contour_transfer_rejects_nominal_zero_kappa():
    with pytest.raises(ValueError):
        contour_transfer_from_intensity_bridge(
            intensity_upper=0.001,
            kappa_lower_per_nm=0.0,
        )


def test_process_budget_composes_additively():
    b = ProcessWindowBridgeBudget(
        process_contour_upper_nm=0.9,
        reconstruction_upper_nm=1.0,
        optical_raster_bridge_upper_nm=0.2,
        other_bridge_upper_nm=0.05,
    )
    assert abs(b.total_upper_nm - 2.15) < 1e-15
    assert b.passes(2.2)
    assert not b.passes(2.1)
