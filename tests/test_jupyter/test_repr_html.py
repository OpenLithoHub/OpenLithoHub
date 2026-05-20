"""Tests for _repr_html_ methods on lithography result dataclasses."""

from __future__ import annotations

import pytest
import torch


def test_prediction_result_repr_html_contains_panel():
    from openlithohub.models.base import PredictionResult

    r = PredictionResult(mask=torch.zeros((16, 16)))
    html = r._repr_html_()
    assert "PredictionResult" in html
    assert "<table" in html


def test_mrc_result_repr_html_pass_badge():
    from openlithohub.benchmark.compliance.mrc import MRCResult

    r = MRCResult(
        passed=True,
        violation_count=0,
        violation_rate=0.0,
        violations=[],
    )
    html = r._repr_html_()
    assert "MRC" in html
    assert "PASS" in html
    assert "FAIL" not in html


def test_mrc_result_repr_html_fail_with_violations():
    from openlithohub.benchmark.compliance.mrc import MRCResult

    r = MRCResult(
        passed=False,
        violation_count=2,
        violation_rate=0.012,
        violations=[
            {"type": "width", "x_nm": 1.0, "y_nm": 2.0, "actual_nm": 30.0, "min_nm": 40.0},
            {"type": "spacing", "x_nm": 5.0, "y_nm": 6.0, "actual_nm": 25.0, "min_nm": 40.0},
        ],
        width_violation_count=1,
        spacing_violation_count=1,
    )
    html = r._repr_html_()
    assert "FAIL" in html
    assert "width" in html
    assert "spacing" in html


def test_curvilinear_mrc_repr_html():
    from openlithohub.benchmark.compliance.mrc import CurvilinearMRCResult

    r = CurvilinearMRCResult(
        passed=False,
        violation_count=1,
        curvature_violations=[{"x_nm": 0.0, "y_nm": 0.0, "radius_nm": 12.0}],
        area_violations=[],
        min_radius_observed_nm=12.0,
        min_area_observed_nm2=None,
    )
    html = r._repr_html_()
    assert "Curvilinear MRC" in html
    assert "FAIL" in html
    assert "12" in html


def test_drc_result_repr_html_summary():
    from openlithohub.benchmark.compliance.drc import DRCResult

    r = DRCResult(
        passed=True,
        violation_count=0,
        violations=[],
        rule_summary={"min_width": 0, "min_area": 0},
    )
    html = r._repr_html_()
    assert "DRC" in html
    assert "PASS" in html
    assert "min_width" in html


def test_monte_carlo_repr_html():
    from openlithohub.benchmark.metrics.monte_carlo import MonteCarloFailureResult

    r = MonteCarloFailureResult(
        bridge_probability=0.001,
        break_probability=0.0,
        failure_probability=0.001,
        num_trials=50,
    )
    html = r._repr_html_()
    assert "Monte Carlo" in html
    assert "0.1000%" in html or "0.10" in html


def test_simulator_result_repr_html():
    from openlithohub.simulators.base import SimulatorResult

    r = SimulatorResult(
        aerial=torch.zeros((8, 8)),
        resist=None,
        backend="hopkins",
        metadata={"kernels": 8},
    )
    html = r._repr_html_()
    assert "SimulatorResult" in html
    assert "hopkins" in html


def test_html_helpers_degrade_without_matplotlib(monkeypatch):
    import builtins

    import openlithohub.jupyter._html as html_mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = html_mod.mask_thumbnail_png_b64(torch.zeros((4, 4)))
    assert out is None


@pytest.mark.parametrize(
    "violations,expected",
    [
        ([], ""),
        ([{"x_nm": 1, "y_nm": 2}], "table"),
    ],
)
def test_violation_table_truncates(violations, expected):
    from openlithohub.jupyter._html import violation_table

    out = violation_table(violations)
    if expected:
        assert expected in out
    else:
        assert out == ""
