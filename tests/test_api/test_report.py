"""Tests for the `Report` aggregate."""

from __future__ import annotations

import json

import pytest
import torch

from openlithohub import LitheEngine, Mask, Report
from openlithohub.benchmark.compliance.drc import check_drc
from openlithohub.benchmark.metrics.epe import compute_epe


def test_evaluate_returns_report(sample_design: torch.Tensor, sample_mask: torch.Tensor) -> None:
    engine = LitheEngine(model="dummy-identity")
    pred = Mask.from_tensor(sample_mask)
    target = Mask.from_tensor(sample_design)
    report = engine.evaluate(pred, target)
    assert isinstance(report, Report)
    assert report.model_name == "dummy-identity"
    assert report.pixel_size_nm == 1.0


def test_flat_fields_match_underlying_calls(
    sample_design: torch.Tensor, sample_mask: torch.Tensor
) -> None:
    """Flat numerical fields should be exact projections of the underlying functional outputs."""
    engine = LitheEngine(model="dummy-identity")
    pred = Mask.from_tensor(sample_mask)
    target = Mask.from_tensor(sample_design)
    report = engine.evaluate(pred, target)

    epe = compute_epe(pred.tensor, target.tensor, pixel_size_nm=1.0)
    drc = check_drc(pred.tensor, pixel_size_nm=1.0)

    assert report.epe_mean_nm == pytest.approx(float(epe["epe_mean_nm"]))
    assert report.epe_max_nm == pytest.approx(float(epe["epe_max_nm"]))
    assert report.drc_violations == drc.violation_count
    assert report.drc_passed == drc.passed


def test_report_to_dict_is_json_serializable(
    sample_design: torch.Tensor, sample_mask: torch.Tensor
) -> None:
    engine = LitheEngine(model="dummy-identity")
    pred = Mask.from_tensor(sample_mask)
    target = Mask.from_tensor(sample_design)
    report = engine.evaluate(pred, target)
    payload = report.to_dict()
    # asdict drops dataclass identity but preserves fields; tensors are not
    # part of the report so the whole tree should round-trip through JSON
    # once we accept that lists of dicts are fine.
    json.dumps(payload, default=str)


def test_curvilinear_mrc_only_for_curvilinear_models(
    sample_design: torch.Tensor, sample_mask: torch.Tensor
) -> None:
    engine = LitheEngine(model="dummy-identity")
    report = engine.evaluate(Mask.from_tensor(sample_mask), Mask.from_tensor(sample_design))
    assert report.raw_curvilinear_mrc is None  # dummy-identity is Manhattan


def test_evaluate_rejects_shape_mismatch() -> None:
    engine = LitheEngine(model="dummy-identity")
    pred = Mask.from_tensor(torch.zeros(64, 64))
    target = Mask.from_tensor(torch.zeros(32, 32))
    with pytest.raises(ValueError, match="shape mismatch"):
        engine.evaluate(pred, target)


def test_evaluate_rejects_pixel_size_mismatch() -> None:
    """EPE is reported in nm; comparing two masks at different pitches without
    resampling silently produces wrong physical numbers."""
    engine = LitheEngine(model="dummy-identity")
    pred = Mask.from_tensor(torch.zeros(64, 64), pixel_size_nm=1.0)
    target = Mask.from_tensor(torch.zeros(64, 64), pixel_size_nm=0.5)
    with pytest.raises(ValueError, match="pixel_size_nm mismatch"):
        engine.evaluate(pred, target)


def test_evaluate_accepts_raw_tensors(
    sample_design: torch.Tensor, sample_mask: torch.Tensor
) -> None:
    engine = LitheEngine(model="dummy-identity")
    report = engine.evaluate(sample_mask, sample_design)
    assert isinstance(report, Report)
