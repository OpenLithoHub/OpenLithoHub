"""Tests for the PINN-based differentiable resist model."""

from __future__ import annotations

import pytest
import torch

from openlithohub.models.resist_pinn import (
    ResistBenchmark,
    ResistCalibrator,
    ResistPhysicsConstraints,
    ResistPINN,
)


def _make_dose_resist(
    batch: int = 4,
    h: int = 32,
    w: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    dose = torch.rand(batch, 1, h, w)
    resist = (dose > 0.5).float()
    return dose, resist


@pytest.fixture
def pinn() -> ResistPINN:
    return ResistPINN(base_channels=8, depth=2)


@pytest.fixture
def dose_resist() -> tuple[torch.Tensor, torch.Tensor]:
    return _make_dose_resist()


def test_output_in_unit_interval(pinn: ResistPINN) -> None:
    dose, _ = _make_dose_resist()
    with torch.no_grad():
        out = pinn(dose)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_output_shape_matches_input(pinn: ResistPINN) -> None:
    dose, _ = _make_dose_resist(batch=2, h=16, w=16)
    with torch.no_grad():
        out = pinn(dose)
    assert out.shape == dose.shape


def test_monotonicity_loss_penalizes_violations() -> None:
    constraints = ResistPhysicsConstraints(monotonicity_weight=1.0)
    dose = torch.tensor([[0.2, 0.4, 0.6, 0.8]]).unsqueeze(0).unsqueeze(0)
    good_pred = torch.tensor([[0.1, 0.3, 0.5, 0.7]]).unsqueeze(0).unsqueeze(0)
    bad_pred = torch.tensor([[0.7, 0.5, 0.3, 0.1]]).unsqueeze(0).unsqueeze(0)
    good_loss = constraints.monotonicity_loss(good_pred, dose)
    bad_loss = constraints.monotonicity_loss(bad_pred, dose)
    assert bad_loss > good_loss


def test_boundary_loss_penalizes_out_of_range() -> None:
    constraints = ResistPhysicsConstraints(boundary_weight=1.0)
    in_range = torch.tensor([[[[0.2, 0.5, 0.8]]]])
    out_range = torch.tensor([[[[-0.5, 0.5, 1.5]]]])
    good_loss = constraints.boundary_loss(in_range)
    bad_loss = constraints.boundary_loss(out_range)
    assert bad_loss > good_loss
    assert good_loss.item() == pytest.approx(0.0)


def test_physics_loss_combines_constraints(pinn: ResistPINN) -> None:
    dose = torch.rand(2, 1, 8, 8)
    prediction = torch.sigmoid(torch.randn(2, 1, 8, 8))
    loss = pinn.physics_loss(dose, prediction)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_train_step_updates_parameters(pinn: ResistPINN) -> None:
    dose, resist = _make_dose_resist(batch=2, h=16, w=16)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-2)
    params_before = [p.clone() for p in pinn.parameters()]
    metrics = pinn.train_step(dose, resist, optimizer)
    any_changed = any(
        not torch.equal(p_before, p_current)
        for p_before, p_current in zip(params_before, pinn.parameters(), strict=True)
    )
    assert any_changed, "Parameters should change after a train step"
    assert "total" in metrics
    assert "data" in metrics
    assert "physics" in metrics


def test_calibrator_trains_and_evaluates() -> None:
    pinn = ResistPINN(base_channels=8, depth=2)
    calibrator = ResistCalibrator(model=pinn)
    dose, resist = _make_dose_resist(batch=4, h=16, w=16)
    history = calibrator.calibrate(dose, resist, n_epochs=3, lr=1e-3)
    assert len(history) == 3
    assert all("total" in h for h in history)
    metrics = calibrator.evaluate(dose, resist)
    assert "mse" in metrics
    assert "edge_placement_error" in metrics
    assert metrics["mse"] >= 0.0


def test_benchmark_compares_threshold_vs_pinn() -> None:
    pinn = ResistPINN(base_channels=8, depth=2)
    benchmark = ResistBenchmark(pinn=pinn)
    dose, gt = _make_dose_resist(batch=2, h=16, w=16)
    results = benchmark.compare(dose, gt, threshold=0.5)
    assert "hard_threshold" in results
    assert "pinn" in results
    for method in ("hard_threshold", "pinn"):
        m = results[method]
        assert "mse" in m
        assert "edge_sharpness" in m
        assert "process_window_accuracy" in m


def test_deterministic_with_seed() -> None:
    def _run() -> torch.Tensor:
        torch.manual_seed(123)
        model = ResistPINN(base_channels=8, depth=2)
        dose = torch.rand(1, 1, 16, 16)
        with torch.no_grad():
            return model(dose)

    out1 = _run()
    out2 = _run()
    assert torch.equal(out1, out2)
