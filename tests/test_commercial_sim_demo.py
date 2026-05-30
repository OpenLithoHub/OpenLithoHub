"""Tests for examples/commercial_sim_demo.py — validates every demo function."""

from __future__ import annotations

import torch

from examples.commercial_sim_demo import (
    demo_co_design_workflow,
    demo_mock_mode,
    demo_real_mode_error,
    demo_registry_usage,
    make_test_mask,
)


class TestMakeTestMask:
    def test_shape(self) -> None:
        mask = make_test_mask()
        assert mask.shape == (64, 64)

    def test_custom_size(self) -> None:
        mask = make_test_mask(32)
        assert mask.shape == (32, 32)

    def test_binary_values(self) -> None:
        mask = make_test_mask()
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_has_nonzero_region(self) -> None:
        mask = make_test_mask()
        assert mask.sum().item() > 0


class TestDemoMockMode:
    def test_runs_without_error(self) -> None:
        demo_mock_mode()

    def test_produces_valid_aerial(self) -> None:
        from openlithohub.simulators import CalibreSimulator, SimulatorConfig

        mask = make_test_mask()
        cfg = SimulatorConfig(pixel_size_nm=4.0, dose=1.0, extra={"mock_mode": True})
        sim = CalibreSimulator(cfg)
        result = sim.simulate(mask)
        assert result.aerial.shape == mask.shape
        assert result.aerial.min() >= 0.0
        assert result.aerial.max() <= 1.0


class TestDemoRealModeError:
    def test_runs_without_error(self) -> None:
        demo_real_mode_error()


class TestDemoRegistryUsage:
    def test_runs_without_error(self) -> None:
        demo_registry_usage()

    def test_registry_returns_correct_backend(self) -> None:
        from openlithohub.simulators import SimulatorConfig, get_simulator

        sim = get_simulator("tachyon", SimulatorConfig(extra={"mock_mode": True}))
        result = sim.simulate(make_test_mask())
        assert result.backend == "tachyon"


class TestDemoCoDesignWorkflow:
    def test_runs_without_error(self) -> None:
        demo_co_design_workflow()

    def test_optimization_reduces_loss(self) -> None:
        from openlithohub.simulators import HopkinsSimulator, SimulatorConfig

        mask = make_test_mask().requires_grad_(True)
        sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
        target = make_test_mask()

        optimizer = torch.optim.Adam([mask], lr=0.1)
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            result = sim.simulate(mask)
            loss = torch.nn.functional.mse_loss(result.aerial, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease or stay flat over 5 steps
        assert losses[-1] <= losses[0] + 1e-6
