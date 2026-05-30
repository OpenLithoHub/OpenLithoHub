"""Tests for DiffCFD process simulation adapters (P3).

DiffCFD v0.7.0 is installed in the test environment, so these tests
exercise the real solvers end-to-end.
"""

from __future__ import annotations

import math

import pytest
import torch

from openlithohub.simulators.base import SimulatorConfig
from openlithohub.simulators.registry import get_simulator

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class TestDiffCFDLithoSimulator:
    def test_loads_via_registry(self):
        sim = get_simulator("diffcfd_litho")
        assert sim.name == "diffcfd_litho"
        assert sim.differentiable is True

    def test_simulate_returns_result(self):
        config = SimulatorConfig(
            extra={
                "thickness_m": torch.tensor(8e-6),
                "residual_solvent": torch.tensor(0.15),
            }
        )
        sim = get_simulator("diffcfd_litho", config)
        mask = torch.ones(32, 32) * 0.8
        result = sim.simulate(mask)
        assert result.backend == "diffcfd_litho"
        assert result.aerial.shape == (32, 32)
        assert result.metadata["model"] == "diffcfd_litho"
        assert "remaining_thickness_m" in result.metadata

    def test_remaining_thickness_positive(self):
        config = SimulatorConfig(
            extra={
                "thickness_m": torch.tensor(8e-6),
                "residual_solvent": torch.tensor(0.15),
            }
        )
        sim = get_simulator("diffcfd_litho", config)
        mask = torch.ones(16, 16) * 0.5
        result = sim.simulate(mask)
        assert result.metadata["remaining_thickness_m"] > 0

    def test_custom_dill_params(self):
        config = SimulatorConfig(
            extra={
                "dill_A": 0.6,
                "dill_B": 0.03,
                "thickness_m": torch.tensor(5e-6),
                "residual_solvent": torch.tensor(0.1),
            }
        )
        sim = get_simulator("diffcfd_litho", config)
        mask = torch.ones(16, 16) * 0.7
        result = sim.simulate(mask)
        assert result.backend == "diffcfd_litho"


class TestDiffCFDSpinCoatSimulator:
    def test_loads_via_registry(self):
        sim = get_simulator("diffcfd_spin_coat")
        assert sim.name == "diffcfd_spin_coat"
        assert sim.differentiable is True

    def test_simulate_returns_result(self):
        import math

        n_steps = 100
        omega = torch.full((n_steps,), 2500.0 * 2.0 * math.pi / 60.0)
        config = SimulatorConfig(extra={"omega_profile": omega, "spin_dt": 0.01})
        sim = get_simulator("diffcfd_spin_coat", config)
        mask = torch.ones(16, 16)
        result = sim.simulate(mask)
        assert result.backend == "diffcfd_spin_coat"
        assert "dry_thickness_m" in result.metadata
        assert "residual_solvent" in result.metadata

    def test_dry_thickness_positive(self):
        n_steps = 100
        omega = torch.full((n_steps,), 3000.0 * 2.0 * math.pi / 60.0)
        config = SimulatorConfig(extra={"omega_profile": omega, "spin_dt": 0.01})
        sim = get_simulator("diffcfd_spin_coat", config)
        result = sim.simulate(torch.ones(8, 8))
        assert result.metadata["dry_thickness_m"] > 0

    def test_higher_rpm_thins_faster(self):
        """Higher RPM causes faster thinning at intermediate time points.

        Both RPMs converge to the same final thickness (polymer mass
        conservation), but at any fixed intermediate time the higher RPM
        produces a thinner film.
        """
        import math

        from diffcfd.solvers.spin_coating import MeyerhoferSolver

        n_steps = 100
        dt = 0.001
        omega_low = torch.full((n_steps,), 2000.0 * 2.0 * math.pi / 60.0)
        omega_high = torch.full((n_steps,), 4000.0 * 2.0 * math.pi / 60.0)

        solver = MeyerhoferSolver()
        h_low, _ = solver(omega_low, dt, h0=8e-6, c0=0.85)
        h_high, _ = solver(omega_high, dt, h0=8e-6, c0=0.85)

        # At step 50, the high-RPM film should be thinner
        assert h_low[50].item() > h_high[50].item()
