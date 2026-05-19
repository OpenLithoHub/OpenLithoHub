"""Tests for the simulator backend interface."""

from __future__ import annotations

import pytest
import torch

from openlithohub.simulators import (
    BaseSimulator,
    CalibreSimulator,
    HopkinsSimulator,
    SimulatorConfig,
    SimulatorResult,
    TachyonSimulator,
    get_simulator,
    list_simulators,
    register_simulator,
)


def _make_mask() -> torch.Tensor:
    mask = torch.zeros(64, 64)
    mask[16:48, 16:48] = 1.0
    return mask


class TestHopkinsSimulator:
    def test_simulate_returns_aerial_and_resist(self) -> None:
        sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
        result = sim.simulate(_make_mask())
        assert isinstance(result, SimulatorResult)
        assert result.backend == "hopkins"
        assert result.aerial.shape == (64, 64)
        assert result.resist is not None
        assert result.resist.shape == (64, 64)
        assert torch.all((result.resist == 0) | (result.resist == 1))

    def test_simulate_propagates_gradients(self) -> None:
        mask = _make_mask().requires_grad_(True)
        sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
        result = sim.simulate(mask)
        result.aerial.sum().backward()
        assert mask.grad is not None
        assert torch.isfinite(mask.grad).all()

    def test_batched_input(self) -> None:
        mask = _make_mask().unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)
        sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
        result = sim.simulate(mask)
        assert result.aerial.shape == (2, 1, 64, 64)
        assert result.metadata["differentiable"] is True


class TestStubAdapters:
    def test_calibre_validates_required_extras(self) -> None:
        with pytest.raises(ValueError, match="calibre_home"):
            CalibreSimulator(SimulatorConfig())

    def test_calibre_simulate_raises_not_implemented(self) -> None:
        sim = CalibreSimulator(
            SimulatorConfig(extra={"calibre_home": "/opt/calibre", "runset": "x.svrf"})
        )
        with pytest.raises(NotImplementedError, match="vendor toolchain"):
            sim.simulate(_make_mask())

    def test_tachyon_validates_required_extras(self) -> None:
        with pytest.raises(ValueError, match="tachyon_home"):
            TachyonSimulator(SimulatorConfig())

    def test_tachyon_simulate_raises_not_implemented(self) -> None:
        sim = TachyonSimulator(
            SimulatorConfig(extra={"tachyon_home": "/opt/tachyon", "recipe": "x.tcl"})
        )
        with pytest.raises(NotImplementedError):
            sim.simulate(_make_mask())


class TestRegistry:
    def test_get_simulator_known(self) -> None:
        sim = get_simulator("hopkins")
        assert isinstance(sim, HopkinsSimulator)

    def test_get_simulator_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown simulator"):
            get_simulator("nonexistent")

    def test_list_simulators_includes_defaults(self) -> None:
        names = list_simulators()
        assert {"hopkins", "calibre", "tachyon"}.issubset(names)

    def test_register_custom_simulator(self) -> None:
        class FakeSim(BaseSimulator):
            name = "fake"

            def simulate(self, mask: torch.Tensor) -> SimulatorResult:
                return SimulatorResult(aerial=torch.zeros_like(mask), backend="fake")

        register_simulator("fake", FakeSim)
        try:
            sim = get_simulator("fake")
            assert isinstance(sim, FakeSim)
        finally:
            from openlithohub.simulators import registry as _registry

            _registry._REGISTRY.pop("fake", None)
