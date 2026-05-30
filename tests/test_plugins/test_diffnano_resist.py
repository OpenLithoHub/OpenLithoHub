"""Tests for DiffNano resist adapter (P1).

DiffNano is NOT installed in the core-only test environment, so these
tests verify the adapter's error handling and, when DiffNano becomes
available, the real integration.
"""

from __future__ import annotations

import pytest
import torch

from openlithohub.plugins import OptionalPluginError

# DiffNano is not installed in the core test env — all tests must handle that.
DIFFNANO_AVAILABLE = True
try:
    import diffnano  # noqa: F401
except ImportError:
    DIFFNANO_AVAILABLE = False


@pytest.mark.skipif(DIFFNANO_AVAILABLE, reason="DiffNano is installed — run full test")
class TestDiffNanoResistNotInstalled:
    def test_adapter_raises_optional_plugin_error(self):
        from openlithohub.plugins.diffnano_resist import DiffNanoResistAdapter

        with pytest.raises(OptionalPluginError, match="diffnano"):
            DiffNanoResistAdapter()

    def test_hopkins_with_diffnano_backend_raises(self):
        from openlithohub.simulators.base import SimulatorConfig
        from openlithohub.simulators.hopkins_sim import HopkinsSimulator

        config = SimulatorConfig(resist_backend="diffnano")
        sim = HopkinsSimulator(config)
        mask = torch.ones(16, 16) * 0.5
        with pytest.raises(OptionalPluginError, match="diffnano"):
            sim.simulate(mask)


@pytest.mark.skipif(not DIFFNANO_AVAILABLE, reason="DiffNano not installed")
class TestDiffNanoResistInstalled:
    def test_adapter_forward(self):
        from openlithohub.plugins.diffnano_resist import DiffNanoResistAdapter

        adapter = DiffNanoResistAdapter(pixel_size_nm=1.0)
        aerial = torch.rand(32, 32)
        result = adapter(aerial)
        assert result.shape == (32, 32)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_adapter_calibrate(self):
        from openlithohub.plugins.diffnano_resist import DiffNanoResistAdapter

        adapter = DiffNanoResistAdapter(pixel_size_nm=1.0)
        dose = torch.rand(16, 16)
        target = (dose > 0.5).float()
        losses = adapter.calibrate([(dose, target)], n_steps=5, lr=0.01)
        assert len(losses) == 5
        assert losses[-1] <= losses[0]
