"""Tests for DiffNano EM solver adapters (P2).

DiffNano is NOT installed in the core test env, so these verify error
handling.  When DiffNano becomes available, the installed tests exercise
real EM simulation.
"""

from __future__ import annotations

import pytest
import torch

from openlithohub.plugins import OptionalPluginError
from openlithohub.simulators.registry import get_simulator

DIFFNANO_AVAILABLE = True
try:
    import diffnano  # noqa: F401
except ImportError:
    DIFFNANO_AVAILABLE = False


@pytest.mark.skipif(DIFFNANO_AVAILABLE, reason="DiffNano is installed — run full test")
class TestDiffNanoEMNotInstalled:
    @pytest.mark.parametrize("backend", ["diffnano_rcwa", "diffnano_fdtd2d", "diffnano_fdfd2d"])
    def test_em_backend_raises_helpful_error(self, backend):
        with pytest.raises(KeyError, match=r"pip install openlithohub\[diffnano\]"):
            get_simulator(backend)

    def test_rcwa_adapter_raises_on_prepare(self):
        from openlithohub.plugins.diffnano_em import DiffNanoRCWA

        sim = DiffNanoRCWA()
        with pytest.raises(OptionalPluginError, match="diffnano"):
            sim.prepare()

    def test_fdtd_adapter_raises_on_prepare(self):
        from openlithohub.plugins.diffnano_em import DiffNanoFDTD2D

        sim = DiffNanoFDTD2D()
        with pytest.raises(OptionalPluginError, match="diffnano"):
            sim.prepare()

    def test_fdfd_adapter_raises_on_prepare(self):
        from openlithohub.plugins.diffnano_em import DiffNanoFDFD2D

        sim = DiffNanoFDFD2D()
        with pytest.raises(OptionalPluginError, match="diffnano"):
            sim.prepare()


@pytest.mark.skipif(not DIFFNANO_AVAILABLE, reason="DiffNano not installed")
class TestDiffNanoEMInstalled:
    def test_rcwa_simulate(self):
        sim = get_simulator("diffnano_rcwa")
        assert sim.name == "diffnano_rcwa"
        mask = torch.rand(16, 16)
        result = sim.simulate(mask)
        assert result.aerial.shape == (16, 16)

    def test_fdtd2d_simulate(self):
        sim = get_simulator("diffnano_fdtd2d")
        assert sim.name == "diffnano_fdtd2d"
        mask = torch.rand(16, 16)
        result = sim.simulate(mask)
        assert result.aerial.shape == (16, 16)

    def test_fdfd2d_simulate(self):
        sim = get_simulator("diffnano_fdfd2d")
        assert sim.name == "diffnano_fdfd2d"
        mask = torch.rand(16, 16)
        result = sim.simulate(mask)
        assert result.aerial.shape == (16, 16)
