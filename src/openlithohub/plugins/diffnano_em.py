"""DiffNano EM solver adapters (RCWA / FDTD / FDFD).

Registers EM backends with the OpenLithoHub simulator registry so users
can do ``get_simulator("diffnano_rcwa", config)``.  Only importable when
the ``[diffnano]`` extra is installed.
"""

from __future__ import annotations

from typing import Any

import torch

from openlithohub.simulators.base import BaseSimulator, SimulatorConfig, SimulatorResult

__all__ = ["DiffNanoRCWA", "DiffNanoFDTD2D", "DiffNanoFDFD2D"]


def _require_diffnano() -> Any:
    from openlithohub.plugins import optional_import

    return optional_import("diffnano.solvers.rcwa", plugin="diffnano")


class DiffNanoRCWA(BaseSimulator):
    """RCWA (rigorous coupled-wave analysis) simulator backend via DiffNano.

    Provides high-accuracy periodic-structure EM simulation as an opt-in
    alternative to the built-in Hopkins transfer-matrix model.  Useful
    for 3D mask (M3D) effects where thin-mask approximation breaks down.
    """

    name = "diffnano_rcwa"
    differentiable = True

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        super().__init__(config)
        self._solver = None

    def prepare(self) -> None:
        mod = _require_diffnano()
        self._solver = mod.RCWASolver(
            n_orders=self.config.extra.get("rcwa_orders", 10),
        )

    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        if self._solver is None:
            self.prepare()

        geometry = mask.unsqueeze(0) if mask.ndim == 2 else mask
        result = self._solver.forward(geometry)

        aerial = result.field.squeeze(0)
        if aerial.ndim == 3:
            aerial = aerial[0]

        return SimulatorResult(
            aerial=aerial,
            resist=None,
            backend=self.name,
            metadata={"solver": "rcwa", "model": "diffnano"},
        )


class DiffNanoFDTD2D(BaseSimulator):
    """2D FDTD (finite-difference time-domain) simulator backend via DiffNano."""

    name = "diffnano_fdtd2d"
    differentiable = True

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        super().__init__(config)
        self._solver = None

    def prepare(self) -> None:
        from openlithohub.plugins import optional_import

        mod = optional_import("diffnano.solvers.fdtd2d", plugin="diffnano")
        self._solver = mod.FDTD2D()

    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        if self._solver is None:
            self.prepare()

        geometry = mask.unsqueeze(0) if mask.ndim == 2 else mask
        result = self._solver.forward(geometry)

        aerial = result.field.squeeze(0)
        if aerial.ndim == 3:
            aerial = aerial[0]

        return SimulatorResult(
            aerial=aerial,
            resist=None,
            backend=self.name,
            metadata={"solver": "fdtd2d", "model": "diffnano"},
        )


class DiffNanoFDFD2D(BaseSimulator):
    """2D FDFD (finite-difference frequency-domain) simulator backend via DiffNano."""

    name = "diffnano_fdfd2d"
    differentiable = True

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        super().__init__(config)
        self._solver = None

    def prepare(self) -> None:
        from openlithohub.plugins import optional_import

        mod = optional_import("diffnano.solvers.fdfd2d", plugin="diffnano")
        self._solver = mod.FDFD2D()

    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        if self._solver is None:
            self.prepare()

        geometry = mask.unsqueeze(0) if mask.ndim == 2 else mask
        result = self._solver.forward(geometry)

        aerial = result.field.squeeze(0)
        if aerial.ndim == 3:
            aerial = aerial[0]

        return SimulatorResult(
            aerial=aerial,
            resist=None,
            backend=self.name,
            metadata={"solver": "fdfd2d", "model": "diffnano"},
        )
