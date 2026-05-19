"""Tachyon simulator adapter — stub.

Real Tachyon integration requires the ASML Brion Tachyon toolchain on
``PATH`` plus a license. We do not bundle either. See
:mod:`openlithohub.simulators.calibre` for the same policy and rationale.
"""

from __future__ import annotations

import torch

from openlithohub.simulators.base import BaseSimulator, SimulatorConfig, SimulatorResult


class TachyonSimulator(BaseSimulator):
    """Stub adapter for ASML Brion Tachyon.

    ``config.extra`` should carry:

    * ``tachyon_home`` (str) — install root.
    * ``recipe`` (str) — path to the ``.tcl`` recipe to execute.
    * ``layer_map`` (dict[str, int]) — OpenLithoHub layer → Tachyon layer.
    """

    name = "tachyon"
    differentiable = False

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        super().__init__(config)
        self._validate_config()

    def _validate_config(self) -> None:
        extra = self.config.extra or {}
        for required in ("tachyon_home", "recipe"):
            if required not in extra:
                raise ValueError(
                    f"TachyonSimulator requires config.extra[{required!r}]; "
                    f"got keys={sorted(extra.keys())}"
                )

    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        del mask
        raise NotImplementedError(
            "TachyonSimulator is a configuration stub. Real Tachyon "
            "integration requires the ASML/Brion toolchain and license; "
            "subclass this adapter and override simulate(). See "
            "docs/rfcs/0003-commercial-simulator-hooks.md for the "
            "integration plan."
        )
