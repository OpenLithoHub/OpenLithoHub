"""Calibre nmOPC simulator adapter — stub.

Real Calibre integration requires the Mentor/Siemens EDA Calibre
toolchain on ``PATH`` plus a valid license file. We do not bundle
either, and we do not ship any code path that calls Calibre internals.

This stub exists so that:

1. Downstream code can target ``CalibreSimulator`` symbolically (e.g. in
   leaderboard config) without conditionally importing it.
2. The expected config schema is documented in one place.
3. A user with Calibre access can subclass and override
   :meth:`simulate` without rewriting the surrounding wiring.

Track the integration RFC at
``docs/rfcs/0003-commercial-simulator-hooks.md`` (TBD).
"""

from __future__ import annotations

import torch

from openlithohub.simulators.base import BaseSimulator, SimulatorConfig, SimulatorResult


class CalibreSimulator(BaseSimulator):
    """Stub adapter for Calibre nmOPC.

    ``config.extra`` should carry:

    * ``calibre_home`` (str) — install root containing ``bin/calibre``.
    * ``runset`` (str) — path to the SVRF runset to execute.
    * ``layer_map`` (dict[str, int]) — maps OpenLithoHub layer names to
      Calibre layer numbers.
    * ``license_server`` (str, optional) — for non-default flexlm setups.
    """

    name = "calibre"
    differentiable = False

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        super().__init__(config)
        self._validate_config()

    def _validate_config(self) -> None:
        extra = self.config.extra or {}
        for required in ("calibre_home", "runset"):
            if required not in extra:
                raise ValueError(
                    f"CalibreSimulator requires config.extra[{required!r}]; "
                    f"got keys={sorted(extra.keys())}"
                )

    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        del mask  # unused in the stub
        raise NotImplementedError(
            "CalibreSimulator is a configuration stub. Real Calibre "
            "integration requires the vendor toolchain and license; "
            "subclass this adapter and override simulate(). See "
            "docs/rfcs/0003-commercial-simulator-hooks.md for the "
            "integration plan."
        )
