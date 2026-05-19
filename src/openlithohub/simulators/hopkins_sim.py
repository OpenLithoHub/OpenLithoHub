"""Hopkins/SOCS simulator adapter — the bundled reference backend."""

from __future__ import annotations

import torch

from openlithohub._utils.hopkins import HopkinsParams, simulate_aerial_image_hopkins
from openlithohub.simulators.base import BaseSimulator, SimulatorConfig, SimulatorResult


class HopkinsSimulator(BaseSimulator):
    """Differentiable Hopkins/SOCS simulator.

    Wraps :func:`openlithohub._utils.hopkins.simulate_aerial_image_hopkins`.
    The full forward pass is auto-differentiable, so this adapter is the
    right choice when used as a training-loop oracle for ILT or AI-OPC.

    Backend-specific options read from ``config.extra``:

    * ``illumination`` (``circular`` | ``annular`` | ``dipole`` |
      ``quasar``) — source shape, default ``circular``.
    * ``num_kernels`` (int) — SOCS truncation order, default 24.
    * ``dipole_angle_deg``, ``pole_opening_deg`` — pole geometry for
      dipole/quasar.
    """

    name = "hopkins"
    differentiable = True

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        super().__init__(config)
        self._hparams = self._build_hparams(self.config)

    @staticmethod
    def _build_hparams(config: SimulatorConfig) -> HopkinsParams:
        extra = config.extra or {}
        return HopkinsParams(
            wavelength_nm=config.wavelength_nm,
            na=config.na,
            sigma=config.sigma,
            sigma_inner=config.sigma_inner,
            pixel_size_nm=config.pixel_size_nm,
            num_kernels=int(extra.get("num_kernels", 24)),
            illumination=extra.get("illumination", "circular"),
            dipole_angle_deg=float(extra.get("dipole_angle_deg", 0.0)),
            pole_opening_deg=float(extra.get("pole_opening_deg", 30.0)),
            defocus_nm=config.defocus_nm,
        )

    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        aerial = simulate_aerial_image_hopkins(
            mask,
            params=self._hparams,
            dose=self.config.dose,
        )
        threshold = self.config.threshold * self.config.dose
        resist = (aerial >= threshold).to(aerial.dtype)
        return SimulatorResult(
            aerial=aerial,
            resist=resist,
            backend=self.name,
            metadata={
                "illumination": self._hparams.illumination,
                "num_kernels": self._hparams.num_kernels,
                "differentiable": True,
            },
        )
