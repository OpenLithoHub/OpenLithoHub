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

    def with_config(self, config: SimulatorConfig) -> HopkinsSimulator:
        """Clone, reusing cached SOCS kernels when only dose/threshold changed.

        SOCS kernel construction depends on optical fields
        (wavelength/NA/sigma/illumination/defocus/pixel_size_nm/num_kernels).
        When those are unchanged, we can hand the new sibling our pre-built
        :class:`HopkinsParams` instead of recomputing.
        """
        sibling = type(self).__new__(type(self))
        BaseSimulator.__init__(sibling, config)
        if self._hparams_match(config):
            sibling._hparams = self._hparams
        else:
            sibling._hparams = self._build_hparams(config)
        return sibling

    def _hparams_match(self, other: SimulatorConfig) -> bool:
        a, b = self.config, other
        if (
            a.wavelength_nm != b.wavelength_nm
            or a.na != b.na
            or a.sigma != b.sigma
            or a.sigma_inner != b.sigma_inner
            or a.pixel_size_nm != b.pixel_size_nm
            or a.defocus_nm != b.defocus_nm
        ):
            return False
        ax = a.extra or {}
        bx = b.extra or {}
        keys = ("num_kernels", "illumination", "dipole_angle_deg", "pole_opening_deg")
        return all(ax.get(k) == bx.get(k) for k in keys)

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
