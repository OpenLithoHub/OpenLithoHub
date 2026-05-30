"""Hopkins/SOCS simulator adapter — the bundled reference backend."""

from __future__ import annotations

import torch

from openlithohub._utils.hopkins import HopkinsParams, simulate_aerial_image_hopkins
from openlithohub._utils.forward_model import _gaussian_diffuse
from openlithohub.simulators.base import BaseSimulator, SimulatorConfig, SimulatorResult


class HopkinsSimulator(BaseSimulator):
    """Differentiable Hopkins/SOCS simulator.

    Wraps :func:`openlithohub._utils.hopkins.simulate_aerial_image_hopkins`.
    The full forward pass is auto-differentiable, so this adapter is the
    right choice when used as a training-loop oracle for ILT or AI-OPC.

    Backend-specific options read from ``config.extra``:

    * ``illumination`` (``circular`` | ``annular`` | ``dipole`` |
      ``quasar``) — source shape, default ``circular``.
    * ``num_kernels`` (int) — SOCS truncation order, default 24
      (per Yang2023_LithoBench §3.2 / Table II — the reference SOCS
      decomposition for ICCAD16 evaluation).
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
        # Issue #52: do NOT scale the threshold by dose. `simulate_aerial_image_hopkins`
        # already multiplies the aerial by dose; if we also multiply the threshold by
        # dose, the comparison `aerial >= threshold * dose` reduces to
        # `aerial_unit >= threshold_unit` — i.e. dose is fully cancelled and the
        # resist contour is invariant under dose. The physical convention is the
        # opposite: changing dose changes the resist contour at a *fixed* clearing
        # threshold, since the resist's clearing intensity is a chemical
        # invariant. PW dose corners, stochastic / Monte-Carlo dose jitter, and
        # PVB dose-axis variation all relied on this.
        threshold = self.config.threshold
        # Opt-in acid diffusion: when resist_diffusion_nm > 0 or quencher > 0,
        # blur the aerial image (proportional to photoacid concentration) and
        # subtract quencher before binarization. Both default to 0.0, which
        # skips this block entirely for bit-identical legacy behavior.
        if self.config.resist_diffusion_nm > 0.0 or self.config.quencher > 0.0:
            sigma_px = self.config.resist_diffusion_nm / max(self.config.pixel_size_nm, 1e-6)
            if sigma_px > 0.1:
                aerial = _gaussian_diffuse(aerial, sigma_px)
            aerial = (aerial - self.config.quencher).clamp(min=0.0)
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
