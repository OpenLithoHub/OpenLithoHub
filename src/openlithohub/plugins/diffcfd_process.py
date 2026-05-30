"""DiffCFD process simulation adapters (lithography + spin coating).

Wraps DiffCFD's ``LithoSolver``, ``MeyerhoferSolver``, and joint
optimization workflow as opt-in simulator backends.  Only importable
when the ``[diffcfd]`` extra is installed.
"""

from __future__ import annotations

import torch

from openlithohub.simulators.base import BaseSimulator, SimulatorConfig, SimulatorResult

__all__ = ["DiffCFDLithoSimulator", "DiffCFDSpinCoatSimulator"]


class DiffCFDLithoSimulator(BaseSimulator):
    """Dill-exposure + Mack-development lithography solver via DiffCFD.

    Models the full resist processing chain: Beer-Lambert attenuation with
    PAC kinetics (Dill) → solvent-dependent dissolution (Mack).  Accepts
    thickness and residual-solvent inputs from a preceding spin-coating step.
    """

    name = "diffcfd_litho"
    differentiable = True

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        super().__init__(config)
        self._solver = None

    def prepare(self) -> None:
        from openlithohub.plugins import optional_import

        mod = optional_import("diffcfd.solvers.litho", plugin="diffcfd")
        extra = self.config.extra
        self._solver = mod.LithoSolver(
            dill_A=extra.get("dill_A", 0.55),
            dill_B=extra.get("dill_B", 0.05),
            dill_C=extra.get("dill_C", 0.014),
            r_max=extra.get("r_max", 150.0),
            r_min=extra.get("r_min", 0.1),
            mack_n=extra.get("mack_n", 5.0),
            mack_a=extra.get("mack_a", 0.5),
            gamma_solvent=extra.get("gamma_solvent", 3.0),
        )

    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        if self._solver is None:
            self.prepare()

        extra = self.config.extra
        thickness = extra.get("thickness_m", torch.tensor(8e-6))
        residual_solvent = extra.get("residual_solvent", torch.tensor(0.15))
        dev_time = extra.get("dev_time_s", 30.0)

        if not isinstance(thickness, torch.Tensor):
            thickness = torch.as_tensor(thickness, dtype=torch.float32)
        if not isinstance(residual_solvent, torch.Tensor):
            residual_solvent = torch.as_tensor(residual_solvent, dtype=torch.float32)

        dose = mask.mean() * self.config.dose
        remaining = self._solver(
            thickness=thickness,
            residual_solvent=residual_solvent,
            exposure_dose=dose,
            dev_time=dev_time,
        )

        return SimulatorResult(
            aerial=mask,
            resist=(remaining.unsqueeze(0).unsqueeze(0).expand_as(mask) > 0).float(),
            backend=self.name,
            metadata={
                "model": "diffcfd_litho",
                "remaining_thickness_m": remaining.item(),
            },
        )


class DiffCFDSpinCoatSimulator(BaseSimulator):
    """Meyerhofer spin-coating solver via DiffCFD.

    Models centrifugal thinning and solvent evaporation to predict
    dry film thickness and residual solvent.  These outputs feed into
    the Dill/Mack lithography chain for joint process optimization.
    """

    name = "diffcfd_spin_coat"
    differentiable = True

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        super().__init__(config)
        self._solver = None

    def prepare(self) -> None:
        from openlithohub.plugins import optional_import

        mod = optional_import("diffcfd.solvers.spin_coating", plugin="diffcfd")
        extra = self.config.extra
        self._solver = mod.MeyerhoferSolver(
            rho=extra.get("rho", 1000.0),
            mu_solvent=extra.get("mu_solvent", 1e-3),
            alpha_visc=extra.get("alpha_visc", 4.5),
            beta_visc=extra.get("beta_visc", 1.5),
            c_evap=extra.get("c_evap", 1.2e-6),
            c_solid=extra.get("c_solid", 0.15),
        )

    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        if self._solver is None:
            self.prepare()

        extra = self.config.extra
        omega_profile = extra.get("omega_profile")
        if omega_profile is None:
            import math

            omega_profile = torch.full((10000,), 2500.0 * 2.0 * math.pi / 60.0)

        dt = extra.get("spin_dt", 0.001)
        h0 = extra.get("h0_m", 8e-6)
        c0 = extra.get("c0", 0.85)

        h_hist, c_hist = self._solver(omega_profile, dt, h0, c0)
        h_dry = h_hist[-1]
        c_dry = c_hist[-1]

        return SimulatorResult(
            aerial=mask,
            resist=None,
            backend=self.name,
            metadata={
                "model": "diffcfd_spin_coat",
                "dry_thickness_m": h_dry.item(),
                "residual_solvent": c_dry.item(),
            },
        )
