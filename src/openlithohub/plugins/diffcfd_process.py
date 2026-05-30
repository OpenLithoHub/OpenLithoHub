"""DiffCFD process simulation adapters (lithography + spin coating).

Wraps DiffCFD's ``LithoSolver``, ``MeyerhoferSolver``, and joint
optimization workflow as opt-in simulator backends.  Only importable
when the ``[diffcfd]`` extra is installed.
"""

from __future__ import annotations

import torch

from openlithohub._constants import (
    DIFFCFD_LITHO_DEFAULTS,
    DIFFCFD_PROCESS_DEFAULTS,
    DIFFCFD_SPIN_COAT_DEFAULTS,
)
from openlithohub.simulators.base import BaseSimulator, SimulatorConfig, SimulatorResult

__all__ = ["DiffCFDLithoSimulator", "DiffCFDSpinCoatSimulator"]

# Re-export dicts for backward compat with tests that import these names
LITHO_DEFAULTS = DIFFCFD_LITHO_DEFAULTS
SPIN_COAT_DEFAULTS = DIFFCFD_SPIN_COAT_DEFAULTS
PROCESS_DEFAULTS = DIFFCFD_PROCESS_DEFAULTS


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
            dill_A=extra.get("dill_A", LITHO_DEFAULTS["dill_A"]),
            dill_B=extra.get("dill_B", LITHO_DEFAULTS["dill_B"]),
            dill_C=extra.get("dill_C", LITHO_DEFAULTS["dill_C"]),
            r_max=extra.get("r_max", LITHO_DEFAULTS["r_max"]),
            r_min=extra.get("r_min", LITHO_DEFAULTS["r_min"]),
            mack_n=extra.get("mack_n", LITHO_DEFAULTS["mack_n"]),
            mack_a=extra.get("mack_a", LITHO_DEFAULTS["mack_a"]),
            gamma_solvent=extra.get("gamma_solvent", LITHO_DEFAULTS["gamma_solvent"]),
        )

    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        if self._solver is None:
            self.prepare()
        assert self._solver is not None

        extra = self.config.extra
        thickness = extra.get("thickness_m", torch.tensor(PROCESS_DEFAULTS["thickness_m"]))
        residual_solvent = extra.get(
            "residual_solvent", torch.tensor(PROCESS_DEFAULTS["residual_solvent"])
        )
        dev_time = extra.get("dev_time_s", PROCESS_DEFAULTS["dev_time_s"])

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
            rho=extra.get("rho", SPIN_COAT_DEFAULTS["rho"]),
            mu_solvent=extra.get("mu_solvent", SPIN_COAT_DEFAULTS["mu_solvent"]),
            alpha_visc=extra.get("alpha_visc", SPIN_COAT_DEFAULTS["alpha_visc"]),
            beta_visc=extra.get("beta_visc", SPIN_COAT_DEFAULTS["beta_visc"]),
            c_evap=extra.get("c_evap", SPIN_COAT_DEFAULTS["c_evap"]),
            c_solid=extra.get("c_solid", SPIN_COAT_DEFAULTS["c_solid"]),
        )

    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        if self._solver is None:
            self.prepare()
        assert self._solver is not None

        extra = self.config.extra
        omega_profile = extra.get("omega_profile")
        if omega_profile is None:
            import math

            omega_profile = torch.full(
                (10000,),
                PROCESS_DEFAULTS["omega_rpm"] * 2.0 * math.pi / 60.0,
            )

        dt = extra.get("spin_dt", PROCESS_DEFAULTS["spin_dt"])
        h0 = extra.get("h0_m", PROCESS_DEFAULTS["h0_m"])
        c0 = extra.get("c0", PROCESS_DEFAULTS["c0"])

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
