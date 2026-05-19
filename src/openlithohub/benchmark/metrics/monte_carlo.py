"""Monte Carlo stochastic-failure evaluation against a simulator backend.

Complements :func:`compute_stochastic_robustness` (which uses a fast
Gaussian-PSF model and Poisson photon noise) by letting callers run the
same Monte Carlo loop against any
:class:`openlithohub.simulators.BaseSimulator` — including the bundled
Hopkins/SOCS model or, with the appropriate adapter, a commercial
simulator.

This is the "give me a stochastic-failure number against my preferred
forward model" entry point that the v0.1 roadmap calls for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from openlithohub._utils.morphology import connected_components
from openlithohub._utils.tensor_ops import ensure_2d
from openlithohub.simulators import BaseSimulator


@dataclass
class MonteCarloFailureResult:
    """Result of a Monte Carlo stochastic-failure run."""

    bridge_probability: float
    break_probability: float
    failure_probability: float
    num_trials: int


def _count_components(binary: torch.Tensor) -> int:
    _, n = connected_components(binary, connectivity=8)
    return n


def monte_carlo_failure_probability(
    mask: torch.Tensor,
    simulator: BaseSimulator,
    num_trials: int = 50,
    dose_jitter_sigma: float = 0.02,
    threshold_jitter_sigma: float = 0.01,
    seed: int | None = None,
    perturb: Callable[[torch.Tensor, torch.Generator], torch.Tensor] | None = None,
) -> MonteCarloFailureResult:
    """Estimate stochastic-failure probability against a simulator backend.

    Runs ``num_trials`` independent simulations with small per-trial
    perturbations to dose and resist threshold (and, if provided, a
    user-supplied ``perturb`` operator on the mask itself). Counts how
    often the resulting resist contour acquires extra connected
    components ("breaks") or merges existing ones ("bridges") relative
    to the nominal run.

    Args:
        mask: ``(H, W)`` real-valued mask in ``[0, 1]``.
        simulator: Simulator backend. Must produce a ``resist`` field;
            backends that don't will be wrapped to threshold the aerial.
        num_trials: Number of perturbed simulations.
        dose_jitter_sigma: Std-dev of multiplicative dose jitter.
        threshold_jitter_sigma: Std-dev of additive resist-threshold
            jitter.
        seed: PRNG seed; ``None`` uses a fresh generator.
        perturb: Optional ``(mask, generator) -> mask`` callable for
            domain-specific perturbations (e.g. mask-write spot jitter).

    Returns:
        :class:`MonteCarloFailureResult`.
    """

    m = ensure_2d(mask).detach()
    nominal = simulator.simulate(m)
    nominal_resist = (
        nominal.resist
        if nominal.resist is not None
        else (nominal.aerial >= simulator.config.threshold * simulator.config.dose).to(
            nominal.aerial.dtype
        )
    )
    nominal_components = _count_components(nominal_resist)

    generator = torch.Generator(device=m.device)
    if seed is not None:
        generator.manual_seed(seed)

    nominal_dose = simulator.config.dose
    nominal_threshold = simulator.config.threshold
    bridge_count = 0
    break_count = 0

    try:
        for _trial in range(num_trials):
            dose_factor = 1.0 + dose_jitter_sigma * torch.randn(1, generator=generator).item()
            threshold_offset = threshold_jitter_sigma * torch.randn(1, generator=generator).item()
            simulator.config.__dict__["dose"] = nominal_dose * float(max(dose_factor, 1e-6))
            simulator.config.__dict__["threshold"] = float(
                max(nominal_threshold + threshold_offset, 1e-6)
            )

            trial_mask = perturb(m, generator) if perturb is not None else m
            result = simulator.simulate(trial_mask)
            resist = (
                result.resist
                if result.resist is not None
                else (result.aerial >= simulator.config.threshold * simulator.config.dose).to(
                    result.aerial.dtype
                )
            )
            trial_components = _count_components(resist)

            if trial_components < nominal_components:
                bridge_count += 1
            elif trial_components > nominal_components:
                break_count += 1
    finally:
        simulator.config.__dict__["dose"] = nominal_dose
        simulator.config.__dict__["threshold"] = nominal_threshold

    bridge_p = bridge_count / max(num_trials, 1)
    break_p = break_count / max(num_trials, 1)
    return MonteCarloFailureResult(
        bridge_probability=bridge_p,
        break_probability=break_p,
        failure_probability=bridge_p + break_p,
        num_trials=num_trials,
    )
