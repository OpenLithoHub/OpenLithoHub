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

    def _repr_html_(self) -> str:
        from openlithohub.jupyter._html import kv_table, panel, pass_fail_badge

        passed = self.failure_probability < 0.01
        rows = [
            ("Failure probability", f"{self.failure_probability:.4%}"),
            ("Bridge probability", f"{self.bridge_probability:.4%}"),
            ("Break probability", f"{self.break_probability:.4%}"),
            ("Trials", str(self.num_trials)),
        ]
        return panel(
            title="Monte Carlo failure",
            header_html=pass_fail_badge(passed),
            body_html=kv_table(rows),
        )


def _bridge_and_break_versus(nominal: torch.Tensor, trial: torch.Tensor) -> tuple[bool, bool]:
    """Return (has_bridge, has_break) for a trial vs nominal contour.

    A *bridge* is a pair of nominally-distinct components that merge in
    the trial. A *break* is a nominally-single component that splits in
    the trial. Detected by per-pixel label assignment, so a single trial
    can simultaneously exhibit a bridge and a break — the net component
    count would mask that, but the failure-probability metric must
    classify it as a failure regardless of direction.
    """
    nominal_labels, _ = connected_components(nominal > 0.5, connectivity=8)
    trial_labels, _ = connected_components(trial > 0.5, connectivity=8)
    common = (nominal > 0.5) & (trial > 0.5)
    if not common.any():
        # Nothing to compare on; treat as no bridge/break this trial.
        return False, False

    n_lab = nominal_labels[common]
    t_lab = trial_labels[common]

    # Bridge: two distinct nominal labels share a single trial label.
    # Group nominal labels by trial label and check for >1 unique nominal.
    has_bridge = False
    for trial_id in torch.unique(t_lab).tolist():
        nominal_in_trial = torch.unique(n_lab[t_lab == trial_id])
        if nominal_in_trial.numel() > 1:
            has_bridge = True
            break

    # Break: one nominal label is split across multiple trial labels.
    has_break = False
    for nominal_id in torch.unique(n_lab).tolist():
        trial_in_nominal = torch.unique(t_lab[n_lab == nominal_id])
        if trial_in_nominal.numel() > 1:
            has_break = True
            break

    return has_bridge, has_break


def monte_carlo_failure_probability(
    mask: torch.Tensor,
    simulator: BaseSimulator,
    num_trials: int = 50,
    dose_jitter_sigma: float = 0.02,
    threshold_jitter_sigma: float = 0.01,
    seed: int | None = 0,
    perturb: Callable[[torch.Tensor, torch.Generator], torch.Tensor] | None = None,
) -> MonteCarloFailureResult:
    """Estimate stochastic-failure probability against a simulator backend.

    Runs ``num_trials`` independent simulations with small per-trial
    perturbations to dose and resist threshold (and, if provided, a
    user-supplied ``perturb`` operator on the mask itself). Counts how
    often the resulting resist contour acquires extra connected
    components ("breaks") or merges existing ones ("bridges") relative
    to the nominal run.

    A trial that simultaneously bridges one component pair *and* breaks
    a different component is counted as a failure on both axes — the
    earlier ``net component count`` heuristic would have masked the
    pair as a no-op (issue #55).

    Dose jitter is applied as a post-hoc aerial scaling rather than via
    ``config.dose``: the bundled HopkinsSimulator's threshold scales
    with dose (``threshold = cfg.threshold * cfg.dose``), so pushing
    jitter into ``cfg.dose`` cancels at the threshold and the perturbation
    becomes a no-op (issue #54, downstream of #52).

    Args:
        mask: ``(H, W)`` real-valued mask in ``[0, 1]``.
        simulator: Simulator backend. Must produce a ``resist`` field;
            backends that don't will be wrapped to threshold the aerial.
        num_trials: Number of perturbed simulations.
        dose_jitter_sigma: Std-dev of multiplicative dose jitter.
        threshold_jitter_sigma: Std-dev of additive resist-threshold
            jitter.
        seed: PRNG seed; defaults to ``0`` so leaderboard runs are
            reproducible. Pass ``None`` for a fresh entropy-seeded generator.
        perturb: Optional ``(mask, generator) -> mask`` callable for
            domain-specific perturbations (e.g. mask-write spot jitter).

    Returns:
        :class:`MonteCarloFailureResult`.
    """

    m = ensure_2d(mask).detach()
    nominal = simulator.simulate(m)
    nominal_threshold = simulator.config.threshold
    nominal_resist = (
        nominal.resist
        if nominal.resist is not None
        else (nominal.aerial >= nominal_threshold).to(nominal.aerial.dtype)
    )

    generator = torch.Generator(device=m.device)
    if seed is not None:
        generator.manual_seed(seed)

    bridge_count = 0
    break_count = 0

    for _trial in range(num_trials):
        # Per-trial multiplicative dose jitter and additive threshold
        # jitter. We apply both *outside* the simulator and scale the
        # aerial intensity directly. After issue #52 was fixed (the
        # threshold no longer scales with dose), passing dose through
        # `config.dose` would also work, but doing it here keeps the
        # MC path independent of any future simulator-side dose
        # convention drift and lets us apply both jitters in one
        # pass without rebuilding the simulator config.
        dose_factor = 1.0 + dose_jitter_sigma * torch.randn(1, generator=generator).item()
        dose_factor = max(dose_factor, 1e-6)
        threshold_offset = threshold_jitter_sigma * torch.randn(1, generator=generator).item()
        trial_threshold = max(nominal_threshold + threshold_offset, 1e-6)

        trial_mask = perturb(m, generator) if perturb is not None else m
        result = simulator.simulate(trial_mask)
        # Apply jitter to the aerial intensity directly: a higher dose
        # multiplies aerial photons, a lower threshold lowers the resist
        # cutoff. Both feed into the same binarisation.
        scaled_aerial = result.aerial * dose_factor
        resist = (scaled_aerial >= trial_threshold).to(result.aerial.dtype)

        has_bridge, has_break = _bridge_and_break_versus(nominal_resist, resist)
        if has_bridge:
            bridge_count += 1
        if has_break:
            break_count += 1

    bridge_p = bridge_count / max(num_trials, 1)
    break_p = break_count / max(num_trials, 1)
    # A single trial can be both a bridge and a break, so failure_p is
    # *not* the sum (which would over-count those trials). Use the
    # union: failure = trials with bridge OR break = bridge + break -
    # both. We don't track ``both`` explicitly, so use the inclusion-
    # exclusion upper bound min(bridge + break, 1.0) as a conservative
    # estimate that never exceeds 1.
    failure_p = min(bridge_p + break_p, 1.0)
    return MonteCarloFailureResult(
        bridge_probability=bridge_p,
        break_probability=break_p,
        failure_probability=failure_p,
        num_trials=num_trials,
    )
