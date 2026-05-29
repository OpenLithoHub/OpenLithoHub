"""Hybrid Z-score convergence monitoring for lithography optimisation.

.. deprecated::
   Prefer ``diff_surrogate.convergence`` — this local copy will be removed in a
   future version.  The symbols below are re-exported from the shared package
   when ``diff-surrogate`` is installed; otherwise the local implementation is
   used as a fallback so existing code continues to work without the dependency.
"""

from __future__ import annotations

try:
    from diff_surrogate.convergence import (
        ConvergenceAction,
        ConvergenceConfig,
        ConvergenceMonitor,
        hybrid_z_score,
    )
except ImportError:
    # Fallback: keep local implementation if diff-surrogate not installed
    from dataclasses import dataclass, field
    from enum import Enum
    from typing import Literal

    import torch

    class ConvergenceAction(str, Enum):
        """Suggested action from the convergence monitor."""
        CONTINUE = "continue"
        EARLY_STOP = "early_stop"
        REDUCE_LR = "reduce_lr"

    @dataclass
    class ConvergenceVerdict:
        """Result from a convergence check.

        Attributes
        ----------
        action : ConvergenceAction
            Suggested optimisation action.
        z_hybrid : float
            Current hybrid Z-score of recent improvements.
        z_standard : float
            Standard Z-score component.
        z_robust : float
            Robust (MAD-based) Z-score component.
        reason : str
            Human-readable explanation of the verdict.
        """

        action: ConvergenceAction
        z_hybrid: float
        z_standard: float
        z_robust: float
        reason: str

    class HybridZScore:
        """Compute a hybrid Z-score blending standard and robust (MAD-based) variants.

        Parameters
        ----------
        weight : float
            Blending weight ``w`` in [0, 1].  ``Z_hybrid = Z_std * (1-w) + Z_rob * w``.
            Default 0.5 gives equal weight to both.
        """

        def __init__(self, weight: float = 0.5) -> None:
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"weight must be in [0, 1], got {weight}")
            self.weight = weight

        def compute(self, values: list[float] | torch.Tensor) -> float:
            if isinstance(values, torch.Tensor):
                vals = values.detach().cpu().float()
            else:
                vals = torch.tensor(values, dtype=torch.float32)

            if vals.numel() < 2:
                return 0.0

            history = vals[:-1]
            current = vals[-1]

            mean = history.mean()
            std = history.std()
            z_std = ((current - mean) / (std + 1e-12)).item()

            median = history.median()
            mad = (history - median).abs().median()
            robust_scale = 1.4826 * mad + 1e-12
            z_rob = ((current - median) / robust_scale).item()

            z_hybrid = z_std * (1.0 - self.weight) + z_rob * self.weight
            return z_hybrid

    class ConvergenceMonitor:  # type: ignore[no-redef]
        """Track optimisation loss history and recommend actions.

        Uses :class:`HybridZScore` to assess whether recent improvements are
        statistically significant relative to the optimisation trajectory.

        Parameters
        ----------
        window : int
            Number of recent loss values to consider for the Z-score computation.
        z_stop_threshold : float
            If the hybrid Z-score of the loss improvement is above this value
            (i.e. the loss is NOT dropping significantly), trigger early stop.
        z_reduce_lr_threshold : float
            Threshold below ``z_stop_threshold``: if Z-score is above this but
            below ``z_stop_threshold``, suggest reducing the learning rate.
        weight : float
            Blending weight for the hybrid Z-score (see :class:`HybridZScore`).
        patience : int
            Number of consecutive "no improvement" checks before triggering
            early stop.  Prevents premature termination from single-step noise.
        """

        def __init__(
            self,
            window: int = 20,
            z_stop_threshold: float = -0.5,
            z_reduce_lr_threshold: float = -1.5,
            weight: float = 0.5,
            patience: int = 5,
        ) -> None:
            self.window = window
            self.z_stop_threshold = z_stop_threshold
            self.z_reduce_lr_threshold = z_reduce_lr_threshold
            self.patience = patience
            self._z_scorer = HybridZScore(weight=weight)

            self._losses: list[float] = []
            self._no_improve_count: int = 0

        def reset(self) -> None:
            self._losses.clear()
            self._no_improve_count = 0

        def update(self, loss: float) -> ConvergenceVerdict:
            self._losses.append(loss)

            if len(self._losses) < 3:
                return ConvergenceVerdict(
                    action=ConvergenceAction.CONTINUE,
                    z_hybrid=0.0,
                    z_standard=0.0,
                    z_robust=0.0,
                    reason="Insufficient data points for convergence assessment.",
                )

            window_losses = self._losses[-self.window:]
            z_hybrid = self._z_scorer.compute(window_losses)

            vals = torch.tensor(window_losses, dtype=torch.float32)
            history = vals[:-1]
            current = vals[-1]

            mean = history.mean()
            std = history.std()
            z_std = ((current - mean) / (std + 1e-12)).item()

            median = history.median()
            mad = (history - median).abs().median()
            z_rob = ((current - median) / (1.4826 * mad + 1e-12)).item()

            if z_hybrid > self.z_stop_threshold:
                self._no_improve_count += 1
            else:
                self._no_improve_count = 0

            if self._no_improve_count >= self.patience:
                return ConvergenceVerdict(
                    action=ConvergenceAction.EARLY_STOP,
                    z_hybrid=z_hybrid,
                    z_standard=z_std,
                    z_robust=z_rob,
                    reason=(
                        f"No significant improvement for {self._no_improve_count} "
                        f"consecutive checks (z_hybrid={z_hybrid:.3f} > "
                        f"{self.z_stop_threshold:.3f})."
                    ),
                )

            if z_hybrid > self.z_reduce_lr_threshold:
                return ConvergenceVerdict(
                    action=ConvergenceAction.REDUCE_LR,
                    z_hybrid=z_hybrid,
                    z_standard=z_std,
                    z_robust=z_rob,
                    reason=(
                        f"Improvement slowing (z_hybrid={z_hybrid:.3f}, "
                        f"threshold={self.z_reduce_lr_threshold:.3f}). "
                        f"Consider reducing learning rate."
                    ),
                )

            return ConvergenceVerdict(
                action=ConvergenceAction.CONTINUE,
                z_hybrid=z_hybrid,
                z_standard=z_std,
                z_robust=z_rob,
                reason=(
                    f"Good progress (z_hybrid={z_hybrid:.3f}). "
                    f"Loss improvement is statistically significant."
                ),
            )

        @property
        def losses(self) -> list[float]:
            return list(self._losses)

        @property
        def best_loss(self) -> float | None:
            return min(self._losses) if self._losses else None

    def hybrid_z_score(values, weight: float = 0.5) -> float:  # type: ignore[no-redef]
        """Compute hybrid Z-score — convenience wrapper around HybridZScore."""
        return HybridZScore(weight=weight).compute(values)

    # ConvergenceConfig is not present in the local fallback; provide a stub
    @dataclass
    class ConvergenceConfig:  # type: ignore[no-redef]
        window: int = 20
        hybrid_weight: float = 0.5
        early_stop_threshold: float = 0.05
        reduce_lr_threshold: float = 0.2
        min_steps: int = 10
        patience: int = 5
