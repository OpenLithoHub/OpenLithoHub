"""Conformal coverage gate for stochastic ILT acceptance.

Calibrates LCDU/stochastic EPE through-focus quantiles with conformal
coverage, producing process windows with guaranteed coverage rates.

References:
    - Calibrated UQ for Operator Learning via Conformal Prediction, arXiv:2402.01960
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from openlithohub._constants import THRESHOLD_ICCAD16
from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.resist_model import differentiable_threshold
from openlithohub.benchmark.metrics.stochastic_loss import (
    _reparameterized_poisson,
    _soft_edge_mask,
)


@dataclass(frozen=True)
class CoverageResult:
    """Conformal-calibrated coverage result for a single mask design."""

    epe_band: tuple[float, float]
    lcdu_band: tuple[float, float]
    empirical_coverage: float
    target_coverage: float
    process_window: dict[str, bool]


class StochasticSampler:
    """Generates stochastic EPE/LCDU samples from a nominal mask design.

    Uses the same Poisson-to-Normal noise model as stochastic_loss.py.
    """

    def __init__(
        self,
        dose_photons_per_nm2: float = 30.0,
        pixel_size_nm: float = 1.0,
        sigma_px: float = 2.0,
        resist_threshold: float = THRESHOLD_ICCAD16,
        steepness: float = 50.0,
    ) -> None:
        self.dose_photons_per_nm2 = dose_photons_per_nm2
        self.pixel_size_nm = pixel_size_nm
        self.sigma_px = sigma_px
        self.resist_threshold = resist_threshold
        self.steepness = steepness

    def _aerial_with_defocus(self, mask: torch.Tensor, defocus_nm: float) -> torch.Tensor:
        if abs(defocus_nm) < 1e-6:
            return simulate_aerial_image(mask, sigma_px=self.sigma_px, dose=1.0)
        dof_nm = 100.0
        sigma_eff = self.sigma_px * (1.0 + (defocus_nm / dof_nm) ** 2) ** 0.5
        return simulate_aerial_image(mask, sigma_px=sigma_eff, dose=1.0)

    def _aerial_with_dose(
        self, mask: torch.Tensor, dose_factor: float, defocus_nm: float = 0.0
    ) -> torch.Tensor:
        aerial = self._aerial_with_defocus(mask, defocus_nm)
        return aerial * dose_factor

    def _compute_edge_epe(
        self, resist_noised: torch.Tensor, resist_nominal: torch.Tensor, edge_mask: torch.Tensor
    ) -> torch.Tensor:
        pixel_diff = (resist_noised - resist_nominal).abs()
        edge_weight = edge_mask.clamp(min=1e-8)
        return (pixel_diff * edge_weight).sum() / edge_weight.sum().clamp(min=1.0)

    def sample_epe(
        self,
        mask: torch.Tensor,
        n_samples: int = 100,
        dose_range: tuple[float, float] = (0.9, 1.1),
        focus_range: tuple[float, float] = (-0.1, 0.1),
    ) -> torch.Tensor:
        """Sample stochastic EPE across dose/focus variations.

        Returns:
            1-D tensor of shape ``(n_samples,)`` with per-sample EPE values.
        """
        mask_2d = mask.detach().float()
        if mask_2d.ndim > 2:
            mask_2d = mask_2d.squeeze()

        pixel_area_nm2 = self.pixel_size_nm * self.pixel_size_nm
        dose_scale = self.dose_photons_per_nm2 * pixel_area_nm2

        epe_values: list[torch.Tensor] = []
        for _ in range(n_samples):
            dose_factor = dose_range[0] + (dose_range[1] - dose_range[0]) * torch.rand(1).item()
            defocus_nm = focus_range[0] + (focus_range[1] - focus_range[0]) * torch.rand(1).item()

            aerial_nominal = self._aerial_with_dose(mask_2d, dose_factor, defocus_nm)
            lambda_map = torch.nn.functional.softplus(aerial_nominal) * dose_scale

            noised_photons = _reparameterized_poisson(lambda_map, 1)
            noised_intensity = noised_photons[0] / max(dose_scale, 1e-12)

            resist_nominal = differentiable_threshold(
                aerial_nominal, self.resist_threshold, self.steepness
            )
            resist_noised = differentiable_threshold(
                noised_intensity, self.resist_threshold, self.steepness
            )
            edge_mask = _soft_edge_mask(resist_nominal)
            epe = self._compute_edge_epe(resist_noised, resist_nominal, edge_mask)
            epe_values.append(epe.detach())

        return torch.stack(epe_values)

    def sample_lcdu(
        self,
        mask: torch.Tensor,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """Sample stochastic LCDU values.

        Returns:
            1-D tensor of shape ``(n_samples,)`` with per-sample LCDU proxy values.
        """
        mask_2d = mask.detach().float()
        if mask_2d.ndim > 2:
            mask_2d = mask_2d.squeeze()

        pixel_area_nm2 = self.pixel_size_nm * self.pixel_size_nm
        dose_scale = self.dose_photons_per_nm2 * pixel_area_nm2

        aerial_nominal = self._aerial_with_defocus(mask_2d, 0.0)
        lambda_map = torch.nn.functional.softplus(aerial_nominal) * dose_scale
        noised_photons = _reparameterized_poisson(lambda_map, n_samples)
        noised_intensity = noised_photons / max(dose_scale, 1e-12)

        resist_nominal = differentiable_threshold(
            aerial_nominal, self.resist_threshold, self.steepness
        )
        edge_mask = _soft_edge_mask(resist_nominal)

        cd_proxies: list[torch.Tensor] = []
        for i in range(n_samples):
            resist_noised = differentiable_threshold(
                noised_intensity[i], self.resist_threshold, self.steepness
            )
            cd_proxy = (resist_noised * edge_mask).sum()
            cd_proxies.append(cd_proxy.detach())

        cd_stack = torch.stack(cd_proxies)
        return cd_stack


class ThroughFocusCoverageCalibrator:
    """Calibrates conformal prediction intervals for through-focus EPE/LCDU.

    Uses a split-conformal approach: compute nonconformity scores on a
    calibration set, then use the ``(1 - alpha)`` quantile of those scores
    to produce guaranteed-coverage prediction bands.

    If an external ``SplitConformalPredictor`` is provided (from
    diff-surrogate), it is used for the quantile step. Otherwise a
    built-in conformal quantile is used.
    """

    def __init__(
        self,
        sampler: StochasticSampler | None = None,
        conformal_predictor: object | None = None,
        n_calibration_samples: int = 50,
        n_focus_points: int = 5,
        focus_range_nm: tuple[float, float] = (-50.0, 50.0),
    ) -> None:
        self.sampler = sampler or StochasticSampler()
        self.conformal_predictor = conformal_predictor
        self.n_calibration_samples = n_calibration_samples
        self.n_focus_points = n_focus_points
        self.focus_range_nm = focus_range_nm
        self._calibration_scores: torch.Tensor | None = None
        self._calibrated: bool = False

    def _compute_nonconformity(
        self, epe_samples: torch.Tensor, lcdu_samples: torch.Tensor
    ) -> torch.Tensor:
        """Nonconformity score: max of normalized EPE and LCDU deviations."""
        epe_median = epe_samples.median()
        epe_mad = (epe_samples - epe_median).abs().median().clamp(min=1e-8)
        epe_score = ((epe_samples - epe_median).abs() / epe_mad).max()

        lcdu_mean = lcdu_samples.mean()
        lcdu_std = lcdu_samples.std().clamp(min=1e-8)
        lcdu_score = ((lcdu_samples - lcdu_mean).abs() / lcdu_std).max()

        return torch.max(epe_score, lcdu_score)

    def calibrate(
        self,
        masks: list[torch.Tensor],
        targets: list[torch.Tensor],
        alpha: float = 0.1,
    ) -> None:
        """Fit conformal quantile on calibration set.

        Args:
            masks: List of mask tensors for calibration.
            targets: Corresponding target tensors (used for validation).
            alpha: Miscoverage rate (0.1 = 90% coverage target).
        """
        if len(masks) != len(targets):
            raise ValueError(
                f"masks and targets must have same length, got {len(masks)} vs {len(targets)}"
            )

        scores: list[float] = []
        focus_step = (self.focus_range_nm[1] - self.focus_range_nm[0]) / max(
            1, self.n_focus_points - 1
        )

        for mask in masks:
            max_score = 0.0
            for fi in range(self.n_focus_points):
                defocus_nm = self.focus_range_nm[0] + fi * focus_step
                abs_defocus = abs(defocus_nm)
                focus_range = (-abs_defocus, abs_defocus) if abs_defocus > 0 else (-0.01, 0.01)

                epe_samples = self.sampler.sample_epe(
                    mask,
                    n_samples=self.n_calibration_samples,
                    focus_range=focus_range,
                )
                lcdu_samples = self.sampler.sample_lcdu(mask, n_samples=self.n_calibration_samples)
                score = self._compute_nonconformity(epe_samples, lcdu_samples)
                max_score = max(max_score, score.item())
            scores.append(max_score)

        if self.conformal_predictor is not None:
            scores_tensor = torch.tensor(scores)
            self._calibration_scores = self._apply_external_predictor(scores_tensor, alpha)
        else:
            self._calibration_scores = torch.tensor(scores)
            n = len(scores)
            q_idx = min(n - 1, int(torch.ceil(torch.tensor((1.0 - alpha) * (n + 1))).item()) - 1)
            q_idx = max(0, q_idx)
            sorted_scores, _ = torch.sort(self._calibration_scores)
            self._calibration_quantile = sorted_scores[q_idx].item()

        self._alpha = alpha
        self._calibrated = True

    def _apply_external_predictor(self, scores: torch.Tensor, alpha: float) -> torch.Tensor:
        """Delegate quantile computation to an external conformal predictor."""
        cp = self.conformal_predictor
        if hasattr(cp, "calibrate"):
            cp.calibrate(scores.numpy(), alpha=alpha)
        return scores

    def predict_coverage(
        self,
        mask: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.1,
    ) -> CoverageResult:
        """Compute conformal-calibrated coverage for a single mask.

        Args:
            mask: Mask tensor to evaluate.
            target: Target design tensor.
            alpha: Miscoverage rate.

        Returns:
            CoverageResult with calibrated bands.
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict_coverage()")

        epe_samples = self.sampler.sample_epe(mask, n_samples=self.n_calibration_samples)
        lcdu_samples = self.sampler.sample_lcdu(mask, n_samples=self.n_calibration_samples)

        epe_median = epe_samples.median().item()
        epe_mad = (epe_samples - epe_samples.median()).abs().median().item()
        epe_mad = max(epe_mad, 1e-8)

        lcdu_mean = lcdu_samples.mean().item()
        lcdu_std = lcdu_samples.std().item()
        lcdu_std = max(lcdu_std, 1e-8)

        quantile = getattr(self, "_calibration_quantile", 1.0)

        epe_lower = max(0.0, epe_median - quantile * epe_mad)
        epe_upper = epe_median + quantile * epe_mad
        lcdu_lower = max(0.0, lcdu_mean - quantile * lcdu_std)
        lcdu_upper = lcdu_mean + quantile * lcdu_std

        empirical = self._compute_empirical_coverage(epe_samples, epe_median, epe_mad, quantile)

        process_window = self._compute_process_window(mask, epe_median, epe_mad, quantile)

        return CoverageResult(
            epe_band=(epe_lower, epe_upper),
            lcdu_band=(lcdu_lower, lcdu_upper),
            empirical_coverage=empirical,
            target_coverage=1.0 - alpha,
            process_window=process_window,
        )

    def _compute_empirical_coverage(
        self,
        epe_samples: torch.Tensor,
        epe_median: float,
        epe_mad: float,
        quantile: float,
    ) -> float:
        """Fraction of samples falling within the conformal band."""
        if epe_mad < 1e-12:
            return 1.0
        lower = epe_median - quantile * epe_mad
        upper = epe_median + quantile * epe_mad
        in_band = ((epe_samples >= lower) & (epe_samples <= upper)).float().mean()
        return in_band.item()

    def _compute_process_window(
        self,
        mask: torch.Tensor,
        epe_median: float,
        epe_mad: float,
        quantile: float,
        n_focus: int = 5,
        n_dose: int = 5,
        focus_range_nm: tuple[float, float] = (-50.0, 50.0),
        dose_range: tuple[float, float] = (0.9, 1.1),
    ) -> dict[str, bool]:
        """Evaluate coverage at each (focus, dose) grid point."""
        window: dict[str, bool] = {}
        focus_vals = torch.linspace(focus_range_nm[0], focus_range_nm[1], n_focus)
        dose_vals = torch.linspace(dose_range[0], dose_range[1], n_dose)

        for f_val in focus_vals:
            for d_val in dose_vals:
                epe_samples = self.sampler.sample_epe(
                    mask,
                    n_samples=20,
                    dose_range=(d_val.item(), d_val.item()),
                    focus_range=(f_val.item(), f_val.item()),
                )
                upper = epe_median + quantile * epe_mad
                key = f"f={f_val.item():.1f}_d={d_val.item():.2f}"
                window[key] = bool(epe_samples.median().item() <= upper)

        return window


class StochasticAcceptanceGate:
    """Acceptance gate using ThroughFocusCoverageCalibrator.

    Accepts a mask design if it meets all three criteria:
    - empirical coverage >= min_coverage
    - EPE band within max_epe_band spec
    - LCDU band within max_lcdu_band spec
    """

    def __init__(
        self,
        calibrator: ThroughFocusCoverageCalibrator | None = None,
        decision_gate: object | None = None,
    ) -> None:
        self.calibrator = calibrator or ThroughFocusCoverageCalibrator()
        self.decision_gate = decision_gate

    def evaluate(
        self,
        mask: torch.Tensor,
        target: torch.Tensor,
        min_coverage: float = 0.9,
        max_epe_band: tuple[float, float] | None = None,
        max_lcdu_band: tuple[float, float] | None = None,
        alpha: float = 0.1,
    ) -> tuple[bool, CoverageResult, list[str]]:
        """Decide if a mask design is acceptable.

        Args:
            mask: Mask tensor to evaluate.
            target: Target design tensor.
            min_coverage: Minimum required empirical coverage.
            max_epe_band: Maximum allowed (lower, upper) EPE band. If None,
                no EPE constraint is applied.
            max_lcdu_band: Maximum allowed (lower, upper) LCDU band. If None,
                no LCDU constraint is applied.
            alpha: Miscoverage rate for conformal calibration.

        Returns:
            Tuple of (accepted, result, reasons).
        """
        result = self.calibrator.predict_coverage(mask, target, alpha=alpha)

        reasons: list[str] = []

        coverage_ok = result.empirical_coverage >= min_coverage
        if not coverage_ok:
            reasons.append(f"Coverage {result.empirical_coverage:.3f} < min {min_coverage:.3f}")

        epe_ok = True
        if max_epe_band is not None:
            epe_ok = result.epe_band[1] <= max_epe_band[1]
            if not epe_ok:
                reasons.append(
                    f"EPE upper band {result.epe_band[1]:.4f} > spec {max_epe_band[1]:.4f}"
                )

        lcdu_ok = True
        if max_lcdu_band is not None:
            lcdu_ok = result.lcdu_band[1] <= max_lcdu_band[1]
            if not lcdu_ok:
                reasons.append(
                    f"LCDU upper band {result.lcdu_band[1]:.4f} > spec {max_lcdu_band[1]:.4f}"
                )

        accepted = coverage_ok and epe_ok and lcdu_ok

        if self.decision_gate is not None and hasattr(self.decision_gate, "decide"):
            accepted = bool(self.decision_gate.decide(accepted, result))

        return accepted, result, reasons


class ProcessWindowPlotter:
    """Computes the process window (focus x dose space where coverage is met)."""

    def __init__(
        self,
        sampler: StochasticSampler | None = None,
        epe_tolerance: float = 2.0,
    ) -> None:
        self.sampler = sampler or StochasticSampler()
        self.epe_tolerance = epe_tolerance
        self._window: torch.Tensor | None = None
        self._focus_values: torch.Tensor | None = None
        self._dose_values: torch.Tensor | None = None

    def compute_window(
        self,
        mask: torch.Tensor,
        target: torch.Tensor,
        focus_range: tuple[float, float] = (-50.0, 50.0),
        dose_range: tuple[float, float] = (0.9, 1.1),
        n_grid: int = 20,
        n_samples_per_point: int = 10,
    ) -> torch.Tensor:
        """Compute 2D boolean process window grid.

        Args:
            mask: Mask tensor.
            target: Target design tensor.
            focus_range: (min_focus_nm, max_focus_nm).
            dose_range: (min_dose, max_dose).
            n_grid: Grid resolution per axis.
            n_samples_per_point: Stochastic samples per grid point.

        Returns:
            Boolean tensor of shape ``(n_grid, n_grid)``.
        """
        mask_2d = mask.detach().float()
        if mask_2d.ndim > 2:
            mask_2d = mask_2d.squeeze()

        focus_vals = torch.linspace(focus_range[0], focus_range[1], n_grid)
        dose_vals = torch.linspace(dose_range[0], dose_range[1], n_grid)

        self._focus_values = focus_vals
        self._dose_values = dose_vals

        window = torch.zeros(n_grid, n_grid, dtype=torch.bool)

        for i, f_val in enumerate(focus_vals):
            for j, d_val in enumerate(dose_vals):
                epe_samples = self.sampler.sample_epe(
                    mask_2d,
                    n_samples=n_samples_per_point,
                    dose_range=(d_val.item(), d_val.item()),
                    focus_range=(f_val.item(), f_val.item()),
                )
                window[i, j] = epe_samples.mean().item() <= self.epe_tolerance

        self._window = window
        return window

    def window_area(self) -> float:
        """Fraction of (focus, dose) space that passes the gate.

        Must be called after ``compute_window()``.
        """
        if self._window is None:
            raise RuntimeError("Must call compute_window() before window_area()")
        return self._window.float().mean().item()
