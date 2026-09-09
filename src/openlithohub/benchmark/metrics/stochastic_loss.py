"""Differentiable stochastic metrics for stochastic-aware ILT optimization.

Extends the evaluation-only ``stochastic`` module with loss functions whose
gradients flow back through the aerial-image forward model, enabling ILT
optimisers to improve robustness against EUV photon shot noise *during*
optimisation rather than only scoring it afterwards.

Key ideas
---------
* **Normal approximation to Poisson** — for large lambda (typical EUV doses
  produce >100 photons per pixel), ``Poisson(lambda) ~ Normal(lambda, sqrt(lambda))``.
  This is differentiable via the reparameterisation trick: draw ``epsilon ~ N(0,1)``
  and compute ``lambda + epsilon * sqrt(lambda)``.
* **Soft edge detection** — Sobel gradient magnitude + sigmoid soft-thresholding
  replaces the hard connected-component labelling used in the evaluation path.
* **Risk measures** — CVaR (Conditional Value at Risk) and pinball quantile
  loss let the optimiser target worst-case stochastic outcomes rather than the
  mean, matching the foundry's defect-rate philosophy (zero microbridges, not
  zero mean bridging).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from openlithohub._constants import THRESHOLD_ICCAD16
from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.resist_model import differentiable_threshold

# ---------------------------------------------------------------------------
# Risk-measure loss functions
# ---------------------------------------------------------------------------


def cvar_loss(errors: torch.Tensor, alpha: float) -> torch.Tensor:
    """Conditional Value at Risk (CVaR) — differentiable.

    Given a tensor of per-sample *errors*, returns the mean of the worst
    ``(1 - alpha)`` fraction.  ``alpha = 0.95`` means "average the worst 5 %".

    Uses ``torch.topk`` which supports autograd — the gradient of the mean
    of the top-k values flows equally to each of the k selected elements.

    Args:
        errors: 1-D tensor of per-sample loss/error values.
        alpha: Confidence level in ``(0, 1)``.  Higher alpha = more
            risk-averse (fewer samples in the tail).

    Returns:
        Scalar tensor — differentiable CVaR estimate.
    """
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = errors.numel()
    if n == 0:
        return errors.new_zeros(())
    k = max(1, min(n, int(torch.ceil(torch.tensor((1.0 - alpha) * n)).item())))
    # topk is differentiable — gradients flow to the k largest elements.
    topk_values, _ = torch.topk(errors, k)
    return topk_values.mean()


def quantile_loss(errors: torch.Tensor, quantile: float) -> torch.Tensor:
    """Differentiable quantile loss (pinball loss).

    ``L(y, q) = max(q * (y - tau), (1 - q) * (tau - y))``
    where ``tau = quantile(errors, q)`` is detached (no gradient through the
    quantile estimate itself, only through the pinball shaping).

    This is a proper scoring rule for quantiles and is widely used in
    quantile regression. For ILT it provides a softer alternative to CVaR
    when the optimiser should target a specific quantile of the error
    distribution rather than the mean of the tail.

    Args:
        errors: 1-D tensor of per-sample loss/error values.
        quantile: Target quantile in ``(0, 1)``.

    Returns:
        Scalar tensor — differentiable pinball loss.
    """
    if quantile <= 0.0 or quantile >= 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    n = errors.numel()
    if n == 0:
        return errors.new_zeros(())
    tau = torch.quantile(errors.detach().float(), quantile)
    residual = errors - tau
    return torch.mean(torch.max(quantile * residual, (quantile - 1.0) * residual))


# ---------------------------------------------------------------------------
# Soft edge detection (differentiable)
# ---------------------------------------------------------------------------


def _soft_edge_mask(image: torch.Tensor, steepness: float = 10.0) -> torch.Tensor:
    """Differentiable edge detection via Sobel + sigmoid soft threshold.

    Returns a soft edge mask (values in [0, 1]) where 1 means "on the edge
    contour". This replaces the hard ``distance_transform + threshold``
    approach in the evaluation-only stochastic module.

    Args:
        image: 2-D tensor (H, W), typically a resist image with values in
            [0, 1].
        steepness: Controls the softness of the edge detection. Higher values
            give sharper edge masks.
    """
    inp = image.float().unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
    ).reshape(1, 1, 3, 3)
    gx = functional.conv2d(inp, sobel_x, padding=1)
    gy = functional.conv2d(inp, sobel_y, padding=1)
    magnitude = (gx.square() + gy.square() + 1e-8).sqrt().squeeze(0).squeeze(0)
    return torch.sigmoid(steepness * (magnitude - 0.1))


# ---------------------------------------------------------------------------
# Differentiable stochastic metrics
# ---------------------------------------------------------------------------


def _reparameterized_poisson(lambda_map: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Draw differentiable Poisson-like samples via the normal approximation.

    For large lambda (typical EUV doses), ``Poisson(lambda) ~ Normal(lambda,
    sqrt(lambda))``.  Using the reparameterisation trick ``z = lambda +
    epsilon * sqrt(lambda)`` with ``epsilon ~ N(0, 1)`` makes the samples
    differentiable with respect to ``lambda_map`` (and hence with respect to
    the aerial image and mask).

    Uses ``lambda = clamp(lambda_raw, min=0)`` so dark regions of the
    aerial image (intensity ~ 0) keep their physical photon counts; a
    softplus here would bias every dark pixel up to ``ln(2) * dose``
    photons and print spurious resist in unexposed areas.

    Returns:
        Tensor of shape ``(n_samples, H, W)`` with noised photon counts
        (clamped to >= 0).
    """
    eps = torch.randn(
        n_samples, *lambda_map.shape, device=lambda_map.device, dtype=lambda_map.dtype
    )
    # clamp(min=0) guards against tiny negative aerial values from
    # convolution ringing; gradient w.r.t. physically valid (>= 0) values
    # is unaffected.
    lambda_pos = lambda_map.clamp(min=0.0)
    # Floor sigma to prevent 0.5/sqrt(0) divergence; the floor only
    # activates when lambda < 1 (far below any realistic EUV dose per pixel).
    sigma = torch.sqrt(lambda_pos + 1e-3)
    samples = lambda_pos.unsqueeze(0) + eps * sigma.unsqueeze(0)
    return samples.clamp(min=0.0)


def differentiable_edge_error(
    mask: torch.Tensor,
    aerial_nominal: torch.Tensor,
    dose_photons_per_nm2: float,
    pixel_size_nm: float,
    n_samples: int = 16,
    resist_threshold: float = THRESHOLD_ICCAD16,
    steepness: float = 50.0,
) -> torch.Tensor:
    """Monte Carlo estimate of expected edge placement error under shot noise.

    For each of ``n_samples`` Monte Carlo trials:
    1. Draw a noised aerial image via the normal approximation to Poisson.
    2. Compute soft resist image (sigmoid threshold).
    3. Measure the pixel-wise absolute difference on the nominal edge contour.

    The result is the mean edge error across all trials, differentiable through
    the mask via the aerial image forward model.

    Args:
        mask: Target mask (H, W), values in [0, 1].
        aerial_nominal: Nominal aerial image (H, W), differentiable w.r.t. mask.
        dose_photons_per_nm2: Exposure dose in photons/nm^2.
        pixel_size_nm: Pixel size in nm.
        n_samples: Number of Monte Carlo samples.
        resist_threshold: Resist development threshold.
        steepness: Sigmoid steepness for soft resist threshold.

    Returns:
        Scalar tensor — expected edge error (differentiable).
    """
    pixel_area_nm2 = pixel_size_nm * pixel_size_nm
    dose_scale = dose_photons_per_nm2 * pixel_area_nm2
    lambda_map = aerial_nominal.clamp(min=0.0) * dose_scale

    noised_photons = _reparameterized_poisson(lambda_map, n_samples)
    noised_intensity = noised_photons / max(dose_scale, 1e-12)

    resist_nominal = differentiable_threshold(aerial_nominal, resist_threshold, steepness)
    edge_mask = _soft_edge_mask(resist_nominal)

    total_error = aerial_nominal.new_zeros(())
    for i in range(n_samples):
        resist_noised = differentiable_threshold(noised_intensity[i], resist_threshold, steepness)
        # softplus(x) ~ |x| for large |x| but is smooth at 0 — avoids the
        # non-differentiable kink of abs().
        pixel_diff = torch.nn.functional.softplus(
            resist_noised - resist_nominal
        ) + torch.nn.functional.softplus(resist_nominal - resist_noised)
        edge_weight = edge_mask.clamp(min=1e-8)
        weighted_error = (pixel_diff * edge_weight).sum() / edge_weight.sum().clamp(min=1.0)
        total_error = total_error + weighted_error

    return total_error / n_samples


def differentiable_lcdu(
    mask: torch.Tensor,
    aerial_nominal: torch.Tensor,
    dose_photons_per_nm2: float,
    pixel_size_nm: float,
    n_samples: int = 16,
    resist_threshold: float = THRESHOLD_ICCAD16,
    steepness: float = 50.0,
) -> torch.Tensor:
    """Differentiable LCDU (Local CD Uniformity) estimate under shot noise.

    LCDU measures the standard deviation of the critical dimension (CD) across
    stochastic trials. This differentiable version:
    1. Runs ``n_samples`` noised trials.
    2. For each trial, computes the "printed CD proxy" — the sum of the soft
       resist image along horizontal scanlines at the nominal edge locations.
    3. Returns the standard deviation of these per-trial CD proxies.

    Args:
        mask: Target mask (H, W), values in [0, 1].
        aerial_nominal: Nominal aerial image (H, W).
        dose_photons_per_nm2: Exposure dose in photons/nm^2.
        pixel_size_nm: Pixel size in nm.
        n_samples: Number of Monte Carlo samples.
        resist_threshold: Resist threshold.
        steepness: Sigmoid steepness.

    Returns:
        Scalar tensor — LCDU estimate in pixel units (differentiable).
    """
    pixel_area_nm2 = pixel_size_nm * pixel_size_nm
    dose_scale = dose_photons_per_nm2 * pixel_area_nm2
    lambda_map = aerial_nominal.clamp(min=0.0) * dose_scale

    noised_photons = _reparameterized_poisson(lambda_map, n_samples)
    noised_intensity = noised_photons / max(dose_scale, 1e-12)

    resist_nominal = differentiable_threshold(aerial_nominal, resist_threshold, steepness)
    edge_mask = _soft_edge_mask(resist_nominal)

    cd_proxies = []
    for i in range(n_samples):
        resist_noised = differentiable_threshold(noised_intensity[i], resist_threshold, steepness)
        # CD proxy: sum of soft-resist along the edge contour (horizontal sum
        # weighted by edge mask gives a local-width estimate).
        cd_proxy = (resist_noised * edge_mask).sum()
        cd_proxies.append(cd_proxy)

    cd_stack = torch.stack(cd_proxies)
    return cd_stack.std()


# ---------------------------------------------------------------------------
# Main loss module
# ---------------------------------------------------------------------------


class StochasticAwareLoss:
    """Differentiable stochastic metrics for ILT optimization.

    Provides a single ``forward(mask, aerial_image_fn)`` call that computes a
    weighted combination of stochastic edge error and LCDU, optionally
    filtered through a risk measure (CVaR or quantile loss) instead of a
    simple mean.

    Usage::

        loss_fn = StochasticAwareLoss(alpha=0.95, n_mc_samples=32)
        loss = loss_fn(mask, lambda m: simulate_aerial_image(m, sigma_px=2.0))
        loss.backward()  # gradients flow to mask

    Args:
        alpha: CVaR confidence level. Higher = more risk-averse.
        n_mc_samples: Number of Monte Carlo noise samples.
        dose_photons_per_nm2: EUV exposure dose.
        pixel_size_nm: Mask pixel size in nm.
        resist_threshold: Resist development threshold.
        steepness: Sigmoid steepness for soft resist.
        weight_edge_error: Weight for the edge-error term.
        weight_lcdu: Weight for the LCDU term.
        risk_measure: ``"cvar"`` or ``"mean"``.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        n_mc_samples: int = 16,
        dose_photons_per_nm2: float = 30.0,
        pixel_size_nm: float = 1.0,
        resist_threshold: float = THRESHOLD_ICCAD16,
        steepness: float = 50.0,
        weight_edge_error: float = 1.0,
        weight_lcdu: float = 1.0,
        risk_measure: str = "cvar",
    ):
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if risk_measure not in ("cvar", "mean"):
            raise ValueError(f"risk_measure must be 'cvar' or 'mean', got {risk_measure}")
        self.alpha = alpha
        self.n_mc_samples = n_mc_samples
        self.dose_photons_per_nm2 = dose_photons_per_nm2
        self.pixel_size_nm = pixel_size_nm
        self.resist_threshold = resist_threshold
        self.steepness = steepness
        self.weight_edge_error = weight_edge_error
        self.weight_lcdu = weight_lcdu
        self.risk_measure = risk_measure

    def forward(
        self,
        mask: torch.Tensor,
        aerial_image_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute the stochastic-aware loss.

        Args:
            mask: Continuous mask tensor (H, W), values in [0, 1].
            aerial_image_fn: Callable that takes a mask and returns the
                aerial image. Must be differentiable (e.g.
                ``simulate_aerial_image``).

        Returns:
            Scalar loss tensor (differentiable w.r.t. mask).
        """
        aerial_nominal = aerial_image_fn(mask)

        pixel_area_nm2 = self.pixel_size_nm * self.pixel_size_nm
        dose_scale = self.dose_photons_per_nm2 * pixel_area_nm2
        lambda_map = aerial_nominal.clamp(min=0.0) * dose_scale

        noised_photons = _reparameterized_poisson(lambda_map, self.n_mc_samples)
        noised_intensity = noised_photons / max(dose_scale, 1e-12)

        resist_nominal = differentiable_threshold(
            aerial_nominal, self.resist_threshold, self.steepness
        )
        edge_mask = _soft_edge_mask(resist_nominal)

        per_sample_errors = []
        cd_proxies = []
        for i in range(self.n_mc_samples):
            resist_noised = differentiable_threshold(
                noised_intensity[i], self.resist_threshold, self.steepness
            )
            pixel_diff = torch.nn.functional.softplus(
                resist_noised - resist_nominal
            ) + torch.nn.functional.softplus(resist_nominal - resist_noised)
            edge_weight = edge_mask.clamp(min=1e-8)
            weighted_error = (pixel_diff * edge_weight).sum() / edge_weight.sum().clamp(min=1.0)
            per_sample_errors.append(weighted_error)
            cd_proxies.append((resist_noised * edge_mask).sum())

        errors_tensor = torch.stack(per_sample_errors)

        if self.risk_measure == "cvar":
            edge_loss = cvar_loss(errors_tensor, self.alpha)
        else:
            edge_loss = errors_tensor.mean()

        if self.weight_lcdu > 0.0:
            cd_stack = torch.stack(cd_proxies)
            lcdu = cd_stack.std()
        else:
            lcdu = edge_loss.new_zeros(())

        return self.weight_edge_error * edge_loss + self.weight_lcdu * lcdu

    # The class-level docstring documents ``loss = loss_fn(mask, ...)``;
    # honour that calling convention instead of requiring ``.forward``.
    __call__ = forward


# ---------------------------------------------------------------------------
# Stochastic Process Window
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StochasticProcessWindowResult:
    """Result from a stochastic process window evaluation."""

    focus_values_nm: list[float]
    mean_epe_per_focus: list[float]
    worst_case_epe_per_focus: list[float]
    lower_bound_nm: float
    upper_bound_nm: float
    window_width_nm: float


class StochasticProcessWindow:
    """Evaluate stochastic process window through focus with shot noise.

    Computes EPE (edge placement error) through a range of focus values,
    with stochastic noise at each focus point. The process window is the
    range of focus values where the stochastic EPE remains below a
    tolerance.

    Args:
        n_samples: Monte Carlo samples per focus point.
        dose_photons_per_nm2: Exposure dose in photons/nm^2.
        pixel_size_nm: Pixel size in nm.
        sigma_px: Gaussian PSF width in pixels.
        resist_threshold: Resist development threshold.
        steepness: Sigmoid steepness for soft resist.
        epe_tolerance: Maximum acceptable EPE (in pixels) for the window.
    """

    def __init__(
        self,
        n_samples: int = 16,
        dose_photons_per_nm2: float = 30.0,
        pixel_size_nm: float = 1.0,
        sigma_px: float = 2.0,
        resist_threshold: float = THRESHOLD_ICCAD16,
        steepness: float = 50.0,
        epe_tolerance: float = 2.0,
    ):
        self.n_samples = n_samples
        self.dose_photons_per_nm2 = dose_photons_per_nm2
        self.pixel_size_nm = pixel_size_nm
        self.sigma_px = sigma_px
        self.resist_threshold = resist_threshold
        self.steepness = steepness
        self.epe_tolerance = epe_tolerance

    def _aerial_with_defocus(self, mask: torch.Tensor, defocus_nm: float) -> torch.Tensor:
        """Compute aerial image with defocus via modified sigma.

        Defocus is approximated by increasing the PSF sigma. In real optical
        systems defocus broadens the point spread function; a simple
        approximation is ``sigma_eff = sigma_0 * sqrt(1 + (defocus / DOF)^2)``.
        """
        if abs(defocus_nm) < 1e-6:
            return simulate_aerial_image(mask, sigma_px=self.sigma_px, dose=1.0)
        # Depth of focus approximation (typical EUV DOF ~ 100 nm).
        dof_nm = 100.0
        sigma_eff = self.sigma_px * (1.0 + (defocus_nm / dof_nm) ** 2) ** 0.5
        return simulate_aerial_image(mask, sigma_px=sigma_eff, dose=1.0)

    def compute(
        self,
        mask: torch.Tensor,
        aerial_image_fn: Callable[..., torch.Tensor] | None = None,
        focus_range_nm: tuple[float, float, float] = (-50.0, 50.0, 10.0),
    ) -> StochasticProcessWindowResult:
        """Compute the stochastic process window.

        Args:
            mask: Continuous mask tensor (H, W).
            aerial_image_fn: Optional override for the aerial image function.
                Called as ``fn(mask, defocus_nm)`` when it accepts two
                positional arguments (so the focus sweep varies defocus),
                otherwise as ``fn(mask)`` — in that fallback the same
                nominal image is reused for every focus point.
                If ``None``, uses the built-in defocus model.
            focus_range_nm: ``(start, stop, step)`` in nm for the focus sweep.

        Returns:
            StochasticProcessWindowResult with window bounds.
        """
        start, stop, step = focus_range_nm
        n_steps = max(1, int((stop - start) / step)) + 1
        focus_values = [start + i * step for i in range(n_steps)]

        mean_epe = []
        worst_epe = []

        # Custom forward models may or may not accept a defocus argument;
        # detect by signature so the sweep actually varies focus when
        # supported (otherwise the same nominal image is reused for every
        # focus point).
        takes_defocus = False
        if aerial_image_fn is not None:
            try:
                takes_defocus = len(inspect.signature(aerial_image_fn).parameters) >= 2
            except (TypeError, ValueError):
                takes_defocus = False

        pixel_area_nm2 = self.pixel_size_nm * self.pixel_size_nm
        dose_scale = self.dose_photons_per_nm2 * pixel_area_nm2

        for defocus_nm in focus_values:
            if aerial_image_fn is not None:
                if takes_defocus:
                    aerial_nominal = aerial_image_fn(mask, defocus_nm)
                else:
                    aerial_nominal = aerial_image_fn(mask)
            else:
                aerial_nominal = self._aerial_with_defocus(mask, defocus_nm)

            lambda_map = aerial_nominal.clamp(min=0.0) * dose_scale
            noised_photons = _reparameterized_poisson(lambda_map, self.n_samples)
            noised_intensity = noised_photons / max(dose_scale, 1e-12)

            resist_nominal = differentiable_threshold(
                aerial_nominal, self.resist_threshold, self.steepness
            )
            edge_mask = _soft_edge_mask(resist_nominal)

            per_sample_epe = []
            for i in range(self.n_samples):
                resist_noised = differentiable_threshold(
                    noised_intensity[i], self.resist_threshold, self.steepness
                )
                pixel_diff = (resist_noised - resist_nominal).abs()
                edge_weight = edge_mask.clamp(min=1e-8)
                epe_i = (pixel_diff * edge_weight).sum() / edge_weight.sum().clamp(min=1.0)
                per_sample_epe.append(epe_i.item())

            mean_epe.append(sum(per_sample_epe) / len(per_sample_epe))
            worst_epe.append(max(per_sample_epe))

        # Find window bounds: the longest *contiguous* run of focus points
        # whose mean EPE stays below tolerance. Taking min/max over all
        # passing points would report a wide window even when an interior
        # focus fails (e.g. ±50 nm pass but best focus fails). When no
        # point passes, the window collapses to zero width.
        in_window = [m < self.epe_tolerance for m in mean_epe]
        lower = focus_values[0]
        upper = focus_values[0]
        best_width = 0
        run_start = None
        for idx, ok in enumerate(in_window + [False]):
            if ok and run_start is None:
                run_start = idx
            elif not ok and run_start is not None:
                width = idx - run_start
                if width > best_width:
                    best_width = width
                    lower = focus_values[run_start]
                    upper = focus_values[idx - 1]
                run_start = None

        return StochasticProcessWindowResult(
            focus_values_nm=focus_values,
            mean_epe_per_focus=mean_epe,
            worst_case_epe_per_focus=worst_epe,
            lower_bound_nm=lower,
            upper_bound_nm=upper,
            window_width_nm=max(0.0, upper - lower),
        )
