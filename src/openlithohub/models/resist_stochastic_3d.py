"""3D resist profile stochastic model with spatially correlated secondary electrons.

Models through-thickness stochastic variation in EUV resist exposure by combining:
  - Spatially correlated secondary electron (SE) generation kernel
  - 3D resist profile (thickness dimension) with line-collapse risk analysis
  - Full 3D stochastic defect simulation with Monte Carlo trials
  - Conformal coverage-calibrated acceptance gate
  - Beyond-LER/LCDU stochastic metrics: failure correlation length, defect
    cluster distribution, and stochastic edge placement error quantiles

References:
    - Fukuda et al., "Spatial correlation probability model for EUV stochastic
      analysis", Proc. SPIE 11147 (2019).
    - imec EUV Accelerator program, 2025-2026 stochastic defectivity roadmap.
    - Siemens Calibre, "Calibration and verification metrics for EUVL stochastic
      models: beyond LER and LCDU", Proc. SPIE 2026.
    - IBM, "demonstrates High-NA EUV below 2 nm nodes at SPIE 2026",
      research.ibm.com, 2026-02.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from openlithohub._constants import THRESHOLD_ICCAD16
from openlithohub._utils.forward_model import apply_resist_threshold, simulate_aerial_image
from openlithohub._utils.tensor_ops import ensure_2d

__all__ = [
    "SecondaryElectronKernel",
    "ResistProfile3D",
    "StochasticDefectModel3D",
    "ConformalCoverageGate3D",
    "Stochastic3DBenchmark",
    "DefectClusterMetrics",
    "StochasticCalibrationMetrics",
]


# ---------------------------------------------------------------------------
# Secondary electron spatial correlation kernel
# ---------------------------------------------------------------------------


class SecondaryElectronKernel:
    """Spatial correlation kernel for secondary electron (SE) exposure events.

    In EUV lithography, a single incident photon generates a cloud of
    secondary electrons whose spatial extent is governed by inelastic
    scattering.  This kernel models that spatial correlation, following
    Fukuda et al. SPIE 11147.

    Parameters
    ----------
    correlation_length_nm : float
        Characteristic SE scattering range in nm.  Typical EUV photoresist
        values are 1--5 nm (Fukuda reports ~2 nm for chemically-amplified
        resists).
    kernel_type : str
        ``"gaussian"`` (default) or ``"exponential"`` correlation shape.
    """

    def __init__(
        self,
        correlation_length_nm: float = 2.0,
        kernel_type: str = "gaussian",
    ) -> None:
        if correlation_length_nm <= 0.0:
            raise ValueError(f"correlation_length_nm must be > 0, got {correlation_length_nm}")
        if kernel_type not in ("gaussian", "exponential"):
            raise ValueError(f"kernel_type must be 'gaussian' or 'exponential', got {kernel_type}")
        self.correlation_length_nm = correlation_length_nm
        self.kernel_type = kernel_type

    def compute_kernel(
        self,
        grid_size: int,
        pixel_size_nm: float = 1.0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build a 3D spatial correlation kernel.

        Parameters
        ----------
        grid_size : int
            Spatial extent of the kernel in pixels (same for x, y, z).
        pixel_size_nm : float
            Pixel pitch in nm.
        device : torch.device or None
            Device for the returned kernel. Defaults to CPU.

        Returns
        -------
        Tensor, shape ``(grid_size, grid_size, grid_size)``, normalised so
        that the kernel sums to 1.
        """
        # (grid_size - 1) // 2 keeps the coordinate list at exactly
        # ``grid_size`` entries for even sizes too.
        radius = (grid_size - 1) // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        # shape (grid_size, 1, 1), (1, grid_size, 1), (1, 1, grid_size)
        dx = coords.unsqueeze(1).unsqueeze(2)
        dy = coords.unsqueeze(0).unsqueeze(2)
        dz = coords.unsqueeze(0).unsqueeze(0)
        r2 = dx**2 + dy**2 + dz**2  # (grid_size, grid_size, grid_size)

        xi = pixel_size_nm / self.correlation_length_nm
        if self.kernel_type == "gaussian":
            kernel = torch.exp(-0.5 * r2 * xi**2)
        else:  # exponential
            r = torch.sqrt(r2 + 1e-12)
            kernel = torch.exp(-r * xi)

        kernel = kernel / kernel.sum()
        return kernel

    def apply_correlation(
        self,
        white_noise: torch.Tensor,
        kernel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convolve white noise with the 3D SE correlation kernel.

        Parameters
        ----------
        white_noise : Tensor
            Shape ``(D, H, W)`` or ``(B, D, H, W)``.  Uncorrelated noise.
        kernel : Tensor or None
            Precomputed kernel from :meth:`compute_kernel`.  If ``None`` a
            default ``3x3x3`` kernel is built.

        Returns
        -------
        Spatially correlated noise tensor, same shape as input.
        """
        if kernel is None:
            kernel = self.compute_kernel(3)

        # The kernel may be precomputed on a different device (torch.arange
        # defaults to CPU); conv3d requires input and weight to match.
        kernel = kernel.to(device=white_noise.device, dtype=torch.float32)
        k = kernel.unsqueeze(0).unsqueeze(0).float()  # (1, 1, kD, kH, kW)
        # F.pad for 5D tensors expects (W_left, W_right, H_left, H_right, D_left, D_right)
        pd = kernel.shape[0] // 2
        pad3d = (pd, pd, pd, pd, pd, pd)

        if white_noise.ndim == 3:
            inp = white_noise.unsqueeze(0).unsqueeze(0).float()  # (1, 1, D, H, W)
            padded = functional.pad(inp, pad3d, mode="replicate")
            out = functional.conv3d(padded, k)
            return out.squeeze(0).squeeze(0)
        elif white_noise.ndim == 4:
            b = white_noise.shape[0]
            inp = white_noise.unsqueeze(1).float()  # (B, 1, D, H, W)
            padded = functional.pad(inp, pad3d, mode="replicate")
            out = functional.conv3d(padded, k.expand(b, -1, -1, -1, -1), groups=b)
            return out.squeeze(1)
        else:
            raise ValueError(f"white_noise must be 3D or 4D, got {white_noise.ndim}D")


# ---------------------------------------------------------------------------
# 3D resist profile
# ---------------------------------------------------------------------------


class ResistProfile3D:
    """3D resist profile model extending through the resist thickness.

    Generates a ``(D, H, W)`` volume from a 2D mask and optional 3D aerial
    image.  Models the through-thickness variation due to absorption,
    standing waves, and top-loss that are invisible to 2D models.

    Parameters
    ----------
    thickness_nm : float
        Resist thickness in nm.
    aspect_ratio_limit : float
        Maximum aspect ratio (height / CD) above which line collapse risk
        is flagged.
    pixel_size_nm : float
        Pixel pitch in nm.
    """

    def __init__(
        self,
        thickness_nm: float = 20.0,
        aspect_ratio_limit: float = 3.0,
        pixel_size_nm: float = 1.0,
    ) -> None:
        if thickness_nm <= 0.0:
            raise ValueError(f"thickness_nm must be > 0, got {thickness_nm}")
        self.thickness_nm = thickness_nm
        self.aspect_ratio_limit = aspect_ratio_limit
        self.pixel_size_nm = pixel_size_nm
        self.n_slices = max(1, int(round(thickness_nm / pixel_size_nm)))

    def generate_profile(
        self,
        mask_2d: torch.Tensor,
        aerial_3d: torch.Tensor | None = None,
        sigma_px: float = 2.0,
    ) -> torch.Tensor:
        """Generate a 3D resist profile from a 2D mask pattern.

        Parameters
        ----------
        mask_2d : Tensor
            ``(H, W)`` binary or continuous mask.
        aerial_3d : Tensor or None
            Optional ``(D, H, W)`` aerial image stack.  If ``None`` the
            method synthesises one from the 2D mask using Beer-Lambert
            absorption through the resist thickness.
        sigma_px : float
            Gaussian PSF sigma for the built-in aerial model.

        Returns
        -------
        Tensor, shape ``(D, H, W)`` with values in [0, 1] representing
        the developed resist fraction at each depth slice.
        """
        m = ensure_2d(mask_2d)
        binary = (m > 0.5).float()
        h, w = binary.shape

        if aerial_3d is not None:
            stack = aerial_3d
        else:
            aerial_2d = simulate_aerial_image(binary, sigma_px=sigma_px, dose=1.0)
            # Beer-Lambert absorption: intensity decays as exp(-alpha * z)
            # Typical CAR absorption coefficient alpha ~ 5 /um => 0.005 /nm
            alpha_per_nm = 0.005
            z_indices = torch.arange(self.n_slices, dtype=torch.float32)
            absorption = torch.exp(-alpha_per_nm * z_indices * self.pixel_size_nm)
            # Standing-wave modulation (simplified): adds ~5 % swing at lambda/(2n)
            # For 13.5 nm / n~1.7 => ~4 nm period
            swing_period_nm = 4.0
            swing_amplitude = 0.05
            phase = 2.0 * 3.14159265 * z_indices * self.pixel_size_nm / swing_period_nm
            swing = 1.0 + swing_amplitude * torch.cos(phase)
            # (D, H, W) = (D, 1, 1) * (1, H, W)
            intensity_3d = absorption.unsqueeze(1).unsqueeze(2) * swing.unsqueeze(1).unsqueeze(2)
            stack = aerial_2d.unsqueeze(0) * intensity_3d

        # Apply resist threshold at each depth slice
        profile = torch.zeros_like(stack)
        for z in range(stack.shape[0]):
            profile[z] = apply_resist_threshold(
                stack[z],
                threshold=THRESHOLD_ICCAD16,
                pixel_size_nm=self.pixel_size_nm,
            )

        return profile

    def compute_line_collapse_risk(self, profile_3d: torch.Tensor) -> float:
        """Estimate the probability of line collapse due to capillary forces.

        Line collapse risk increases with aspect ratio (height / linewidth).
        Thin lines at the top of the resist are most vulnerable during
        development rinse.  This heuristic scores risk from the 3D profile.

        Parameters
        ----------
        profile_3d : Tensor
            ``(D, H, W)`` resist volume.

        Returns
        -------
        Scalar risk metric in [0, 1].  Values > 0.5 indicate elevated risk.
        """
        d, h, w = profile_3d.shape
        if d < 2:
            return 0.0

        # Measure average linewidth at top and bottom slices
        top_slice = profile_3d[-1]
        bot_slice = profile_3d[0]

        # Average CD along rows: sum of resist pixels per row, averaged
        top_cds = top_slice.sum(dim=1)
        bot_cds = bot_slice.sum(dim=1)
        avg_top_cd = top_cds[top_cds > 0].float().mean().item() if (top_cds > 0).any() else 0.0
        avg_bot_cd = bot_cds[bot_cds > 0].float().mean().item() if (bot_cds > 0).any() else 0.0

        # Effective CD (minimum of top and bottom) in pixel units
        cd_px = max(1.0, min(avg_top_cd, avg_bot_cd))
        cd_nm = cd_px * self.pixel_size_nm
        aspect_ratio = self.thickness_nm / max(cd_nm, 1.0)

        # Sigmoid risk: low below aspect_ratio_limit, rising steeply above
        steepness = 5.0
        risk = 1.0 / (
            1.0 + torch.exp(torch.tensor(-steepness * (aspect_ratio - self.aspect_ratio_limit)))
        )
        return risk.item()

    def compute_lcdu_3d(self, profile_3d: torch.Tensor) -> torch.Tensor:
        """Compute 3D LCDU (Local CD Uniformity) across the profile.

        LCDU measures the per-slice standard deviation of the resist width,
        then aggregates across depth.  A 3D LCDU captures through-thickness
        variation invisible to 2D metrics.

        Parameters
        ----------
        profile_3d : Tensor
            ``(D, H, W)`` resist volume (ensemble of stochastic trials is
            expected as ``(N, D, H, W)`` for meaningful statistics; for a
            single profile, returns per-slice CD variation).

        Returns
        -------
        Tensor, shape ``(D,)`` with per-slice LCDU in pixel units.
        """
        if profile_3d.ndim == 4:
            # (N, D, H, W): ensemble of trials. LCDU is the trial-to-trial
            # CD variation, so take the std across trials per row first and
            # then average over rows — a flat std over (N, H) would mix the
            # deterministic row-to-row (pattern) variation into the metric.
            n, d, h, w = profile_3d.shape
            lcdu_per_slice = torch.zeros(d, device=profile_3d.device)
            for z in range(d):
                # Per-trial CD: sum of resist pixels along columns for each row
                cds = profile_3d[:, z, :, :].sum(dim=2).float()  # (N, H)
                lcdu_per_slice[z] = cds.std(dim=0).mean()
            return lcdu_per_slice
        else:
            # Single (D, H, W) profile: CD per row
            d, h, w = profile_3d.shape
            lcdu_per_slice = torch.zeros(d, device=profile_3d.device)
            for z in range(d):
                row_cds = profile_3d[z, :, :].sum(dim=1).float()
                lcdu_per_slice[z] = row_cds.std()
            return lcdu_per_slice

    def vertical_correlation(self, profile_3d: torch.Tensor) -> torch.Tensor:
        """Inter-layer correlation across the resist thickness.

        Measures Pearson correlation between adjacent depth slices.
        High correlation indicates the stochastic pattern is coherent
        through the thickness; low correlation suggests de-correlated
        defectivity at different depths.

        Parameters
        ----------
        profile_3d : Tensor
            ``(D, H, W)`` resist volume.

        Returns
        -------
        Tensor, shape ``(D - 1,)`` with correlation between slice ``i``
        and ``i + 1``, for ``i in [0, D - 2]``.
        """
        d = profile_3d.shape[0]
        if d < 2:
            return torch.tensor([])

        correlations = []
        for i in range(d - 1):
            s1 = profile_3d[i].flatten().float()
            s2 = profile_3d[i + 1].flatten().float()
            if s1.std() < 1e-8 or s2.std() < 1e-8:
                correlations.append(torch.tensor(1.0 if torch.equal(s1, s2) else 0.0))
            else:
                # Pearson r with population (1/N) moments so the
                # denominator matches the numerator's normalisation —
                # torch.std defaults to the unbiased (1/(N-1)) estimate,
                # which would bias the ratio by (N-1)/N.
                c1 = s1 - s1.mean()
                c2 = s2 - s2.mean()
                corr = torch.dot(c1, c2) / (
                    s1.numel() * c1.std(unbiased=False) * c2.std(unbiased=False)
                )
                correlations.append(corr)
        return torch.stack(correlations)


# ---------------------------------------------------------------------------
# 3D stochastic defect model
# ---------------------------------------------------------------------------


class StochasticDefectModel3D:
    """Full 3D stochastic defect model combining SE kernel, 3D profile, and MC simulation.

    Parameters
    ----------
    se_kernel : SecondaryElectronKernel
        Spatial correlation model for secondary electron generation.
    resist : ResistProfile3D
        3D resist profile model.
    dose_photons : float
        Exposure dose in photons / nm^2 at the wafer.
    sigma_px : float
        Gaussian PSF sigma in pixels for the aerial image model.
    """

    def __init__(
        self,
        se_kernel: SecondaryElectronKernel | None = None,
        resist: ResistProfile3D | None = None,
        dose_photons: float = 30.0,
        sigma_px: float = 2.0,
    ) -> None:
        self.se_kernel = se_kernel or SecondaryElectronKernel()
        self.resist = resist or ResistProfile3D()
        self.dose_photons = dose_photons
        self.sigma_px = sigma_px

    def simulate_defects(
        self,
        mask: torch.Tensor,
        n_mc: int = 100,
        seed: int = 0,
    ) -> torch.Tensor:
        """Run Monte Carlo stochastic defect simulation in 3D.

        For each trial:
        1. Generate the nominal 3D aerial image.
        2. Draw Poisson photon noise, spatially correlated via the SE kernel.
        3. Apply resist threshold at each depth.
        4. Record where the noisy resist differs from the nominal.

        Parameters
        ----------
        mask : Tensor
            ``(H, W)`` binary mask.
        n_mc : int
            Number of Monte Carlo trials.
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        Tensor, shape ``(D, H, W)`` with per-voxel defect probability in [0, 1].
        """
        m = ensure_2d(mask)
        binary = (m > 0.5).float()
        h, w = binary.shape

        # Nominal 3D aerial image via Beer-Lambert + standing-wave model
        aerial_2d = simulate_aerial_image(binary, sigma_px=self.sigma_px, dose=1.0)
        n_slices = self.resist.n_slices

        alpha_per_nm = 0.005
        z_indices = torch.arange(n_slices, dtype=torch.float32)
        absorption = torch.exp(-alpha_per_nm * z_indices * self.resist.pixel_size_nm)
        swing_period_nm = 4.0
        swing_amplitude = 0.05
        phase = 2.0 * 3.14159265 * z_indices * self.resist.pixel_size_nm / swing_period_nm
        swing = 1.0 + swing_amplitude * torch.cos(phase)
        intensity_3d = absorption.unsqueeze(1).unsqueeze(2) * swing.unsqueeze(1).unsqueeze(2)
        aerial_3d_nominal = aerial_2d.unsqueeze(0) * intensity_3d  # (D, H, W)

        # Nominal resist profile
        nominal_profile = self.resist.generate_profile(mask, aerial_3d_nominal, self.sigma_px)

        pixel_area_nm2 = self.resist.pixel_size_nm**2
        dose_scale = self.dose_photons * pixel_area_nm2

        # Precompute SE correlation kernel for noise generation. Built on
        # the mask's device so the convolutions below do not hit a
        # CPU/GPU device mismatch.
        kernel_size = min(5, min(h, w))
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel_size = max(3, kernel_size)
        se_kernel = self.se_kernel.compute_kernel(
            kernel_size, self.resist.pixel_size_nm, device=mask.device
        )

        generator = torch.Generator(device=mask.device)
        generator.manual_seed(seed)

        defect_accumulator = torch.zeros_like(nominal_profile)

        for _ in range(n_mc):
            # Poisson noise at each depth
            lambda_3d = aerial_3d_nominal.clamp(min=0.0) * dose_scale
            white_noise = torch.randn(
                lambda_3d.shape,
                generator=generator,
                device=lambda_3d.device,
                dtype=lambda_3d.dtype,
            )
            # Spatial correlation applied slice-by-slice for memory efficiency
            correlated_noise = torch.zeros_like(lambda_3d)
            k2d = se_kernel[kernel_size // 2]  # middle slice = 2D projection
            k2d_norm = k2d / k2d.sum().clamp(min=1e-12)
            k2d_conv = k2d_norm.unsqueeze(0).unsqueeze(0)
            pad_2d = kernel_size // 2

            for z in range(n_slices):
                sigma_z = torch.sqrt(lambda_3d[z].clamp(min=1e-3))
                noised_2d = lambda_3d[z] + white_noise[z] * sigma_z
                noised_2d = noised_2d.clamp(min=0.0)

                # Apply 2D spatial correlation via convolution
                inp = noised_2d.unsqueeze(0).unsqueeze(0)
                inp_padded = functional.pad(inp, (pad_2d, pad_2d, pad_2d, pad_2d), mode="replicate")
                correlated = functional.conv2d(inp_padded, k2d_conv).squeeze(0).squeeze(0)

                # Rescale to match original statistics
                if correlated.std() > 1e-8:
                    correlated = (
                        correlated - correlated.mean()
                    ) / correlated.std() * noised_2d.std() + noised_2d.mean()
                correlated_noise[z] = correlated.clamp(min=0.0)

            # Apply resist threshold at each depth
            noisy_resist = torch.zeros_like(nominal_profile)
            for z in range(n_slices):
                intensity_z = correlated_noise[z] / max(dose_scale, 1e-12)
                noisy_resist[z] = apply_resist_threshold(
                    intensity_z,
                    threshold=THRESHOLD_ICCAD16,
                    pixel_size_nm=self.resist.pixel_size_nm,
                )

            # Accumulate defects
            defect_accumulator += (noisy_resist != nominal_profile).float()

        return defect_accumulator / n_mc

    def compute_failure_rate_3d(
        self,
        mask: torch.Tensor,
        n_mc: int = 500,
        seed: int = 0,
    ) -> torch.Tensor:
        """Compute per-depth failure rates.

        Parameters
        ----------
        mask : Tensor
            ``(H, W)`` binary mask.
        n_mc : int
            Number of Monte Carlo trials.
        seed : int
            RNG seed.

        Returns
        -------
        Tensor, shape ``(D,)`` with per-depth failure rates in [0, 1].
        """
        defect_map = self.simulate_defects(mask, n_mc=n_mc, seed=seed)
        # Average failure rate across spatial dims for each depth slice
        return defect_map.mean(dim=(1, 2))

    def through_focus_quantile(
        self,
        mask: torch.Tensor,
        focus_range: tuple[float, float, float] = (-50.0, 50.0, 10.0),
        alpha: float = 0.1,
        n_mc: int = 20,
        seed: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute through-focus defect quantiles.

        Sweeps defocus and computes the alpha-quantile of per-pixel defect
        probability at each focus position.

        Parameters
        ----------
        mask : Tensor
            ``(H, W)`` binary mask.
        focus_range : tuple
            ``(start_nm, stop_nm, step_nm)`` for focus sweep.
        alpha : float
            Quantile level (0.1 = worst 10 %).
        n_mc : int
            MC trials per focus point.
        seed : int
            RNG seed.

        Returns
        -------
        dict with:
        - ``focus_values``: ``(N,)`` tensor of focus positions.
        - ``quantile_defects``: ``(N,)`` tensor of alpha-quantile defect rates.
        - ``mean_defects``: ``(N,)`` tensor of mean defect rates.
        """
        start, stop, step = focus_range
        n_steps = max(1, int((stop - start) / step)) + 1
        focus_values = [start + i * step for i in range(n_steps)]

        m = ensure_2d(mask)
        binary = (m > 0.5).float()
        sigma_base = self.sigma_px

        quantiles = []
        means = []

        generator = torch.Generator(device=mask.device)
        generator.manual_seed(seed)

        pixel_area_nm2 = self.resist.pixel_size_nm**2
        dose_scale = self.dose_photons * pixel_area_nm2

        for defocus_nm in focus_values:
            # Defocus broadens the PSF
            dof_nm = 100.0
            sigma_eff = sigma_base * (1.0 + (defocus_nm / dof_nm) ** 2) ** 0.5

            aerial = simulate_aerial_image(binary, sigma_px=sigma_eff, dose=1.0)
            nominal_resist = apply_resist_threshold(
                aerial,
                threshold=THRESHOLD_ICCAD16,
                pixel_size_nm=self.resist.pixel_size_nm,
            )

            lambda_map = aerial.clamp(min=0.0) * dose_scale
            trial_defect_fracs = []

            for _ in range(n_mc):
                photons = torch.poisson(lambda_map, generator=generator)
                noisy_intensity = photons / max(dose_scale, 1e-12)
                noisy_resist = apply_resist_threshold(
                    noisy_intensity,
                    threshold=THRESHOLD_ICCAD16,
                    pixel_size_nm=self.resist.pixel_size_nm,
                )
                defect_frac = (noisy_resist != nominal_resist).float().mean().item()
                trial_defect_fracs.append(defect_frac)

            trial_tensor = torch.tensor(trial_defect_fracs)
            quantiles.append(torch.quantile(trial_tensor, 1.0 - alpha).item())
            means.append(trial_tensor.mean().item())

        return {
            "focus_values": torch.tensor(focus_values),
            "quantile_defects": torch.tensor(quantiles),
            "mean_defects": torch.tensor(means),
        }


# ---------------------------------------------------------------------------
# Conformal coverage gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageMetrics:
    """Metrics from a conformal coverage evaluation."""

    coverage: float
    mean_bandwidth: float
    verdict: str
    n_violations: int


class ConformalCoverageGate3D:
    """Coverage-calibrated acceptance gate for 3D stochastic predictions.

    Uses split conformal prediction to provide finite-sample coverage
    guarantees on defect-rate predictions.  The gate decides ACCEPT /
    REJECT / UNCERTAIN based on whether the coverage of the prediction
    interval meets the target.

    Parameters
    ----------
    target_coverage : float
        Desired marginal coverage level (e.g. 0.9 for 90 %).
    alpha : float
        Significance level for conformal prediction.
    """

    def __init__(
        self,
        target_coverage: float = 0.9,
        alpha: float = 0.1,
    ) -> None:
        if not (0.0 < target_coverage < 1.0):
            raise ValueError(f"target_coverage must be in (0, 1), got {target_coverage}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.target_coverage = target_coverage
        self.alpha = alpha
        self._predictor: Any = None
        self._gate: Any = None

    def calibrate(
        self,
        masks: list[torch.Tensor],
        ground_truth_defects: list[torch.Tensor],
        predicted_defects: list[torch.Tensor] | None = None,
    ) -> None:
        """Calibrate the conformal predictor on a held-out calibration set.

        Parameters
        ----------
        masks : list of Tensor
            Calibration mask patterns, each ``(H, W)``.
        ground_truth_defects : list of Tensor
            True defect probability maps, each ``(H, W)`` or ``(D, H, W)``.
        predicted_defects : list of Tensor or None
            Model predictions matching what :meth:`evaluate` will feed at
            inference time. When omitted, the mask mean is used as a
            self-consistency proxy (matching the historical behaviour).
        """
        from diff_surrogate.conformal import SplitConformalPredictor
        from diff_surrogate.decision import AcceptRejectGate

        # Flatten spatial dims for conformal calibration. Calibrate against
        # the same predictor statistic evaluate() consumes (the predicted
        # defect mean), otherwise the conformal bands do not apply to the
        # gated prediction.
        preds = predicted_defects if predicted_defects is not None else masks
        cal_preds_list = []
        cal_targets_list = []
        for _m, p, gt in zip(masks, preds, ground_truth_defects, strict=False):
            cal_preds_list.append(p.flatten().float().mean().unsqueeze(0))
            cal_targets_list.append(gt.flatten().float().mean().unsqueeze(0))

        cal_preds = torch.cat(cal_preds_list)
        cal_targets = torch.cat(cal_targets_list)

        self._predictor = SplitConformalPredictor()
        self._predictor.calibrate(cal_preds, cal_targets, alpha=self.alpha)

        self._gate = AcceptRejectGate(
            min_coverage=self.target_coverage,
        )

    def evaluate(
        self,
        mask: torch.Tensor,
        predicted_defects: torch.Tensor,
        actual_defects: torch.Tensor,
    ) -> tuple[Any, CoverageMetrics]:
        """Evaluate a single prediction against ground truth.

        Parameters
        ----------
        mask : Tensor
            Input mask ``(H, W)``.
        predicted_defects : Tensor
            Predicted defect probability map.
        actual_defects : Tensor
            Ground truth defect probability map.

        Returns
        -------
        (DecisionVerdict, CoverageMetrics) tuple.
        """
        from diff_surrogate.conformal import coverage_score
        from diff_surrogate.decision import AcceptRejectGate

        pred_mean = predicted_defects.flatten().float().mean().unsqueeze(0)
        actual_mean = actual_defects.flatten().float().mean().unsqueeze(0)

        if self._predictor is not None:
            lower, upper = self._predictor.predict(pred_mean)
            achieved_coverage = coverage_score(actual_mean, lower, upper, alpha=self.alpha)
            # NOTE: dict.get's default is evaluated eagerly, so computing a
            # boolean fallback inline would raise on multi-element bounds
            # even when the "coverage" key exists. Branch explicitly and
            # reduce elementwise.
            if "coverage" in achieved_coverage:
                cov_value = achieved_coverage["coverage"]
            else:
                inside = (actual_mean >= lower) & (actual_mean <= upper)
                cov_value = float(inside.float().mean().item())
        else:
            # Fallback: simple interval around prediction
            bandwidth = max((predicted_defects.flatten().float().std().item(), 0.01))
            lower = pred_mean - bandwidth
            upper = pred_mean + bandwidth
            cov_value = 1.0 if (actual_mean >= lower and actual_mean <= upper) else 0.0

        mean_bandwidth = (upper - lower).item() / 2.0

        gate = self._gate or AcceptRejectGate(min_coverage=self.target_coverage)
        verdict, gate_info = gate.evaluate(pred_mean, lower, upper, coverage=cov_value)

        n_violations = int((predicted_defects > actual_defects + mean_bandwidth).sum().item())

        metrics = CoverageMetrics(
            coverage=cov_value if isinstance(cov_value, float) else float(cov_value),
            mean_bandwidth=mean_bandwidth,
            verdict=verdict.name,
            n_violations=n_violations,
        )
        return verdict, metrics

    def process_window_3d(
        self,
        mask: torch.Tensor,
        focus_range: tuple[float, float, float] = (-50.0, 50.0, 10.0),
        dose_range: tuple[float, float, float] = (20.0, 40.0, 5.0),
        n_mc: int = 20,
    ) -> dict[str, Any]:
        """Evaluate the 3D process window with coverage guarantee.

        Sweeps focus and dose, computing defect rates at each operating
        point and flagging points where the conformal coverage condition
        is violated.

        Parameters
        ----------
        mask : Tensor
            ``(H, W)`` binary mask.
        focus_range : tuple
            ``(start_nm, stop_nm, step_nm)`` for focus sweep.
        dose_range : tuple
            ``(start, stop, step)`` in photons/nm^2 for dose sweep.
        n_mc : int
            MC trials per (focus, dose) point.

        Returns
        -------
        dict with:
        - ``defect_grid``: ``(N_focus, N_dose)`` tensor of mean defect rates.
        - ``coverage_grid``: ``(N_focus, N_dose)`` boolean tensor (True =
          coverage satisfied).
        - ``focus_values``: list of focus positions.
        - ``dose_values``: list of dose values.
        """
        m = ensure_2d(mask)
        binary = (m > 0.5).float()

        f_start, f_stop, f_step = focus_range
        d_start, d_stop, d_step = dose_range
        n_focus = max(1, int((f_stop - f_start) / f_step)) + 1
        n_dose = max(1, int((d_stop - d_start) / d_step)) + 1
        focus_values = [f_start + i * f_step for i in range(n_focus)]
        dose_values = [d_start + i * d_step for i in range(n_dose)]

        defect_grid = torch.zeros(n_focus, n_dose)
        coverage_grid = torch.ones(n_focus, n_dose, dtype=torch.bool)

        pixel_size_nm = 1.0
        sigma_px = 2.0
        pixel_area_nm2 = pixel_size_nm**2
        generator = torch.Generator(device=mask.device)
        generator.manual_seed(42)

        for fi, defocus_nm in enumerate(focus_values):
            sigma_eff = sigma_px * (1.0 + (defocus_nm / 100.0) ** 2) ** 0.5
            aerial = simulate_aerial_image(binary, sigma_px=sigma_eff, dose=1.0)
            nominal_resist = apply_resist_threshold(
                aerial,
                threshold=THRESHOLD_ICCAD16,
                pixel_size_nm=pixel_size_nm,
            )

            for di, dose_val in enumerate(dose_values):
                dose_scale = dose_val * pixel_area_nm2
                lambda_map = aerial.clamp(min=0.0) * dose_scale

                defect_sum = 0.0
                for _ in range(n_mc):
                    photons = torch.poisson(lambda_map, generator=generator)
                    noisy_intensity = photons / max(dose_scale, 1e-12)
                    noisy_resist = apply_resist_threshold(
                        noisy_intensity,
                        threshold=THRESHOLD_ICCAD16,
                        pixel_size_nm=pixel_size_nm,
                    )
                    defect_sum += (noisy_resist != nominal_resist).float().mean().item()

                mean_defect = defect_sum / n_mc
                defect_grid[fi, di] = mean_defect

                # Coverage check: defect rate must be below threshold
                # The process window is the region where defect rate < 5%
                coverage_grid[fi, di] = mean_defect < 0.05

        return {
            "defect_grid": defect_grid,
            "coverage_grid": coverage_grid,
            "focus_values": focus_values,
            "dose_values": dose_values,
        }


# ---------------------------------------------------------------------------
# 3D vs 2D benchmark
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult3D:
    """Result from a 3D vs 2D stochastic benchmark comparison."""

    model_3d_defect_rate: float
    model_2d_defect_rate: float
    defect_rate_diff: float
    model_3d_line_collapse_risk: float
    coverage_calibration_3d: float
    coverage_calibration_2d: float
    monotonicity_check: bool


class Stochastic3DBenchmark:
    """Compare 3D vs 2D stochastic predictions on synthetic patterns.

    Runs both the 3D model (this module) and the existing 2D Bayesian
    stochastic model on the same synthetic test masks and reports side-by-side
    metrics: defect rate prediction accuracy, coverage calibration, and line
    collapse risk monotonicity.

    Parameters
    ----------
    se_kernel : SecondaryElectronKernel or None
        Override for the default SE kernel.
    resist : ResistProfile3D or None
        Override for the default 3D resist model.
    dose_photons : float
        Exposure dose.
    sigma_px : float
        PSF sigma.
    """

    def __init__(
        self,
        se_kernel: SecondaryElectronKernel | None = None,
        resist: ResistProfile3D | None = None,
        dose_photons: float = 30.0,
        sigma_px: float = 2.0,
    ) -> None:
        self.se_kernel = se_kernel or SecondaryElectronKernel()
        self.resist = resist or ResistProfile3D(thickness_nm=10.0, pixel_size_nm=1.0)
        self.dose_photons = dose_photons
        self.sigma_px = sigma_px

    def _make_test_masks(self) -> list[tuple[str, torch.Tensor]]:
        """Generate a set of synthetic test masks with varying line/space."""
        masks = []
        size = 32
        for half_cd in [2, 3, 4, 6]:
            m = torch.zeros(size, size)
            pitch = half_cd * 2
            for x in range(0, size, pitch):
                m[:, x : x + half_cd] = 1.0
            masks.append((f"cd_{half_cd * 2}px", m))
        return masks

    def run(self, n_seeds: int = 3) -> list[BenchmarkResult3D]:
        """Run the benchmark comparison.

        Parameters
        ----------
        n_seeds : int
            Number of random seeds to average over.

        Returns
        -------
        List of BenchmarkResult3D, one per test mask.
        """
        from openlithohub.benchmark.metrics.stochastic import compute_stochastic_robustness

        test_masks = self._make_test_masks()
        results = []

        for _label, mask in test_masks:
            defect_rates_3d = []
            defect_rates_2d = []
            collapse_risks = []
            cov_3d_list = []
            cov_2d_list = []

            for seed in range(n_seeds):
                # 3D model
                model_3d = StochasticDefectModel3D(
                    se_kernel=self.se_kernel,
                    resist=self.resist,
                    dose_photons=self.dose_photons,
                    sigma_px=self.sigma_px,
                )
                defect_map_3d = model_3d.simulate_defects(mask, n_mc=30, seed=seed)
                # Average over depth and spatial dims
                defect_rate_3d = defect_map_3d.mean().item()
                defect_rates_3d.append(defect_rate_3d)

                # Line collapse risk
                profile = self.resist.generate_profile(mask, sigma_px=self.sigma_px)
                collapse_risks.append(self.resist.compute_line_collapse_risk(profile))

                # 2D baseline
                robustness = compute_stochastic_robustness(
                    mask,
                    num_trials=30,
                    dose_photons_per_nm2=self.dose_photons,
                    pixel_size_nm=self.resist.pixel_size_nm,
                    seed=seed,
                )
                defect_rate_2d = (
                    robustness["bridge_probability"] + robustness["break_probability"]
                ) / 2.0
                defect_rates_2d.append(defect_rate_2d)

                # Coverage calibration (synthetic: use self-consistency)
                cov_3d_list.append(
                    1.0 - abs(defect_rate_3d - defect_rate_2d) / max(defect_rate_2d, 1e-6)
                )
                cov_2d_list.append(1.0)

            mean_3d = sum(defect_rates_3d) / n_seeds
            mean_2d = sum(defect_rates_2d) / n_seeds
            mean_collapse = sum(collapse_risks) / n_seeds

            # Monotonicity: collapse risk should increase as CD decreases
            # (checked externally by the caller comparing all results)

            results.append(
                BenchmarkResult3D(
                    model_3d_defect_rate=mean_3d,
                    model_2d_defect_rate=mean_2d,
                    defect_rate_diff=mean_3d - mean_2d,
                    model_3d_line_collapse_risk=mean_collapse,
                    coverage_calibration_3d=max(0.0, min(1.0, sum(cov_3d_list) / n_seeds)),
                    coverage_calibration_2d=max(0.0, min(1.0, sum(cov_2d_list) / n_seeds)),
                    monotonicity_check=True,  # Placeholder; validated by test
                )
            )

        # Check monotonicity: collapse risk should decrease as CD grows.
        # _make_test_masks orders patterns from smallest to largest CD, so
        # a physically consistent model yields a non-increasing risk
        # sequence.
        risks = [r.model_3d_line_collapse_risk for r in results]
        monotone = all(risks[i] >= risks[i + 1] for i in range(len(risks) - 1))
        if not monotone:
            # Risks may not be perfectly monotone on small grids; mark as-is
            results = [
                BenchmarkResult3D(
                    r.model_3d_defect_rate,
                    r.model_2d_defect_rate,
                    r.defect_rate_diff,
                    r.model_3d_line_collapse_risk,
                    r.coverage_calibration_3d,
                    r.coverage_calibration_2d,
                    monotonicity_check=monotone,
                )
                for r in results
            ]

        return results


# ---------------------------------------------------------------------------
# Beyond-LER / LCDU stochastic metrics
# ---------------------------------------------------------------------------
#
# Siemens Calibre SPIE 2026 proposes calibration metrics that go beyond
# traditional LER (line-edge roughness) and LCDU (local CD uniformity):
#   - Failure correlation length: spatial scale over which stochastic
#     failures are correlated, indicating clustering of defects.
#   - Defect cluster distribution: histogram of contiguous defect cluster
#     sizes, characterising the tail of large-cluster events.
#   - Stochastic EPE quantile: the edge placement error at a given
#     statistical quantile across an ensemble of stochastic trials.
# ---------------------------------------------------------------------------


class DefectClusterMetrics:
    """Beyond-LER/LCDU stochastic metrics for EUV defect analysis.

    Provides spatial-statistical metrics that characterise stochastic
    defectivity more richly than scalar LER or LCDU:
      - Failure correlation length: autocorrelation decay distance.
      - Defect cluster size distribution: connected-component statistics.
      - Stochastic EPE quantile: edge placement error at a quantile.

    Parameters
    ----------
    pixel_size_nm : float
        Physical pixel pitch in nanometres.
    connectivity : int
        Connectivity for connected-component labelling (4 or 8).
    """

    def __init__(
        self,
        pixel_size_nm: float = 1.0,
        connectivity: int = 4,
    ) -> None:
        if pixel_size_nm <= 0.0:
            raise ValueError(f"pixel_size_nm must be > 0, got {pixel_size_nm}")
        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
        self.pixel_size_nm = pixel_size_nm
        self.connectivity = connectivity

    def failure_correlation_length(self, defect_map: torch.Tensor) -> float:
        """Compute the spatial correlation length of failures.

        Uses the 2D autocorrelation of the binary defect map and measures
        the distance at which the autocorrelation drops to 1/e of its
        peak value.  The correlation length is averaged over x and y
        directions.

        Parameters
        ----------
        defect_map : Tensor
            ``(H, W)`` defect probability map or binary defect map.
            Values are treated as-is (probability or 0/1).

        Returns
        -------
        Correlation length in nm.  Returns 0.0 if the defect map is
        uniform (no spatial structure).
        """
        d = ensure_2d(defect_map).float()
        d = d - d.mean()

        # Autocorrelation via FFT
        d_fft = torch.fft.fft2(d)
        autocorr: torch.Tensor = torch.fft.ifft2(d_fft * torch.conj(d_fft)).real
        autocorr = torch.fft.fftshift(autocorr)

        h, w = autocorr.shape
        center_y, center_x = h // 2, w // 2
        peak = autocorr[center_y, center_x].item()

        if peak < 1e-12:
            return 0.0

        threshold = peak / math.e

        # Measure correlation length along x. When the autocorrelation
        # never drops below 1/e within the half-window the map is strongly
        # (long-range) correlated — report the full half-window rather
        # than 0, which would wrongly suggest "no spatial structure".
        row = autocorr[center_y, :]
        cx_left = 0
        for i in range(center_x, -1, -1):
            if row[i].item() < threshold:
                cx_left = center_x - i
                break
        else:
            cx_left = center_x
        cx_right = 0
        for i in range(center_x, w):
            if row[i].item() < threshold:
                cx_right = i - center_x
                break
        else:
            cx_right = center_x
        corr_x = (cx_left + cx_right) / 2.0

        # Measure correlation length along y
        col = autocorr[:, center_x]
        cy_top = 0
        for i in range(center_y, -1, -1):
            if col[i].item() < threshold:
                cy_top = center_y - i
                break
        else:
            cy_top = center_y
        cy_bottom = 0
        for i in range(center_y, h):
            if col[i].item() < threshold:
                cy_bottom = i - center_y
                break
        else:
            cy_bottom = center_y
        corr_y = (cy_top + cy_bottom) / 2.0

        # Average correlation length in pixels, convert to nm
        corr_px = (corr_x + corr_y) / 2.0
        return corr_px * self.pixel_size_nm

    def defect_cluster_distribution(self, defect_map: torch.Tensor) -> dict[str, Any]:
        """Compute the size distribution of defect clusters.

        Connected-component analysis on the binarised defect map yields
        a histogram of cluster sizes (in pixels).  This is useful for
        detecting whether stochastic defects tend to cluster (indicating
        spatial correlation) or are isolated (Poisson-like).

        Parameters
        ----------
        defect_map : Tensor
            ``(H, W)`` defect probability map.  Thresholded at 0.5 to
            produce a binary defect mask.

        Returns
        -------
        Dict with:
        - ``n_clusters``: total number of defect clusters.
        - ``cluster_sizes``: list of cluster sizes (pixels), sorted
          descending.
        - ``max_cluster_size``: size of the largest cluster.
        - ``mean_cluster_size``: mean cluster size.
        - ``size_histogram``: dict mapping size -> count.
        """
        d = ensure_2d(defect_map).float()
        binary = (d > 0.5).float()

        h, w = binary.shape
        visited = torch.zeros(h, w, dtype=torch.bool)
        cluster_sizes: list[int] = []

        # Direction offsets for connectivity
        if self.connectivity == 4:
            offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        else:
            offsets = [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]

        # Flood-fill connected-component labelling
        for y in range(h):
            for x in range(w):
                if binary[y, x] > 0.5 and not visited[y, x]:
                    size = 0
                    stack = [(y, x)]
                    while stack:
                        cy, cx = stack.pop()
                        if cy < 0 or cy >= h or cx < 0 or cx >= w:
                            continue
                        if visited[cy, cx]:
                            continue
                        if binary[cy, cx] <= 0.5:
                            continue
                        visited[cy, cx] = True
                        size += 1
                        for dy, dx in offsets:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                                stack.append((ny, nx))
                    cluster_sizes.append(size)

        cluster_sizes.sort(reverse=True)

        size_histogram: dict[int, int] = {}
        for s in cluster_sizes:
            size_histogram[s] = size_histogram.get(s, 0) + 1

        return {
            "n_clusters": len(cluster_sizes),
            "cluster_sizes": cluster_sizes,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "mean_cluster_size": (
                sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0
            ),
            "size_histogram": size_histogram,
        }

    def stochastic_epe_quantile(
        self,
        profile_ensemble: torch.Tensor,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Compute stochastic edge placement error at a given quantile.

        Measures the variation of the edge position across an ensemble of
        stochastic trials and returns the alpha-quantile of the EPE
        distribution.  This is a direct measure of worst-case (tail)
        edge placement, complementing the mean LER metric.

        Parameters
        ----------
        profile_ensemble : Tensor
            ``(N, H, W)`` ensemble of N binary resist profiles from
            stochastic trials.  Each trial is a 2D binary resist image.
        alpha : float
            Quantile level (e.g. 0.05 for the 5th percentile, or 0.95
            for the 95th percentile).  Returns the absolute deviation
            from the mean edge position at this quantile.

        Returns
        -------
        Dict with:
        - ``epe_quantile_nm``: the alpha-quantile of absolute EPE in nm.
        - ``epe_mean_nm``: mean absolute EPE in nm.
        - ``epe_std_nm``: standard deviation of absolute EPE in nm.
        - ``n_trials``: number of stochastic trials used.
        """
        if profile_ensemble.ndim != 3:
            raise ValueError(f"profile_ensemble must be 3D (N, H, W), got {profile_ensemble.ndim}D")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        n, h, w = profile_ensemble.shape

        # For each trial, measure the edge position along each column.
        # Edge = transition from 0->1 or 1->0 along rows (vertical edge).
        edge_positions = []
        for trial_idx in range(n):
            trial = profile_ensemble[trial_idx].float()
            # Compute horizontal gradient to find edges
            grad_x = trial[:, 1:] - trial[:, :-1]
            grad_x = functional.pad(grad_x, (0, 1, 0, 0))  # (H, W)

            # Edge pixels are where gradient magnitude is nonzero
            edge_mask = grad_x.abs() > 0.5

            if edge_mask.any():
                # Mean edge x-position (weighted by gradient magnitude)
                weights = grad_x.abs()
                total_weight = weights.sum()
                if total_weight > 1e-8:
                    y_coords = torch.arange(h, dtype=torch.float32).unsqueeze(1).expand(h, w)
                    x_coords = torch.arange(w, dtype=torch.float32).unsqueeze(0).expand(h, w)
                    mean_y = (weights * y_coords).sum() / total_weight
                    mean_x = (weights * x_coords).sum() / total_weight
                    edge_positions.append((mean_y.item(), mean_x.item()))

        if len(edge_positions) < 2:
            return {
                "epe_quantile_nm": 0.0,
                "epe_mean_nm": 0.0,
                "epe_std_nm": 0.0,
                "n_trials": n,
            }

        # Compute per-trial EPE as distance from mean edge position
        mean_pos_y = sum(p[0] for p in edge_positions) / len(edge_positions)
        mean_pos_x = sum(p[1] for p in edge_positions) / len(edge_positions)

        epe_values = []
        for py, px in edge_positions:
            dist = math.sqrt((py - mean_pos_y) ** 2 + (px - mean_pos_x) ** 2)
            epe_values.append(dist)

        epe_tensor = torch.tensor(epe_values)
        quantile_val = torch.quantile(epe_tensor, alpha).item()
        mean_val = epe_tensor.mean().item()
        std_val = epe_tensor.std().item() if len(epe_values) > 1 else 0.0

        return {
            "epe_quantile_nm": quantile_val * self.pixel_size_nm,
            "epe_mean_nm": mean_val * self.pixel_size_nm,
            "epe_std_nm": std_val * self.pixel_size_nm,
            "n_trials": n,
        }


# ---------------------------------------------------------------------------
# Stochastic calibration metrics (combining all beyond-LER metrics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationReport:
    """Aggregated stochastic calibration report."""

    failure_corr_length_nm: float
    n_defect_clusters: int
    max_cluster_size: int
    mean_cluster_size: float
    epe_quantile_nm: float
    epe_mean_nm: float
    epe_std_nm: float
    conformal_coverage: float
    conformal_verdict: str


class StochasticCalibrationMetrics:
    """Combines all beyond-LER stochastic metrics and validates via conformal coverage.

    Provides a single entry point to compute DefectClusterMetrics and
    validate the resulting predictions through the
    :class:`ConformalCoverageGate3D` coverage gate.

    Parameters
    ----------
    cluster_metrics : DefectClusterMetrics
        Instance for spatial defect cluster analysis.
    coverage_gate : ConformalCoverageGate3D
        Instance for conformal coverage validation.
    """

    def __init__(
        self,
        cluster_metrics: DefectClusterMetrics | None = None,
        coverage_gate: ConformalCoverageGate3D | None = None,
    ) -> None:
        self.cluster_metrics = cluster_metrics or DefectClusterMetrics()
        self.coverage_gate = coverage_gate or ConformalCoverageGate3D()

    def evaluate(
        self,
        defect_map: torch.Tensor,
        profile_ensemble: torch.Tensor | None = None,
        epe_alpha: float = 0.05,
        predicted_defects: torch.Tensor | None = None,
        actual_defects: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> CalibrationReport:
        """Run full stochastic calibration evaluation.

        Parameters
        ----------
        defect_map : Tensor
            ``(H, W)`` defect probability map.
        profile_ensemble : Tensor or None
            ``(N, H, W)`` ensemble of binary resist profiles for EPE
            quantile computation.  If None, EPE metrics are set to 0.
        epe_alpha : float
            Quantile for stochastic EPE.
        predicted_defects : Tensor or None
            Predicted defect map for coverage evaluation.
        actual_defects : Tensor or None
            Ground truth defect map for coverage evaluation.
        mask : Tensor or None
            Input mask for coverage evaluation.

        Returns
        -------
        :class:`CalibrationReport` with all metrics.
        """
        corr_length = self.cluster_metrics.failure_correlation_length(defect_map)
        cluster_dist = self.cluster_metrics.defect_cluster_distribution(defect_map)

        epe_result: dict[str, Any] = {
            "epe_quantile_nm": 0.0,
            "epe_mean_nm": 0.0,
            "epe_std_nm": 0.0,
        }
        if profile_ensemble is not None:
            epe_result = self.cluster_metrics.stochastic_epe_quantile(
                profile_ensemble,
                alpha=epe_alpha,
            )

        # Conformal coverage check
        coverage = 0.0
        verdict = "UNCERTAIN"
        if predicted_defects is not None and actual_defects is not None and mask is not None:
            _, metrics = self.coverage_gate.evaluate(
                mask,
                predicted_defects,
                actual_defects,
            )
            coverage = metrics.coverage
            verdict = metrics.verdict

        return CalibrationReport(
            failure_corr_length_nm=corr_length,
            n_defect_clusters=cluster_dist["n_clusters"],
            max_cluster_size=cluster_dist["max_cluster_size"],
            mean_cluster_size=cluster_dist["mean_cluster_size"],
            epe_quantile_nm=epe_result["epe_quantile_nm"],
            epe_mean_nm=epe_result["epe_mean_nm"],
            epe_std_nm=epe_result["epe_std_nm"],
            conformal_coverage=coverage,
            conformal_verdict=verdict,
        )
