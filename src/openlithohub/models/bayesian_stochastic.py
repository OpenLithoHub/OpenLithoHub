"""EUV stochastic defect prediction with Bayesian deep learning.

Implements per-pixel failure-probability, LER (line-edge roughness), and LWR
(line-width roughness) heatmaps from a mask or aerial image input.

Two inference modes:

1. **Poisson-MC** (default, no training required): directly simulates EUV
   photon shot noise through the forward model, runs K Poisson trials, and
   aggregates into per-pixel statistics. Physics-grounded baseline that needs
   no learned weights.

2. **MC-Dropout**: a small U-Net with dropout layers, trained on synthetic
   Poisson-MC ground truth. At inference, K stochastic forward passes with
   dropout active produce per-pixel mean + variance, which are converted to
   failure probability / LER / LWR maps.

References:
    - De Bisschop, P. (2017–2024), "Stochastic effects in EUV lithography"
      (SPIE series).
    - Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation".
    - Lakshminarayanan et al. (2017), "Simple and Scalable Predictive
      Uncertainty Estimation using Deep Ensembles".
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from openlithohub._constants import THRESHOLD_ICCAD16
from openlithohub._utils.forward_model import apply_resist_threshold, simulate_aerial_image
from openlithohub._utils.tensor_ops import ensure_2d
from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry

__all__ = [
    "BayesianStochasticModel",
    "StochasticUNet",
    "generate_synthetic_ground_truth",
]

InferenceMode = Literal["poisson", "mc_dropout"]


# ---------------------------------------------------------------------------
# Small U-Net with dropout for MC-Dropout inference
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class StochasticUNet(nn.Module):
    """Lightweight U-Net with dropout for MC-Dropout stochastic prediction.

    Input: aerial image ``(B, 1, H, W)``.
    Output: dict with ``failure_logit``, ``ler_log``, ``lwr_log`` each
    ``(B, 1, H, W)``.  The caller applies sigmoid / exp to get physical units.
    """

    def __init__(self, base_channels: int = 32, dropout_p: float = 0.1) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = _ConvBlock(1, c, dropout_p)
        self.enc2 = _ConvBlock(c, c * 2, dropout_p)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(c * 2, c * 4, dropout_p)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _ConvBlock(c * 4 + c * 2, c * 2, dropout_p)
        self.dec1 = _ConvBlock(c * 2 + c, c, dropout_p)
        self.head_failure = nn.Conv2d(c, 1, 1)
        self.head_ler = nn.Conv2d(c, 1, 1)
        self.head_lwr = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return {
            "failure_logit": self.head_failure(d1),
            "ler_log": self.head_ler(d1),
            "lwr_log": self.head_lwr(d1),
        }


# ---------------------------------------------------------------------------
# Synthetic ground-truth generation
# ---------------------------------------------------------------------------


def generate_synthetic_ground_truth(
    mask: torch.Tensor,
    n_mc: int = 500,
    dose_photons_per_nm2: float = 30.0,
    pixel_size_nm: float = 1.0,
    sigma_px: float = 2.0,
    resist_threshold: float = THRESHOLD_ICCAD16,
    resist_diffusion_nm: float = 0.0,
    quencher: float = 0.0,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Generate synthetic stochastic ground truth via Poisson MC simulation.

    Returns
    -------
    dict with:
    - ``failure_prob``: ``(H, W)`` per-pixel failure probability in [0, 1]
    - ``ler_nm``: ``(H, W)`` per-pixel LER σ in nm (non-negative)
    - ``lwr_nm``: ``(H, W)`` per-pixel LWR σ in nm (non-negative)
    - ``aerial``: ``(H, W)`` deterministic aerial image
    """
    m = ensure_2d(mask)
    binary = (m > 0.5).float()
    aerial = simulate_aerial_image(binary, sigma_px=sigma_px, dose=1.0)
    resist_nominal = apply_resist_threshold(
        aerial,
        threshold=resist_threshold,
        resist_diffusion_nm=resist_diffusion_nm,
        pixel_size_nm=pixel_size_nm,
        quencher=quencher,
    )
    pixel_area_nm2 = pixel_size_nm * pixel_size_nm
    dose_scale = dose_photons_per_nm2 * pixel_area_nm2
    lambda_map = aerial.clamp(min=0.0) * dose_scale

    generator = torch.Generator(device=mask.device)
    generator.manual_seed(seed)

    resist_samples = []
    for _ in range(n_mc):
        photons = torch.poisson(lambda_map, generator=generator)
        noisy_intensity = photons / max(dose_scale, 1e-12)
        noisy_resist = apply_resist_threshold(
            noisy_intensity,
            threshold=resist_threshold,
            resist_diffusion_nm=resist_diffusion_nm,
            pixel_size_nm=pixel_size_nm,
            quencher=quencher,
        )
        resist_samples.append(noisy_resist)

    resist_stack = torch.stack(resist_samples)

    # Per-pixel failure probability: fraction of MC samples where the resist
    # state differs from the nominal (i.e. a stochastic "defect" at that pixel).
    nominal_binary = (resist_nominal > resist_threshold).float()
    sample_binaries = (resist_stack > resist_threshold).float()
    failure_prob = (sample_binaries != nominal_binary.unsqueeze(0)).float().mean(dim=0)

    # Per-pixel LER: standard deviation of the resist output across MC
    # samples, scaled to nm. At edge pixels this captures edge-position
    # uncertainty; in bulk it is near zero.
    resist_std = resist_stack.std(dim=0)
    ler_nm = resist_std * pixel_size_nm

    # Per-pixel LWR: approximate as 2× LER (conservative; exact LWR requires
    # paired edge displacement which is only defined along line features).
    lwr_nm = 2.0 * ler_nm

    return {
        "failure_prob": failure_prob,
        "ler_nm": ler_nm,
        "lwr_nm": lwr_nm,
        "aerial": aerial,
    }


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


@registry.register
class BayesianStochasticModel(LithographyModel):
    """EUV stochastic defect prediction with per-pixel uncertainty heatmaps.

    Parameters
    ----------
    n_mc_samples : int
        Number of Monte Carlo samples for inference.
    mode : str
        ``"poisson"`` (physics MC, no training) or ``"mc_dropout"``
        (learned U-Net with dropout).
    dose_photons_per_nm2 : float
        EUV exposure dose in photons/nm² at the wafer.
    pixel_size_nm : float
        Mask pixel size in nm.
    sigma_px : float
        Gaussian PSF σ in pixels for the aerial image forward model.
    resist_threshold : float
        Resist development threshold.
    resist_diffusion_nm : float
        Acid diffusion length in nm.
    quencher : float
        Quencher concentration.
    seed : int or None
        RNG seed for reproducibility. None → system entropy.
    base_channels : int
        U-Net base channel count (mc_dropout mode only).
    dropout_p : float
        Dropout probability (mc_dropout mode only).
    """

    NAME = "bayesian-stochastic"
    SUPPORTS_CURVILINEAR = True
    RECEPTIVE_FIELD_PX = 32

    def __init__(
        self,
        n_mc_samples: int = 32,
        mode: InferenceMode = "poisson",
        dose_photons_per_nm2: float = 30.0,
        pixel_size_nm: float = 1.0,
        sigma_px: float = 2.0,
        resist_threshold: float = THRESHOLD_ICCAD16,
        resist_diffusion_nm: float = 0.0,
        quencher: float = 0.0,
        seed: int | None = 0,
        base_channels: int = 32,
        dropout_p: float = 0.1,
    ) -> None:
        self.n_mc_samples = n_mc_samples
        self.mode = mode
        self.dose_photons_per_nm2 = dose_photons_per_nm2
        self.pixel_size_nm = pixel_size_nm
        self.sigma_px = sigma_px
        self.resist_threshold = resist_threshold
        self.resist_diffusion_nm = resist_diffusion_nm
        self.quencher = quencher
        self.seed = seed
        self._unet: StochasticUNet | None = None
        self._base_channels = base_channels
        self._dropout_p = dropout_p

    @property
    def unet(self) -> StochasticUNet:
        if self._unet is None:
            self._unet = StochasticUNet(self._base_channels, self._dropout_p)
        return self._unet

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        """Predict stochastic defect heatmaps from a mask or aerial image.

        Parameters
        ----------
        design : Tensor, shape ``(H, W)`` or ``(B, 1, H, W)``
            Binary or continuous mask pattern.

        Returns
        -------
        PredictionResult
            ``mask`` = mean printed pattern.
            ``metadata`` contains ``failure_prob``, ``ler_nm``, ``lwr_nm``
            each ``(H, W)``.
        """
        m = ensure_2d(design)
        binary = (m > 0.5).float()
        aerial = simulate_aerial_image(binary, sigma_px=self.sigma_px, dose=1.0)

        if self.mode == "mc_dropout" and self._unet is not None:
            failure_prob, ler_nm, lwr_nm, mean_resist = self._predict_mc_dropout(aerial)
        else:
            failure_prob, ler_nm, lwr_nm, mean_resist = self._predict_poisson(aerial)

        return PredictionResult(
            mask=mean_resist,
            metadata={
                "failure_prob": failure_prob,
                "ler_nm": ler_nm,
                "lwr_nm": lwr_nm,
                "mode": self.mode,
                "n_mc_samples": self.n_mc_samples,
                "dose_photons_per_nm2": self.dose_photons_per_nm2,
            },
        )

    def _predict_poisson(
        self, aerial: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Poisson-MC inference: run K photon-noise trials directly."""
        resist_nominal = apply_resist_threshold(
            aerial,
            threshold=self.resist_threshold,
            resist_diffusion_nm=self.resist_diffusion_nm,
            pixel_size_nm=self.pixel_size_nm,
            quencher=self.quencher,
        )
        pixel_area_nm2 = self.pixel_size_nm * self.pixel_size_nm
        dose_scale = self.dose_photons_per_nm2 * pixel_area_nm2
        lambda_map = aerial.clamp(min=0.0) * dose_scale

        generator = torch.Generator(device=aerial.device)
        if self.seed is not None:
            generator.manual_seed(self.seed)

        resist_samples = []
        for _ in range(self.n_mc_samples):
            photons = torch.poisson(lambda_map, generator=generator)
            noisy_intensity = photons / max(dose_scale, 1e-12)
            noisy_resist = apply_resist_threshold(
                noisy_intensity,
                threshold=self.resist_threshold,
                resist_diffusion_nm=self.resist_diffusion_nm,
                pixel_size_nm=self.pixel_size_nm,
                quencher=self.quencher,
            )
            resist_samples.append(noisy_resist)

        resist_stack = torch.stack(resist_samples)
        mean_resist = resist_stack.mean(dim=0)

        nominal_binary = (resist_nominal > self.resist_threshold).float()
        sample_binaries = (resist_stack > self.resist_threshold).float()
        failure_prob = (sample_binaries != nominal_binary.unsqueeze(0)).float().mean(dim=0)

        resist_std = resist_stack.std(dim=0)
        ler_nm = resist_std * self.pixel_size_nm
        lwr_nm = 2.0 * ler_nm

        return failure_prob, ler_nm, lwr_nm, mean_resist

    def _predict_mc_dropout(
        self, aerial: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """MC-Dropout inference: K forward passes with dropout active."""
        net = self.unet
        net.train()

        inp = aerial.unsqueeze(0).unsqueeze(0)
        failure_logits = []
        ler_logs = []
        lwr_logs = []

        for _ in range(self.n_mc_samples):
            out = net(inp)
            failure_logits.append(out["failure_logit"].squeeze(0).squeeze(0))
            ler_logs.append(out["ler_log"].squeeze(0).squeeze(0))
            lwr_logs.append(out["lwr_log"].squeeze(0).squeeze(0))

        failure_logits_t = torch.stack(failure_logits)
        ler_logs_t = torch.stack(ler_logs)
        lwr_logs_t = torch.stack(lwr_logs)

        failure_prob = torch.sigmoid(failure_logits_t).mean(dim=0)
        ler_nm = torch.exp(ler_logs_t).mean(dim=0) * self.pixel_size_nm
        lwr_nm = torch.exp(lwr_logs_t).mean(dim=0) * self.pixel_size_nm

        mean_resist = apply_resist_threshold(
            aerial,
            threshold=self.resist_threshold,
            resist_diffusion_nm=self.resist_diffusion_nm,
            pixel_size_nm=self.pixel_size_nm,
            quencher=self.quencher,
        )

        return failure_prob, ler_nm, lwr_nm, mean_resist

    def calibration_curve(
        self,
        design: torch.Tensor,
        n_bins: int = 10,
        n_ground_truth_mc: int = 500,
    ) -> dict[str, list[float]]:
        """Compute predicted vs observed failure-probability calibration.

        Splits predicted failure probabilities into quantile bins and
        compares against ground truth from a large Poisson MC run.

        Returns
        -------
        dict with ``predicted`` and ``observed`` lists of per-bin means.
        """
        gt = generate_synthetic_ground_truth(
            design,
            n_mc=n_ground_truth_mc,
            dose_photons_per_nm2=self.dose_photons_per_nm2,
            pixel_size_nm=self.pixel_size_nm,
            sigma_px=self.sigma_px,
            resist_threshold=self.resist_threshold,
            resist_diffusion_nm=self.resist_diffusion_nm,
            quencher=self.quencher,
        )
        result = self.predict(design)
        pred = result.metadata["failure_prob"]
        obs = gt["failure_prob"]

        pred_flat = pred.flatten()
        obs_flat = obs.flatten()
        boundaries = torch.quantile(pred_flat, torch.linspace(0, 1, n_bins + 1))
        predicted_means: list[float] = []
        observed_means: list[float] = []
        for i in range(n_bins):
            lo = boundaries[i]
            hi = boundaries[i + 1]
            if i == n_bins - 1:
                mask_bin = (pred_flat >= lo) & (pred_flat <= hi)
            else:
                mask_bin = (pred_flat >= lo) & (pred_flat < hi)
            if mask_bin.any():
                predicted_means.append(pred_flat[mask_bin].mean().item())
                observed_means.append(obs_flat[mask_bin].mean().item())
            else:
                predicted_means.append(float(lo))
                observed_means.append(0.0)
        return {"predicted": predicted_means, "observed": observed_means}

    def to_torch_module(self) -> nn.Module:
        if self.mode == "mc_dropout":
            return self.unet
        return nn.Module()
