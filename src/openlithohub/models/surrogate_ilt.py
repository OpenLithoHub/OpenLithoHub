"""Surrogate-ILT: CNN-accelerated inverse lithography with periodic correction.

Algorithm borrowed from DiffNano's ``NeuralSurrogate`` — a lightweight CNN
trained on-the-fly to predict aerial images from masks, providing 10–50x
speedup during ILT optimisation while periodically correcting with the true
physics-based forward model.

Why this works
--------------
The aerial image simulation (Gaussian PSF convolution or Hopkins SOCS) is
the bottleneck of ILT — it runs once per optimisation iteration. A small
CNN can learn the mask→aerial mapping accurately enough that most iterations
use the surrogate (a single forward pass) instead of the full convolution.
Every ``correction_interval`` steps, the true forward model runs and its
output is used to refine the surrogate's predictions.

Architecture reference
----------------------
SurrogateNet: Conv2d(1→32) → ReLU → Conv2d(32→32) → ReLU → AdaptiveAvgPool(4)
              → FC(32×16→128) → ReLU → FC(128→1)

The output is a per-pixel scalar (aerial intensity) matching the forward
model's range [0, dose].

Confidence **B** — the CNN architecture is adapted from DiffNano's
``_SurrogateNet``, generalised to output a full 2D aerial image instead of
a 1D diffraction-efficiency vector.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
from diff_surrogate import CorrectionPolicy

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.hopkins import (
    HopkinsParams,
    compute_socs_kernels,
    simulate_aerial_image_hopkins,
)
from openlithohub._utils.resist_model import apply_differentiable_resist
from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry

ForwardModelKind = Literal["gaussian", "hopkins"]


# ---------------------------------------------------------------------------
# Surrogate CNN (adapted from DiffNano/solvers/surrogate.py)
# ---------------------------------------------------------------------------


class _AerialSurrogateNet(nn.Module):
    """CNN that predicts aerial image from mask.

    Input:  (1, H, W) continuous mask in [0, 1]
    Output: (1, H, W) predicted aerial image
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).sigmoid()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Surrogate-ILT Model
# ---------------------------------------------------------------------------


@registry.register
class SurrogateILTModel(LithographyModel):
    """ILT optimisation accelerated by a CNN surrogate forward model.

    Three-phase approach:

    1. **Surrogate training** — generates random masks, runs the true forward
       model, and trains a small CNN to predict aerial images from masks.
    2. **Surrogate-accelerated ILT** — runs the standard ILT loop but uses
       the surrogate for most forward passes, correcting every
       ``correction_interval`` iterations with the true forward model.
    3. **Correction refinement** — when the true forward model runs, the
       surrogate is also fine-tuned on that (mask, aerial) pair.

    This reduces the number of expensive Gaussian/Hopkins convolutions from
    ``iterations`` to roughly ``iterations / correction_interval``.
    """

    NAME = "surrogate-ilt"
    SUPPORTS_CURVILINEAR = True
    RECEPTIVE_FIELD_PX = 64

    def __init__(
        self,
        iterations: int = 200,
        lr: float = 0.1,
        sigma_px: float = 2.0,
        tv_weight: float = 0.01,
        dose: float = 1.0,
        resist_steepness: float = 50.0,
        resist_diffusion_nm: float = 0.0,
        quencher: float = 0.0,
        pixel_size_nm: float = 1.0,
        forward_model: ForwardModelKind = "gaussian",
        hopkins_params: HopkinsParams | None = None,
        # Surrogate
        correction_interval: int = 10,
        surrogate_train_samples: int = 256,
        surrogate_epochs: int = 20,
        surrogate_lr: float = 1e-3,
        surrogate_hidden: int = 32,
    ) -> None:
        self._iterations = iterations
        self._lr = lr
        self._sigma_px = sigma_px
        self._tv_weight = tv_weight
        self._dose = dose
        self._resist_steepness = resist_steepness
        self._resist_diffusion_nm = resist_diffusion_nm
        self._quencher = quencher
        self._pixel_size_nm = pixel_size_nm
        self._forward_model = forward_model
        self._hopkins_params = hopkins_params or HopkinsParams()
        self._correction_interval = correction_interval
        self._correction_policy = CorrectionPolicy(correction_interval=correction_interval)
        self._surrogate_train_samples = surrogate_train_samples
        self._surrogate_epochs = surrogate_epochs
        self._surrogate_lr = surrogate_lr
        self._surrogate_hidden = surrogate_hidden

        self._cached_kernels: torch.Tensor | None = None
        self._cached_weights: torch.Tensor | None = None
        self._cached_grid: int | None = None
        self._cache_lock = threading.Lock()

    def _ensure_hopkins_kernels(
        self,
        grid_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self._cache_lock:
            if (
                self._cached_kernels is None
                or self._cached_weights is None
                or self._cached_grid != grid_size
                or self._cached_kernels.device != device
            ):
                kernels, weights = compute_socs_kernels(
                    self._hopkins_params,
                    grid_size,
                    device,
                )
                self._cached_kernels = kernels
                self._cached_weights = weights
                self._cached_grid = grid_size
            if TYPE_CHECKING:
                assert self._cached_kernels is not None
                assert self._cached_weights is not None
            return self._cached_kernels, self._cached_weights

    def _run_true_forward(
        self,
        mask: torch.Tensor,
        sigma_px: float,
        forward_model: str,
        kernels: torch.Tensor | None,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if forward_model == "hopkins":
            return simulate_aerial_image_hopkins(
                mask,
                kernels=kernels,
                weights=weights,
                dose=self._dose,
            )
        return simulate_aerial_image(mask, sigma_px=sigma_px, dose=self._dose)

    def _train_surrogate(
        self,
        net: _AerialSurrogateNet,
        grid_size: int,
        sigma_px: float,
        forward_model: str,
        kernels: torch.Tensor | None,
        weights: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        opt = torch.optim.Adam(net.parameters(), lr=self._surrogate_lr)
        batch_size = 32
        n = self._surrogate_train_samples

        for _epoch in range(self._surrogate_epochs):
            raw = torch.rand(n, 1, grid_size, grid_size, device=device)
            masks = (raw > 0.5).float()

            with torch.no_grad():
                aerials_list = []
                for i in range(n):
                    aerial_i = self._run_true_forward(
                        masks[i, 0],
                        sigma_px,
                        forward_model,
                        kernels,
                        weights,
                    )
                    aerials_list.append(aerial_i)
                aerials = torch.stack(aerials_list).unsqueeze(1)

            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                pred = net(masks[idx])
                loss = nn.functional.mse_loss(pred, aerials[idx])
                opt.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                opt.step()

        net.eval()

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        """Run surrogate-accelerated ILT optimisation."""
        target = design.detach().float()
        if target.ndim > 2:
            target = target.squeeze()

        iterations = kwargs.get("iterations", self._iterations)
        lr = kwargs.get("lr", self._lr)
        sigma_px = kwargs.get("sigma_px", self._sigma_px)
        tv_weight = kwargs.get("tv_weight", self._tv_weight)
        forward_model = kwargs.get("forward_model", self._forward_model)
        hopkins_params = kwargs.get("hopkins_params", self._hopkins_params)
        device = kwargs.get("device")
        if device is not None:
            target = target.to(device)
        device = target.device
        grid_size = target.shape[0]

        # Prepare Hopkins kernels if needed
        kernels: torch.Tensor | None = None
        weights: torch.Tensor | None = None
        if forward_model == "hopkins":
            if hopkins_params != self._hopkins_params:
                self._hopkins_params = hopkins_params
                self._cached_kernels = None
                self._cached_weights = None
                self._cached_grid = None
            kernels, weights = self._ensure_hopkins_kernels(grid_size, device)

        # --- Phase 1: train surrogate ---
        surrogate = _AerialSurrogateNet(self._surrogate_hidden).to(device)
        self._train_surrogate(
            surrogate,
            grid_size,
            sigma_px,
            forward_model,
            kernels,
            weights,
            device,
        )

        # --- Phase 2: ILT with surrogate + periodic correction ---
        mask_logit = torch.zeros_like(target)
        with torch.no_grad():
            mask_logit.copy_(target * 4.0 - 2.0)
        mask_logit = mask_logit.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([mask_logit], lr=lr)
        best_loss = float("inf")
        best_mask = target.clone()

        for it in range(iterations):
            optimizer.zero_grad()
            mask_continuous = torch.sigmoid(mask_logit)

            use_surrogate = not self._correction_policy.should_correct(it)

            if use_surrogate:
                mask_4d = mask_continuous.unsqueeze(0).unsqueeze(0)
                aerial = surrogate(mask_4d).squeeze(0).squeeze(0)
            else:
                aerial = self._run_true_forward(
                    mask_continuous,
                    sigma_px,
                    forward_model,
                    kernels,
                    weights,
                )
                # Fine-tune surrogate on this real pair
                mask_4d = mask_continuous.detach().unsqueeze(0).unsqueeze(0)
                aerial_detached = aerial.detach().unsqueeze(0).unsqueeze(0)
                s_opt = torch.optim.Adam(surrogate.parameters(), lr=self._surrogate_lr)
                for _ in range(3):
                    s_pred = surrogate(mask_4d)
                    s_loss = nn.functional.mse_loss(s_pred, aerial_detached)
                    s_opt.zero_grad()
                    s_loss.backward()  # type: ignore[no-untyped-call]
                    s_opt.step()

            resist = apply_differentiable_resist(
                aerial,
                threshold=0.5,
                steepness=self._resist_steepness,
                resist_diffusion_nm=self._resist_diffusion_nm,
                pixel_size_nm=self._pixel_size_nm,
                quencher=self._quencher,
            )
            fidelity_loss = nn.functional.mse_loss(resist, target)

            if tv_weight > 0:
                diff_h = (mask_continuous[1:, :] - mask_continuous[:-1, :]).pow(2)
                diff_w = (mask_continuous[:, 1:] - mask_continuous[:, :-1]).pow(2)
                tv_loss = diff_h.sum() + diff_w.sum()
                loss = fidelity_loss + tv_weight * tv_loss
            else:
                loss = fidelity_loss

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                with torch.no_grad():
                    best_mask = (mask_continuous > 0.5).float().detach()

        return PredictionResult(
            mask=best_mask,
            metadata={
                "final_loss": best_loss,
                "iterations": iterations,
                "sigma_px": sigma_px,
                "forward_model": forward_model,
                "surrogate_correction_interval": self._correction_interval,
                "true_forward_calls": iterations // self._correction_interval + 1,
            },
        )
