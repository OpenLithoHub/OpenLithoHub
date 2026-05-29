"""VAE-ILT: Latent-space inverse lithography via variational autoencoder.

Algorithm borrowed from DiffNano's ``LearnedRepresentation`` (VAE design-space
learning) and adapted to computational lithography. Instead of optimizing
directly in high-dimensional pixel space (H×W logit mask), the optimiser
works in a compressed latent vector whose dimensionality is orders of
magnitude smaller. This yields:

* **Smoother loss landscape** — the decoder's conv-deconv architecture acts
  as a learned regulariser that naturally favours mask-like outputs.
* **Faster convergence** — fewer optimisation variables means fewer steps.
* **Implicit MRC compliance** — the decoder was trained on binary-ish masks,
  so its outputs tend to respect minimum feature-size constraints.

Architecture reference
----------------------
Encoder: Conv2d(stride=2) → Conv2d(stride=2) → AdaptiveAvgPool → FC(μ, logvar)
Decoder: FC → reshape → ConvTranspose2d ×2 → bilinear upsample → sigmoid

Training is self-supervised: random binary masks are generated on-the-fly,
so no external dataset is required.

Confidence **B** — the VAE architecture is a direct port of DiffNano's
``_Encoder`` / ``_Decoder`` with ``float32`` replacing ``float64`` to match
the rest of OpenLithoHub.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.hopkins import (
    HopkinsParams,
    compute_socs_kernels,
    simulate_aerial_image_hopkins,
)
from openlithohub._utils.resist_model import differentiable_threshold
from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry

ForwardModelKind = Literal["gaussian", "hopkins"]


# ---------------------------------------------------------------------------
# VAE components (adapted from DiffNano/design/representation_learning.py)
# ---------------------------------------------------------------------------


class _Encoder(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int = 16, out_size: int = 64, hidden: int = 32):
        super().__init__()
        self.out_size = out_size
        self.hidden = hidden
        self.fc = nn.Linear(latent_dim, hidden * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, 1, 3, stride=2, padding=1, output_padding=1),
        )
        self.final_upsample = nn.Upsample(
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(-1, self.hidden, 4, 4)
        h = self.deconv(h)
        h = self.final_upsample(h)
        return torch.sigmoid(h)


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# ---------------------------------------------------------------------------
# VAE-ILT Model
# ---------------------------------------------------------------------------


@registry.register
class VAEILTModel(LithographyModel):
    """Inverse Lithography Technology via VAE latent-space optimisation.

    Two-phase approach:

    1. **VAE pre-training** — trains a convolutional VAE on random binary
       masks so the decoder learns a compact, mask-like manifold.
    2. **Latent optimisation** — freezes the decoder and optimises a latent
       vector ``z`` to minimise the lithography fidelity loss (aerial image
       vs. target), optionally with TV regularisation.

    Because the latent dimension is typically 16–64 (vs. H×W ≈ 4096 pixels),
    convergence is faster and the decoder's inductive bias keeps the output
    structurally similar to valid mask patterns.
    """

    NAME = "vae-ilt"
    SUPPORTS_CURVILINEAR = True
    RECEPTIVE_FIELD_PX = 64

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_channels: int = 32,
        iterations: int = 200,
        lr: float = 0.05,
        sigma_px: float = 2.0,
        tv_weight: float = 0.01,
        dose: float = 1.0,
        resist_steepness: float = 50.0,
        forward_model: ForwardModelKind = "gaussian",
        hopkins_params: HopkinsParams | None = None,
        # VAE training
        vae_train_masks: int = 512,
        vae_epochs: int = 30,
        vae_lr: float = 1e-3,
    ) -> None:
        self._latent_dim = latent_dim
        self._hidden = hidden_channels
        self._iterations = iterations
        self._lr = lr
        self._sigma_px = sigma_px
        self._tv_weight = tv_weight
        self._dose = dose
        self._resist_steepness = resist_steepness
        self._forward_model = forward_model
        self._hopkins_params = hopkins_params or HopkinsParams()
        self._vae_train_masks = vae_train_masks
        self._vae_epochs = vae_epochs
        self._vae_lr = vae_lr

        self._cached_kernels: torch.Tensor | None = None
        self._cached_weights: torch.Tensor | None = None
        self._cached_grid: int | None = None

    def _build_vae(self, grid_size: int, device: torch.device) -> tuple[_Encoder, _Decoder]:
        encoder = _Encoder(self._latent_dim, self._hidden).to(device)
        decoder = _Decoder(self._latent_dim, grid_size, self._hidden).to(device)
        return encoder, decoder

    def _train_vae(
        self,
        encoder: _Encoder,
        decoder: _Decoder,
        grid_size: int,
        device: torch.device,
    ) -> None:
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=self._vae_lr,
        )
        for _ in range(self._vae_epochs):
            # Generate random binary-ish masks on the fly
            raw = torch.rand(self._vae_train_masks, 1, grid_size, grid_size, device=device)
            masks = (raw > 0.5).float()

            mu, logvar = encoder(masks)
            z = _reparameterize(mu, logvar)
            recon = decoder(z)

            recon_loss = nn.functional.mse_loss(recon, masks, reduction="sum")
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
            loss = recon_loss + kl_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

        encoder.eval()
        decoder.eval()

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        """Run VAE-ILT optimisation.

        Phase 1: trains a VAE on random masks of the same spatial size.
        Phase 2: encodes the target, then optimises the latent vector under
        the lithography fidelity loss.
        """
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

        # --- Phase 1: train VAE ---
        encoder, decoder = self._build_vae(grid_size, device)
        self._train_vae(encoder, decoder, grid_size, device)

        # --- Phase 2: latent optimisation ---
        kernels: torch.Tensor | None = None
        weights: torch.Tensor | None = None
        if forward_model == "hopkins":
            kernels, weights = compute_socs_kernels(hopkins_params, grid_size, device)

        # Encode target as starting latent vector
        with torch.no_grad():
            target_4d = target.unsqueeze(0).unsqueeze(0)
            mu, _ = encoder(target_4d)
            z_init = mu.squeeze(0).clone()

        z = z_init.detach().requires_grad_(True)
        # Freeze decoder
        for p in decoder.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.Adam([z], lr=lr)
        best_loss = float("inf")
        best_mask = target.clone()

        for _it in range(iterations):
            optimizer.zero_grad()

            mask_continuous = decoder(z.unsqueeze(0)).squeeze(0).squeeze(0)

            if forward_model == "hopkins":
                aerial = simulate_aerial_image_hopkins(
                    mask_continuous,
                    kernels=kernels,
                    weights=weights,
                    dose=self._dose,
                )
            else:
                aerial = simulate_aerial_image(mask_continuous, sigma_px=sigma_px, dose=self._dose)

            resist = differentiable_threshold(
                aerial,
                threshold=0.5,
                steepness=self._resist_steepness,
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
                "latent_dim": self._latent_dim,
                "forward_model": forward_model,
            },
        )
