"""Posterior warm-start: diverse mask family from a learned conditional VAE.

Unlike GANOPCWarmStart (noise injection around a single network forward pass),
this module learns a conditional distribution p(mask | target) using a
conditional VAE. During training the encoder maps (target, mask) pairs to a
Gaussian latent; during inference the decoder samples diverse masks from
p(mask | target) by drawing z ~ N(0, I) and decoding (z, target).

Key differences from ``warm_start.py`` providers:
- GANOPCWarmStart/NeuralILTWarmStart inject *additive noise on logits* —
  limited diversity, same deterministic backbone.
- PosteriorWarmStart samples from a *learned posterior* — much richer
  candidate diversity because each sample traverses a different latent code.

Also provides ``BatchILTScheduler`` that takes multiple candidate masks,
refines each independently via a user-supplied ILT refiner, scores them
with ``CandidateScorer``, and returns the top-k.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from openlithohub.models.warm_start import CandidateScorer

# ---------------------------------------------------------------------------
# Conditional VAE components
# ---------------------------------------------------------------------------


class _Encoder(nn.Module):
    """Encode (target, mask) → (mu, logvar) in latent space.

    Takes a 2-channel input (target || mask concatenation), downsamples
    twice with stride-2 convolutions, then pools to a vector and projects
    to (mu, logvar).
    """

    def __init__(self, latent_dim: int = 32, base_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_mu = nn.Linear(base_channels, latent_dim)
        self.fc_logvar = nn.Linear(base_channels, latent_dim)

    def forward(
        self, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([target, mask], dim=1)
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    """Decode (z, target) → mask.

    The latent code z is spatially broadcast and concatenated with the target,
    then upsampled back to the original resolution.
    """

    def __init__(self, latent_dim: int = 32, base_channels: int = 32) -> None:
        super().__init__()
        self.base_channels = base_channels
        # z is broadcast to spatial dims and concatenated with target (1 ch)
        self.fc = nn.Linear(latent_dim, base_channels * 4 * 4)
        self.up1 = nn.ConvTranspose2d(
            base_channels + 1, base_channels, 3, stride=2, padding=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.up2 = nn.ConvTranspose2d(
            base_channels, base_channels, 3, stride=2, padding=1, output_padding=1
        )
        self.bn2 = nn.BatchNorm2d(base_channels)
        self.outc = nn.Conv2d(base_channels, 1, 1)

    def forward(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        height, width = target.shape[2], target.shape[3]

        h = self.fc(z).reshape(batch, self.base_channels, 4, 4)
        # Upsample to match target spatial dims
        h = nn.functional.interpolate(h, size=(height // 2, width // 2), mode="nearest")
        # Concatenate target as conditioning
        target_half = nn.functional.interpolate(
            target, size=(height // 2, width // 2), mode="nearest"
        )
        h = torch.cat([h, target_half], dim=1)
        h = torch.relu(self.bn1(self.up1(h)))
        # Upsample again — may exceed target size due to stride rounding
        h = nn.functional.interpolate(h, size=(height, width), mode="nearest")
        h = torch.relu(self.bn2(self.up2(h)))
        # Final resize to exact target size
        h = nn.functional.interpolate(h, size=(height, width), mode="nearest")
        logits = self.outc(h)
        return torch.sigmoid(logits)


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# ---------------------------------------------------------------------------
# PosteriorWarmStart
# ---------------------------------------------------------------------------


class PosteriorWarmStart(nn.Module):
    """Posterior warm-start: generate diverse mask family from learned distribution.

    Unlike GANOPCWarmStart (noise injection), this learns a conditional
    distribution p(mask | target) and samples diverse candidates from the
    posterior.

    Architecture: conditional VAE that encodes (target, mask) pairs during
    training and samples diverse masks from p(mask | target) during inference.
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 32, base_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.encoder = _Encoder(latent_dim=latent_dim, base_channels=base_channels)
        self.decoder = _Decoder(latent_dim=latent_dim, base_channels=base_channels)

        # Xavier init for stable training
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (target, mask) pair to latent distribution parameters."""
        t4 = self._ensure_4d(target)
        m4 = self._ensure_4d(mask)
        result: tuple[torch.Tensor, torch.Tensor] = self.encoder(t4, m4)
        return result

    def decode(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Decode latent + target -> mask."""
        t4 = self._ensure_4d(target)
        batch_size = z.shape[0]
        if t4.shape[0] == 1 and batch_size > 1:
            t4 = t4.expand(batch_size, -1, -1, -1)
        mask_4d: torch.Tensor = self.decoder(z, t4)
        return mask_4d.squeeze(1)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """Single sample from posterior (for WarmStartProvider protocol)."""
        t4 = self._ensure_4d(target)
        batch_size = t4.shape[0]
        z = torch.randn(batch_size, self.latent_dim, device=t4.device, dtype=t4.dtype)
        mask_4d = self.decoder(z, t4)
        mask = mask_4d.squeeze(0).squeeze(0)
        return self._match_ndim(mask, target)

    def generate_initial_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Single deterministic mask (mean of prior, z=0)."""
        t4 = self._ensure_4d(target)
        batch_size = t4.shape[0]
        z = torch.zeros(batch_size, self.latent_dim, device=t4.device, dtype=t4.dtype)
        mask_4d = self.decoder(z, t4)
        mask = mask_4d.squeeze(0).squeeze(0)
        return self._match_ndim(mask, target)

    def generate_candidates(
        self, target: torch.Tensor, n_candidates: int = 5
    ) -> list[torch.Tensor]:
        """Sample n_candidates from the learned posterior."""
        t4 = self._ensure_4d(target)
        batch_size = t4.shape[0]
        results: list[torch.Tensor] = []
        for _ in range(n_candidates):
            z = torch.randn(batch_size, self.latent_dim, device=t4.device, dtype=t4.dtype)
            mask_4d = self.decoder(z, t4)
            mask = mask_4d.squeeze(0).squeeze(0)
            results.append(self._match_ndim(mask, target))
        return results

    def loss(self, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ELBO loss: reconstruction (BCE) + KL divergence."""
        t4 = self._ensure_4d(target)
        m4 = self._ensure_4d(mask)
        mu, logvar = self.encoder(t4, m4)
        z = _reparameterize(mu, logvar)
        recon = self.decoder(z, t4)

        # Reconstruction: binary cross-entropy per pixel, summed
        recon_loss = nn.functional.binary_cross_entropy(recon, m4, reduction="sum")

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss

    def train_posterior(
        self,
        targets: list[torch.Tensor],
        masks: list[torch.Tensor],
        n_epochs: int = 50,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train the posterior distribution on (target, mask) pairs.

        Args:
            targets: List of target tensors, each (H, W) or (1, H, W).
            masks: List of corresponding mask tensors.
            n_epochs: Training epochs.
            lr: Learning rate.

        Returns:
            List of per-epoch loss values.
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses: list[float] = []

        for _epoch in range(n_epochs):
            epoch_loss = 0.0
            for tgt, msk in zip(targets, masks, strict=False):
                optimizer.zero_grad()
                loss = self.loss(tgt, msk)
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)

        self.eval()
        return losses

    @staticmethod
    def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.ndim == 3:
            return tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _match_ndim(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if ref.ndim == mask.ndim:
            return mask
        if ref.ndim == 2 and mask.ndim == 3:
            mask = mask.squeeze(0)
        elif ref.ndim == 3 and mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return mask


# ---------------------------------------------------------------------------
# BatchILTScheduler
# ---------------------------------------------------------------------------


class BatchILTScheduler:
    """Batch ILT refinement + scoring for posterior candidates.

    Takes multiple candidate masks, refines each independently,
    then scores and ranks them using CandidateScorer.
    """

    def __init__(
        self,
        ilt_refiner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scorer: CandidateScorer | None = None,
    ) -> None:
        self.ilt_refiner = ilt_refiner
        self.scorer = scorer if scorer is not None else CandidateScorer()

    def refine_and_select(
        self,
        candidates: list[torch.Tensor],
        target: torch.Tensor,
        top_k: int = 1,
    ) -> dict[str, object]:
        """Refine all candidates, score, return top_k.

        Args:
            candidates: List of mask tensors to refine.
            target: Design target tensor.
            top_k: Number of top candidates to return.

        Returns:
            Dict with ``best_mask``, ``all_results``, ``best_score``.
        """
        target_f = target.detach().float()
        if target_f.ndim > 2:
            target_f = target_f.squeeze()

        refined: list[tuple[torch.Tensor, float]] = []
        for mask in candidates:
            mask_f = mask.detach().float()
            if mask_f.ndim > 2:
                mask_f = mask_f.squeeze()
            refined_mask = self.ilt_refiner(mask_f, target_f)
            score = self.scorer.score(refined_mask, target_f)
            refined.append((refined_mask, score))

        refined.sort(key=lambda pair: pair[1])
        best_mask = refined[0][0]
        best_score = refined[0][1]

        return {
            "best_mask": best_mask,
            "all_results": refined[:top_k],
            "best_score": best_score,
        }
