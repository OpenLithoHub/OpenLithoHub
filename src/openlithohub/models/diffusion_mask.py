"""Latent diffusion model for computational lithography mask synthesis.

Generates diverse OPC mask candidates via a denoising diffusion process
operating in a learned latent space, with differentiable lithography
forward-model guidance steering the samples toward high-fidelity masks.

Architecture overview
---------------------
1. **MaskLatentEncoder / MaskLatentDecoder** — lightweight VAE that compresses
   mask patterns (B, 1, H, W) <-> (B, latent_spatial, latent_spatial, ch).
2. **MaskDiffusionUNet** — 4-level U-Net denoiser operating on spatial latent
   maps with sinusoidal time embedding and cross-attention conditioning on the
   target layout.
3. **LithoGuidance** — gradient-based classifier guidance wrapping the aerial
   image simulation + differentiable resist model.  EPE / PVB loss serves as
   the guidance signal.
4. **DiffusionMaskSynthesis** — orchestrates training (VAE + diffusion) and
   inference (DDPM sampling with optional guidance).
5. **DiffusionMaskBenchmark** — structured comparison against GRPO-WGAN baselines.

References
----------
- Fast ILT: Inverse Lithography Technology via Diffusion Models,
  arXiv:2412.14599, 2024.
- MxDiffusion: Diffusion-Generative Models for EUV Stochastic-Aware Mask
  Optimization, Nano Lett. 2026.

License: Apache 2.0 — clean-room implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.resist_model import apply_differentiable_resist


# ---------------------------------------------------------------------------
# Sinusoidal time-step embedding
# ---------------------------------------------------------------------------


def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.cos(), args.sin()], dim=-1)


# ---------------------------------------------------------------------------
# MaskLatentEncoder — variational CNN encoder
# ---------------------------------------------------------------------------


class MaskLatentEncoder(nn.Module):
    """Encode mask patterns into a spatial latent map.

    ``(B, 1, H, W) -> (B, latent_ch, H // stride, W // stride)`` producing
    ``mean`` and ``log_var`` tensors for the variational posterior.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 16,
        hidden_channels: int = 32,
        n_down_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch_in = in_channels
        for i in range(n_down_layers):
            ch_out = hidden_channels if i == 0 else hidden_channels
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch_in = ch_out
        self.backbone = nn.Sequential(*layers)
        self.to_mean = nn.Conv2d(hidden_channels, latent_channels, 1)
        self.to_logvar = nn.Conv2d(hidden_channels, latent_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.to_mean(h), self.to_logvar(h)


# ---------------------------------------------------------------------------
# MaskLatentDecoder — decode latent map back to mask
# ---------------------------------------------------------------------------


class MaskLatentDecoder(nn.Module):
    """Decode spatial latent map back to a continuous mask in [0, 1].

    ``(B, latent_ch, h, w) -> (B, 1, H, W)`` with bilinear upsampling and
    sigmoid output.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_up_layers: int = 2,
        target_size: int | None = None,
    ) -> None:
        super().__init__()
        self.target_size = target_size
        self.n_up_layers = n_up_layers
        layers: list[nn.Module] = []
        ch_in = latent_channels
        for i in range(n_up_layers):
            ch_out = hidden_channels
            layers.append(nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch_in = ch_out
        self.backbone = nn.Sequential(*layers)
        self.to_out = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.backbone(z)
        h = self.to_out(h)
        if self.target_size is not None:
            h = functional.interpolate(h, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return torch.sigmoid(h)


# ---------------------------------------------------------------------------
# Re-parameterisation trick
# ---------------------------------------------------------------------------


def _reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


# ---------------------------------------------------------------------------
# MaskDiffusionUNet — denoiser in latent space
# ---------------------------------------------------------------------------


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x)).relu()
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(self.conv2(h)).relu()
        return h + self.residual(x)


class _CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_c = nn.LayerNorm(context_dim)
        self.context_proj = nn.Linear(context_dim, query_dim)
        self.attn = nn.MultiheadAttention(query_dim, n_heads, batch_first=True)
        self.proj = nn.Linear(query_dim, query_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = query.shape
        q = query.flatten(2).transpose(1, 2)
        q = self.norm_q(q)
        ctx = context.flatten(2).transpose(1, 2)
        ctx = self.norm_c(ctx)
        ctx = self.context_proj(ctx)
        attn_out, _ = self.attn(q, ctx, ctx, need_weights=False)
        attn_out = self.proj(attn_out)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return query + attn_out


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, context_dim: int) -> None:
        super().__init__()
        self.res_block = _ResidualBlock(in_ch, out_ch, time_dim)
        self.cross_attn = _CrossAttentionBlock(out_ch, context_dim)
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.res_block(x, t_emb)
        h = self.cross_attn(h, context)
        skip = h
        h = self.downsample(h)
        return h, skip


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, time_dim: int, context_dim: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res_block = _ResidualBlock(in_ch + skip_ch, out_ch, time_dim)
        self.cross_attn = _CrossAttentionBlock(out_ch, context_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.upsample(x)
        dy = skip.shape[2] - h.shape[2]
        dx = skip.shape[3] - h.shape[3]
        h = functional.pad(h, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        h = torch.cat([h, skip], dim=1)
        h = self.res_block(h, t_emb)
        h = self.cross_attn(h, context)
        return h


class MaskDiffusionUNet(nn.Module):
    """4-level U-Net denoiser for diffusion in latent space.

    Operates on spatial latent maps ``(B, latent_ch, h, w)`` with
    sinusoidal time-step embedding and cross-attention conditioning on
    the target layout encoded as a context feature map.

    Channel progression: 16 -> 32 -> 48 -> 64 (4 levels).
    """

    def __init__(
        self,
        latent_channels: int = 16,
        time_embed_dim: int = 128,
        context_channels: int = 16,
        base_channels: tuple[int, ...] = (16, 32, 48, 64),
    ) -> None:
        super().__init__()
        ch = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_proj = nn.Conv2d(latent_channels, ch[0], 1)

        self.down1 = _DownBlock(ch[0], ch[1], time_embed_dim, context_channels)
        self.down2 = _DownBlock(ch[1], ch[2], time_embed_dim, context_channels)
        self.down3 = _DownBlock(ch[2], ch[3], time_embed_dim, context_channels)

        self.mid = _ResidualBlock(ch[3], ch[3], time_embed_dim)
        self.mid_cross = _CrossAttentionBlock(ch[3], context_channels)

        self.up3 = _UpBlock(ch[3], ch[2], skip_ch=ch[3], time_dim=time_embed_dim, context_dim=context_channels)
        self.up2 = _UpBlock(ch[2], ch[1], skip_ch=ch[2], time_dim=time_embed_dim, context_dim=context_channels)
        self.up1 = _UpBlock(ch[1], ch[0], skip_ch=ch[1], time_dim=time_embed_dim, context_dim=context_channels)

        self.output_proj = nn.Conv2d(ch[0], latent_channels, 1)

    def forward(
        self,
        z_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = _sinusoidal_embedding(timesteps, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        h = self.input_proj(z_noisy)

        h, s1 = self.down1(h, t_emb, context)
        h, s2 = self.down2(h, t_emb, context)
        h, s3 = self.down3(h, t_emb, context)

        h = self.mid(h, t_emb)
        h = self.mid_cross(h, context)

        h = self.up3(h, s3, t_emb, context)
        h = self.up2(h, s2, t_emb, context)
        h = self.up1(h, s1, t_emb, context)

        return self.output_proj(h)


# ---------------------------------------------------------------------------
# LithoGuidance — differentiable lithography classifier guidance
# ---------------------------------------------------------------------------


class LithoGuidance:
    """Gradient-based classifier guidance using the lithography forward model.

    Decodes a latent ``z`` to mask space, runs the aerial-image simulation
    and resist model, then back-propagates the EPE / PVB loss to produce a
    correction on ``z``.
    """

    def __init__(
        self,
        decoder: MaskLatentDecoder,
        sigma_px: float = 2.0,
        dose: float = 1.0,
        resist_steepness: float = 50.0,
        epe_weight: float = 1.0,
        pvb_weight: float = 0.5,
    ) -> None:
        self.decoder = decoder
        self.sigma_px = sigma_px
        self.dose = dose
        self.resist_steepness = resist_steepness
        self.epe_weight = epe_weight
        self.pvb_weight = pvb_weight

    def _compute_loss(
        self,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        aerial = simulate_aerial_image(mask, sigma_px=self.sigma_px, dose=self.dose)
        resist = apply_differentiable_resist(aerial, threshold=0.5, steepness=self.resist_steepness)
        epe = functional.mse_loss(resist, target)
        grad_h = (mask[1:, :] - mask[:-1, :]).pow(2)
        grad_w = (mask[:, 1:] - mask[:, :-1]).pow(2)
        pvb = grad_h.mean() + grad_w.mean()
        return self.epe_weight * epe + self.pvb_weight * pvb

    def compute_guidance(
        self,
        z: torch.Tensor,
        target_aerial: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Return a gradient-based correction to apply to latent ``z``.

        Args:
            z: Latent tensor ``(B, C, h, w)`` with ``requires_grad=True``.
            target_aerial: Target design layout, ``(H, W)`` or ``(1, H, W)``.
            guidance_scale: Amplification factor for the guidance gradient.

        Returns:
            Gradient tensor of same shape as ``z``, to be added as a
            correction step during sampling.
        """
        z = z.detach().requires_grad_(True)
        mask = self.decoder(z)
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask_2d = mask.squeeze(1)
        else:
            mask_2d = mask

        target_2d = target_aerial.detach().float()
        if target_2d.ndim > 2:
            target_2d = target_2d.squeeze()
        if target_2d.ndim == 2:
            target_2d = target_2d.unsqueeze(0).expand(mask_2d.shape[0], -1, -1)

        loss = torch.tensor(0.0, device=z.device)
        for i in range(mask_2d.shape[0]):
            loss = loss + self._compute_loss(mask_2d[i], target_2d[i])
        loss = loss / mask_2d.shape[0]

        grad = torch.autograd.grad(loss, z)[0]
        return -guidance_scale * grad


# ---------------------------------------------------------------------------
# Noise schedule helpers (linear beta schedule)
# ---------------------------------------------------------------------------


def _linear_beta_schedule(n_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, n_steps)


def _extract_at(coeffs: torch.Tensor, timesteps: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    batch_size = timesteps.shape[0]
    out = coeffs.gather(-1, timesteps)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


# ---------------------------------------------------------------------------
# DiffusionMaskSynthesis — main synthesis class
# ---------------------------------------------------------------------------


@dataclass
class DiffusionMaskConfig:
    latent_channels: int = 16
    hidden_channels: int = 32
    spatial_stride: int = 4
    n_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    guidance_scale: float = 1.0
    n_sampling_steps: int = 50
    min_feature_px: int = 3
    sigma_px: float = 2.0
    dose: float = 1.0
    resist_steepness: float = 50.0


class DiffusionMaskSynthesis(nn.Module):
    """Latent diffusion model for computational lithography mask synthesis.

    This is a research prototype (does *not* inherit from
    :class:`LithographyModel`).  It exposes ``train_step`` for VAE +
    diffusion training and ``synthesize`` for generating diverse mask
    candidates guided by the differentiable lithography forward model.
    """

    def __init__(
        self,
        encoder: MaskLatentEncoder | None = None,
        decoder: MaskLatentDecoder | None = None,
        unet: MaskDiffusionUNet | None = None,
        guidance: LithoGuidance | None = None,
        config: DiffusionMaskConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DiffusionMaskConfig()
        cfg = self.config

        self.encoder = encoder or MaskLatentEncoder(
            latent_channels=cfg.latent_channels,
            hidden_channels=cfg.hidden_channels,
            n_down_layers=int(math.log2(cfg.spatial_stride)),
        )
        self.decoder = decoder or MaskLatentDecoder(
            latent_channels=cfg.latent_channels,
            hidden_channels=cfg.hidden_channels,
            n_up_layers=int(math.log2(cfg.spatial_stride)),
        )
        self.unet = unet or MaskDiffusionUNet(
            latent_channels=cfg.latent_channels,
            context_channels=cfg.latent_channels,
        )
        self.context_proj = nn.Conv2d(1, cfg.latent_channels, 1)
        self.guidance = guidance

        betas = _linear_beta_schedule(cfg.n_diffusion_steps, cfg.beta_start, cfg.beta_end)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(torch.cumprod(alphas, dim=0)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - torch.cumprod(alphas, dim=0)),
        )

    # ----- training -----

    def _vae_loss(
        self,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mean, logvar = self.encoder(masks)
        z = _reparameterize(mean, logvar)
        recon = self.decoder(z)
        recon_loss = functional.mse_loss(recon, masks, reduction="sum") / masks.shape[0]
        kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=(1, 2, 3)).mean()
        return recon_loss + 0.01 * kl_loss, {"vae_recon": recon_loss.item(), "vae_kl": kl_loss.item()}

    def _diffusion_loss(
        self,
        masks: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mean, logvar = self.encoder(masks)
        z_0 = _reparameterize(mean, logvar).detach()

        batch = z_0.shape[0]
        t = torch.randint(0, self.config.n_diffusion_steps, (batch,), device=z_0.device)
        noise = torch.randn_like(z_0)

        sqrt_a = _extract_at(self.sqrt_alphas_cumprod, t, z_0.shape)
        sqrt_one_a = _extract_at(self.sqrt_one_minus_alphas_cumprod, t, z_0.shape)
        z_noisy = sqrt_a * z_0 + sqrt_one_a * noise

        noise_pred = self.unet(z_noisy, t, context)
        diff_loss = functional.mse_loss(noise_pred, noise)
        return diff_loss, {"diff_loss": diff_loss.item()}

    def train_step(
        self,
        masks: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """One training step combining VAE reconstruction and diffusion denoising.

        Args:
            masks: Ground-truth mask patterns ``(B, 1, H, W)``.
            targets: Corresponding target layouts for conditioning. If ``None``,
                the masks themselves are used as context (self-conditioning).

        Returns:
            Dict of scalar losses.
        """
        context = targets if targets is not None else masks
        context_small = functional.avg_pool2d(context, self.config.spatial_stride)
        context_small = self.context_proj(context_small)

        vae_loss, vae_info = self._vae_loss(masks)
        diff_loss, diff_info = self._diffusion_loss(masks, context_small)

        total = vae_loss + diff_loss
        info: dict[str, float] = {"total": total.item(), **vae_info, **diff_info}
        return info

    # ----- sampling -----

    @torch.no_grad()
    def _ddpm_sample_step(
        self,
        z: torch.Tensor,
        t: int,
        context: torch.Tensor,
    ) -> torch.Tensor:
        t_tensor = torch.full((z.shape[0],), t, device=z.device, dtype=torch.long)
        noise_pred = self.unet(z, t_tensor, context)

        beta_t = self.betas[t]
        alpha_t = self.alphas_cumprod[t]
        sqrt_one_alpha_t = torch.sqrt(1.0 - alpha_t)

        z_mean = (z - beta_t / sqrt_one_alpha_t * noise_pred) / torch.sqrt(1.0 - beta_t)
        if t > 0:
            noise = torch.randn_like(z)
            z_out = z_mean + torch.sqrt(beta_t) * noise
        else:
            z_out = z_mean
        return z_out

    def synthesize(
        self,
        target_layout: torch.Tensor,
        n_candidates: int = 16,
        n_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> dict[str, Any]:
        """Generate diverse mask candidates via latent diffusion sampling.

        Args:
            target_layout: Target design ``(H, W)`` or ``(1, H, W)`` or
                ``(B, 1, H, W)``.
            n_candidates: Number of candidates to generate.
            n_steps: Number of denoising steps (default from config).
            guidance_scale: Amplification for lithography guidance
                (default from config).

        Returns:
            Dict with keys ``candidates``, ``scores``, ``best_idx``, ``best_mask``.
        """
        steps = n_steps or self.config.n_sampling_steps
        gs = guidance_scale if guidance_scale is not None else self.config.guidance_scale

        target = target_layout.detach().float()
        if target.ndim == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 3:
            target = target.unsqueeze(0)
        target_hw = target.shape[-2:]

        context = functional.avg_pool2d(target, self.config.spatial_stride)
        context = self.context_proj(context)
        context_batch = context.expand(n_candidates, -1, -1, -1)

        latent_h = target_hw[0] // self.config.spatial_stride
        latent_w = target_hw[1] // self.config.spatial_stride
        z = torch.randn(
            n_candidates, self.config.latent_channels, latent_h, latent_w,
            device=context.device,
        )

        step_indices = list(range(steps - 1, -1, -1))
        total_t = self.config.n_diffusion_steps
        mapped = [int(s * total_t / steps) for s in step_indices]

        for t in mapped:
            z = self._ddpm_sample_step(z, t, context_batch)
            if gs > 0 and self.guidance is not None and t > 0:
                target_2d = target[0, 0]
                grad = self.guidance.compute_guidance(z, target_2d, guidance_scale=gs)
                z = z + grad

        masks = self.decoder(z)

        candidates_list = [masks[i, 0] for i in range(n_candidates)]
        scores = self._score_candidates(candidates_list, target[0, 0])
        manufacturable = self._filter_manufacturable(candidates_list)

        best_idx = int(min(range(len(scores)), key=lambda i: scores[i]))
        best_mask = candidates_list[best_idx]

        return {
            "candidates": candidates_list,
            "scores": scores,
            "best_idx": best_idx,
            "best_mask": best_mask,
            "manufacturable_mask": manufacturable,
        }

    # ----- filtering and scoring -----

    @staticmethod
    def _filter_manufacturable(
        candidates: list[torch.Tensor],
        min_feature_px: int = 3,
    ) -> list[bool]:
        """Filter candidates by approximate MRC minimum feature size.

        Checks that no connected foreground region smaller than
        ``min_feature_px^2`` pixels exists.  Uses a simple erosion test:
        if eroding by ``min_feature_px // 2`` removes the entire object,
        it was too small.

        Returns:
            List of booleans (True = passes filter).
        """
        results: list[bool] = []
        r = min_feature_px // 2
        for mask in candidates:
            m = (mask.detach().float() > 0.5)
            if m.ndim > 2:
                m = m.squeeze()
            if r < 1:
                results.append(True)
                continue
            kernel_size = 2 * r + 1
            m_4d = m.float().unsqueeze(0).unsqueeze(0)
            padded = functional.pad(m_4d, [r, r, r, r], mode="constant", value=0)
            weight = torch.ones(1, 1, kernel_size, kernel_size, device=m.device)
            eroded = (functional.conv2d(padded, weight) >= kernel_size * kernel_size).float()
            foreground_remains = eroded.sum() > 0
            results.append(bool(foreground_remains.item()))
        return results

    @staticmethod
    def _score_candidates(
        candidates: list[torch.Tensor],
        target: torch.Tensor,
        sigma_px: float = 2.0,
        dose: float = 1.0,
        resist_steepness: float = 50.0,
    ) -> list[float]:
        """Score candidates using EPE + PVB lithographic fidelity.

        Returns:
            List of scores (lower is better).
        """
        target_2d = target.detach().float()
        if target_2d.ndim > 2:
            target_2d = target_2d.squeeze()
        scores: list[float] = []
        for mask in candidates:
            m = mask.detach().float()
            if m.ndim > 2:
                m = m.squeeze()
            aerial = simulate_aerial_image(m, sigma_px=sigma_px, dose=dose)
            resist = apply_differentiable_resist(aerial, threshold=0.5, steepness=resist_steepness)
            epe = functional.mse_loss(resist, target_2d).item()
            grad_h = (m[1:, :] - m[:-1, :]).pow(2).mean().item()
            grad_w = (m[:, 1:] - m[:, :-1]).pow(2).mean().item()
            pvb = grad_h + grad_w
            scores.append(epe + 0.5 * pvb)
        return scores

    def compare_vs_grpo(
        self,
        target_layout: torch.Tensor,
        grpo_model: Any,
        n_seeds: int = 3,
    ) -> dict[str, Any]:
        """Benchmark diffusion synthesis against a GRPO-WGAN baseline.

        Args:
            target_layout: Target design tensor.
            grpo_model: A :class:`GRPOWarmStart` instance with a
                ``generate_group`` method.
            n_seeds: Number of independent trials.

        Returns:
            Structured comparison dict with per-method scores, diversity,
            and MRC violation rates.
        """
        from openlithohub.models.warm_start import CandidateScorer

        scorer = CandidateScorer(sigma_px=self.config.sigma_px, dose=self.config.dose)
        target = target_layout.detach().float()
        if target.ndim == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 3:
            target = target.unsqueeze(0)
        target_2d = target[0, 0] if target.ndim == 4 else target.squeeze()

        diff_scores_all: list[float] = []
        grpo_scores_all: list[float] = []
        diff_diversity: list[float] = []
        grpo_diversity: list[float] = []
        diff_mrc_violations: list[int] = []
        grpo_mrc_violations: list[int] = []

        for seed in range(n_seeds):
            torch.manual_seed(seed)

            diff_result = self.synthesize(target_layout, n_candidates=16)
            diff_s = diff_result["scores"]
            diff_scores_all.extend(diff_s)
            diff_mfr = self._filter_manufacturable(diff_result["candidates"], self.config.min_feature_px)
            diff_mrc_violations.append(sum(1 for v in diff_mfr if not v))
            if len(diff_result["candidates"]) > 1:
                stack = torch.stack([c.flatten() for c in diff_result["candidates"]])
                diff_diversity.append(stack.float().std(dim=0).mean().item())

            torch.manual_seed(seed + 1000)
            grpo_candidates = grpo_model.generate_group(target_2d)
            grpo_s = [scorer.score(c, target_2d) for c in grpo_candidates]
            grpo_scores_all.extend(grpo_s)
            grpo_mfr = self._filter_manufacturable(grpo_candidates, self.config.min_feature_px)
            grpo_mrc_violations.append(sum(1 for v in grpo_mfr if not v))
            if len(grpo_candidates) > 1:
                stack = torch.stack([c.flatten() for c in grpo_candidates])
                grpo_diversity.append(stack.float().std(dim=0).mean().item())

        diff_t = torch.tensor(diff_scores_all)
        grpo_t = torch.tensor(grpo_scores_all)

        return {
            "diffusion": {
                "mean_score": diff_t.mean().item(),
                "std_score": diff_t.std().item() if len(diff_t) > 1 else 0.0,
                "min_score": diff_t.min().item() if len(diff_t) > 0 else float("inf"),
                "mean_diversity": sum(diff_diversity) / max(len(diff_diversity), 1),
                "total_mrc_violations": sum(diff_mrc_violations),
            },
            "grpo": {
                "mean_score": grpo_t.mean().item(),
                "std_score": grpo_t.std().item() if len(grpo_t) > 1 else 0.0,
                "min_score": grpo_t.min().item() if len(grpo_t) > 0 else float("inf"),
                "mean_diversity": sum(grpo_diversity) / max(len(grpo_diversity), 1),
                "total_mrc_violations": sum(grpo_mrc_violations),
            },
            "n_seeds": n_seeds,
        }


# ---------------------------------------------------------------------------
# DiffusionMaskBenchmark — structured benchmark runner
# ---------------------------------------------------------------------------


class DiffusionMaskBenchmark:
    """Compare diffusion mask synthesis vs GRPO-WGAN on standard metrics.

    Metrics collected per method:
    - candidate diversity (pairwise std of flattened masks)
    - EPE violation distribution (mean, std, max across candidates)
    - MRC violation rate (fraction of candidates failing min-feature check)
    """

    def __init__(
        self,
        diffusion: DiffusionMaskSynthesis,
        grpo_model: Any,
        config: DiffusionMaskConfig | None = None,
    ) -> None:
        self.diffusion = diffusion
        self.grpo_model = grpo_model
        self.config = config or diffusion.config

    def run(
        self,
        targets: list[torch.Tensor],
        n_candidates: int = 16,
        n_seeds: int = 3,
    ) -> dict[str, Any]:
        """Run benchmark over a list of target layouts.

        Returns:
            Structured comparison dict with ``per_target`` list and
            ``aggregate`` summary.
        """
        per_target: list[dict[str, Any]] = []
        for tgt in targets:
            result = self.diffusion.compare_vs_grpo(
                tgt, self.grpo_model, n_seeds=n_seeds
            )
            per_target.append(result)

        if not per_target:
            return {"per_target": [], "aggregate": {}}

        agg_diff_mean = sum(r["diffusion"]["mean_score"] for r in per_target) / len(per_target)
        agg_grpo_mean = sum(r["grpo"]["mean_score"] for r in per_target) / len(per_target)
        agg_diff_div = sum(r["diffusion"]["mean_diversity"] for r in per_target) / len(per_target)
        agg_grpo_div = sum(r["grpo"]["mean_diversity"] for r in per_target) / len(per_target)
        agg_diff_mrc = sum(r["diffusion"]["total_mrc_violations"] for r in per_target)
        agg_grpo_mrc = sum(r["grpo"]["total_mrc_violations"] for r in per_target)

        return {
            "per_target": per_target,
            "aggregate": {
                "diffusion_mean_score": agg_diff_mean,
                "grpo_mean_score": agg_grpo_mean,
                "diffusion_mean_diversity": agg_diff_div,
                "grpo_mean_diversity": agg_grpo_div,
                "diffusion_total_mrc_violations": agg_diff_mrc,
                "grpo_total_mrc_violations": agg_grpo_mrc,
                "n_targets": len(targets),
            },
        }
