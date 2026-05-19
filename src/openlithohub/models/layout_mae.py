"""Layout-MAE — masked autoencoder for rasterised PDK layouts (RFC 0001 prototype).

Minimal, training-runnable ViT-S MAE in pure PyTorch. Implements the
architecture and pretraining loop described in
``docs/rfcs/0001-base-model.md`` with one purpose: validate the recipe on
the synthetic-batch tier, end-to-end, on a single machine.

What this prototype is:
- A standalone ViT encoder + ViT decoder MAE.
- Mask-and-reconstruct loss on masked patches only (per-pixel L1).
- A ``train_step`` you can call from a notebook or test.

What it is NOT (yet):
- No `from_pretrained` / Hub release. Pretraining at the RFC scale (200k
  steps, A100) is a v0.2 deliverable; this code is the recipe.
- No fine-tune adapter API yet. ``set_decoder`` is wired to a stub for
  the v0.2 follow-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LayoutMAEConfig:
    """ViT-S defaults from RFC 0001."""

    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 1
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    decoder_embed_dim: int = 256
    decoder_depth: int = 4
    decoder_num_heads: int = 8
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.75

    @property
    def num_patches(self) -> int:
        n = self.image_size // self.patch_size
        return n * n


def _sincos_pos_embed(num_patches: int, embed_dim: int) -> torch.Tensor:
    grid_size = int(math.sqrt(num_patches))
    assert grid_size * grid_size == num_patches
    pos_h = torch.arange(grid_size, dtype=torch.float32)
    pos_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(pos_h, pos_w, indexing="ij"), dim=0).reshape(2, -1)

    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim / 4.0)
    omega = 1.0 / (10000**omega)

    out_h = torch.einsum("m,d->md", grid[0], omega)
    out_w = torch.einsum("m,d->md", grid[1], omega)
    pos_embed = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)], dim=1
    )
    return pos_embed


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class LayoutMAE(nn.Module):
    """Masked-autoencoder over rasterised layout patches."""

    def __init__(self, config: LayoutMAEConfig | None = None):
        super().__init__()
        cfg = config or LayoutMAEConfig()
        self.config = cfg

        self.patch_embed = nn.Conv2d(
            cfg.in_channels, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.encoder_pos_embed = nn.Parameter(
            _sincos_pos_embed(cfg.num_patches, cfg.embed_dim), requires_grad=False
        )
        self.encoder_blocks = nn.ModuleList(
            [_Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.encoder_norm = nn.LayerNorm(cfg.embed_dim)

        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = nn.Parameter(
            _sincos_pos_embed(cfg.num_patches, cfg.decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _Block(cfg.decoder_embed_dim, cfg.decoder_num_heads, cfg.mlp_ratio)
                for _ in range(cfg.decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(cfg.decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            cfg.decoder_embed_dim, cfg.patch_size * cfg.patch_size * cfg.in_channels
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        b, c, h, w = imgs.shape
        assert h == cfg.image_size and w == cfg.image_size
        p = cfg.patch_size
        x = imgs.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, cfg.num_patches, p * p * c)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        b = x.shape[0]
        p = cfg.patch_size
        grid = cfg.image_size // p
        x = x.reshape(b, grid, grid, p, p, cfg.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, cfg.in_channels, cfg.image_size, cfg.image_size)
        return x

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        len_keep = int(n * (1 - mask_ratio))
        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        mask = torch.ones(b, n, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_kept, mask, ids_restore

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Frozen-feature path used by downstream consumers (no masking)."""
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.encoder_pos_embed
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x

    def forward(
        self, imgs: torch.Tensor, mask_ratio: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        ratio = cfg.mask_ratio if mask_ratio is None else mask_ratio

        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.encoder_pos_embed
        x_kept, mask, ids_restore = self.random_masking(x, ratio)
        for blk in self.encoder_blocks:
            x_kept = blk(x_kept)
        x_kept = self.encoder_norm(x_kept)

        x_dec = self.decoder_embed(x_kept)
        b, n_full, _ = x.shape
        n_kept = x_dec.shape[1]
        mask_tokens = self.mask_token.expand(b, n_full - n_kept, -1)
        x_full = torch.cat([x_dec, mask_tokens], dim=1)
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]))
        x_full = x_full + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)
        pred = self.decoder_pred(x_full)
        return pred, mask, ids_restore

    def reconstruction_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """L1 over masked patches only (RFC 0001 §Architecture)."""
        target = self.patchify(imgs)
        per_patch = (pred - target).abs().mean(dim=-1)
        loss = (per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        return loss


def train_step(
    model: LayoutMAE,
    imgs: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Single MAE training step. Returns the scalar reconstruction loss."""
    model.train()
    pred, mask, _ = model(imgs)
    loss = model.reconstruction_loss(imgs, pred, mask)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
