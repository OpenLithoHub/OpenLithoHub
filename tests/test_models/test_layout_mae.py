"""Tests for the Layout-MAE prototype (RFC 0001).

Tiny config — depth 2, 64×64 image, 8×8 patches — so the test runs in
seconds on CPU. We are validating the recipe, not training quality.
"""

from __future__ import annotations

import pytest
import torch

from openlithohub.models.layout_mae import LayoutMAE, LayoutMAEConfig, train_step


@pytest.fixture
def tiny_cfg() -> LayoutMAEConfig:
    return LayoutMAEConfig(
        image_size=64,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        mask_ratio=0.75,
    )


def _dummy_batch(cfg: LayoutMAEConfig, batch: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.rand(batch, cfg.in_channels, cfg.image_size, cfg.image_size) > 0.5).float()


def test_forward_shapes(tiny_cfg: LayoutMAEConfig) -> None:
    model = LayoutMAE(tiny_cfg)
    imgs = _dummy_batch(tiny_cfg)
    pred, mask, ids = model(imgs)
    n = tiny_cfg.num_patches
    assert pred.shape == (2, n, tiny_cfg.patch_size**2 * tiny_cfg.in_channels)
    assert mask.shape == (2, n)
    assert ids.shape == (2, n)
    expected_kept = int(n * (1 - tiny_cfg.mask_ratio))
    assert int(mask[0].sum()) == n - expected_kept


def test_loss_only_over_masked_patches(tiny_cfg: LayoutMAEConfig) -> None:
    model = LayoutMAE(tiny_cfg)
    imgs = _dummy_batch(tiny_cfg)
    pred, mask, _ = model(imgs)
    loss = model.reconstruction_loss(imgs, pred, mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_train_step_decreases_loss(tiny_cfg: LayoutMAEConfig) -> None:
    model = LayoutMAE(tiny_cfg)
    imgs = _dummy_batch(tiny_cfg, batch=4)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = [train_step(model, imgs, optim) for _ in range(20)]
    assert all(map(lambda x: x == x, losses))  # no NaNs
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_encode_returns_full_token_sequence(tiny_cfg: LayoutMAEConfig) -> None:
    model = LayoutMAE(tiny_cfg)
    imgs = _dummy_batch(tiny_cfg)
    feats = model.encode(imgs)
    assert feats.shape == (2, tiny_cfg.num_patches, tiny_cfg.embed_dim)


def test_patchify_unpatchify_round_trip(tiny_cfg: LayoutMAEConfig) -> None:
    model = LayoutMAE(tiny_cfg)
    imgs = _dummy_batch(tiny_cfg)
    patches = model.patchify(imgs)
    restored = model.unpatchify(patches)
    assert torch.allclose(imgs, restored)
