"""Train v0.1 "seed" weights for the Neural-ILT U-Net.

This is a thin wrapper around ``scripts/train_neural_ilt.py`` that pins
every knob to the values used to produce the public v0.1 release on
HuggingFace (``openlithohub/neural-ilt-v0.1``). The point is not research
quality — it is *reproducibility*: anyone running this script on any
PyTorch backend should land within numerical noise of the published
weights.

What "v0.1 seed weights" means:

- Trained on synthetic dummy layouts, not LithoBench. The U-Net learns
  to copy structure from `design` to `mask` while staying consistent
  with the Hopkins/SOCS aerial image — i.e. it learns the identity-with-
  rounding behaviour, not real OPC corrections.
- Useful as a starting point for fine-tuning on real data, and as a
  non-trivial baseline whose EPE numbers are *not* `inf` and which other
  models can be compared against.
- Not a substitute for a proper LithoBench-trained release (planned as
  v1.0 once a public training corpus is wired in).

Usage:
    python scripts/train_neural_ilt_seed.py
        # → checkpoints/neural_ilt_v0_1.pt
        # → checkpoints/neural_ilt_v0_1.metadata.json

The script auto-selects MPS / CUDA / CPU in that order. Determinism on
MPS is best-effort — the dataloader uses a single worker and torch is
seeded, but small numerical drift across PyTorch minor versions is
expected.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub.data.dummy import DummyLayoutSpec, generate_dummy_layout
from openlithohub.models._unet import UNet


class _NonEmptyDummyPairs(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Pairs of (design, target_mask) on a 64-grid with non-empty content.

    The default ``DummyLayoutSpec`` uses ``min_width_nm=40`` against
    ``pixel_size_nm=1`` — at ``size=64`` that morphological-opening
    radius wipes the layout to all-zeros, which collapses BCE training
    onto the trivial constant solution. We override ``min_width_nm``
    and ``min_spacing_nm`` to a few pixels so the generated layouts
    actually carry signal.
    """

    def __init__(self, n: int, size: int) -> None:
        self.n = n
        self.size = size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spec = DummyLayoutSpec(
            size=self.size,
            pixel_size_nm=1.0,
            min_width_nm=8.0,
            min_spacing_nm=8.0,
            fill_ratio=0.3,
            seed=idx,
        )
        layout = generate_dummy_layout(spec)
        return layout, layout


@dataclass
class SeedConfig:
    epochs: int = 200
    batch_size: int = 8
    dataset_size: int = 64
    grid: int = 64
    lr: float = 2e-3
    consistency_weight: float = 1.0
    sigma_px: float = 4.5
    output: Path = Path("checkpoints/neural_ilt_v0_1.pt")
    seed: int = 0


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _forward(mask_continuous: torch.Tensor, cfg: SeedConfig) -> torch.Tensor:
    """Gaussian aerial-image proxy.

    We use the Gaussian forward (not Hopkins/SOCS) here for training only,
    because PyTorch's MPS backend does not yet support
    ``torch.linalg.svd``, which the Hopkins kernel build requires. The
    Gaussian model is a fast convolutional approximation that runs
    natively on CUDA / MPS / CPU and is sufficient for learning the
    "identity-with-rounding" behaviour these seed weights encode.

    The published baseline numbers in ``baselines/results.md`` are still
    computed against the Hopkins forward at evaluation time — only
    *training* uses the Gaussian proxy.
    """
    return simulate_aerial_image(mask_continuous, sigma_px=cfg.sigma_px)


def _step(model: UNet, batch: tuple[torch.Tensor, torch.Tensor], cfg: SeedConfig, device: torch.device) -> torch.Tensor:
    design, target_mask = batch
    design = design.to(device).unsqueeze(1)
    target_mask = target_mask.to(device).unsqueeze(1)

    logits = model(design)
    mask_continuous = torch.sigmoid(logits)

    bce = functional.binary_cross_entropy_with_logits(logits, target_mask)
    aerial = _forward(mask_continuous, cfg)
    consistency = functional.mse_loss(aerial, design)
    return bce + cfg.consistency_weight * consistency


def _train(cfg: SeedConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device_str = _pick_device()
    device = torch.device(device_str)
    print(f"Training on device={device_str}")

    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] = _NonEmptyDummyPairs(
        n=cfg.dataset_size, size=cfg.grid
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    history: list[float] = []
    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad()
            loss = _step(model, batch, cfg, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        scheduler.step()
        mean = sum(epoch_losses) / max(1, len(epoch_losses))
        history.append(mean)
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:3d}: loss={mean:.4f}")

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.output)

    metadata = {
        "config": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "dataset_size": cfg.dataset_size,
            "grid": cfg.grid,
            "lr": cfg.lr,
            "consistency_weight": cfg.consistency_weight,
            "seed": cfg.seed,
            "device": device_str,
            "forward_model": "gaussian",
            "forward_sigma_px": cfg.sigma_px,
        },
        "history": history,
        "final_loss": history[-1] if history else math.nan,
    }
    metadata_path = cfg.output.with_suffix(".metadata.json")
    import json

    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nFinal loss: {history[-1]:.4f}")
    print(f"Checkpoint: {cfg.output}")
    print(f"Metadata:   {metadata_path}")
    return metadata


def main() -> None:
    _train(SeedConfig())


if __name__ == "__main__":
    main()

