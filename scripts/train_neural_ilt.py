"""Reference training script for the Neural-ILT U-Net baseline.

This script trains the U-Net used by `openlithohub.models.neural_ilt`. It is
deliberately small — designed as a starting point a contributor can copy and
adapt, not as a production trainer.

Usage:
    # Smoke test (one batch, no real training) — used by CI.
    python scripts/train_neural_ilt.py --smoke-test

    # Real training on LithoBench:
    python scripts/train_neural_ilt.py \
        --data-root /path/to/lithobench \
        --epochs 50 \
        --batch-size 8 \
        --output checkpoints/neural_ilt.pt

After training, upload the checkpoint to HuggingFace Hub:

    huggingface-cli upload <user>/<repo> checkpoints/neural_ilt.pt

then point `NeuralILTModel(pretrained=True)` at it (see
`src/openlithohub/models/neural_ilt.py:58`).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.hopkins import HopkinsParams, simulate_aerial_image_hopkins
from openlithohub.data import generate_dummy_layout
from openlithohub.data.lithobench import LithoBenchDataset
from openlithohub.models._unet import UNet


@dataclass
class TrainConfig:
    data_root: Path | None
    dataset: str
    epochs: int
    batch_size: int
    lr: float
    sigma_px: float
    forward_model: str
    device: str
    output: Path
    smoke_test: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_root"] = str(self.data_root) if self.data_root else None
        d["output"] = str(self.output)
        return d


class _DummyPairs(Dataset):
    """Fallback dataset for `--smoke-test`: design = mask = random dummy layout."""

    def __init__(self, n: int = 16, size: int = 64) -> None:
        self.n = n
        self.size = size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        layout = generate_dummy_layout(size=self.size, seed=idx)
        return layout, layout  # design, target_mask


class _LithoBenchPairs(Dataset):
    """Adapter wrapping LithoBenchDataset to yield (design, mask) tensors."""

    def __init__(self, root: Path) -> None:
        self.inner = LithoBenchDataset(root=root)
        # Filter to samples that actually have a ground-truth mask.
        self._indices = [
            i for i in range(len(self.inner)) if self.inner[i].mask is not None
        ]
        if not self._indices:
            raise RuntimeError(f"No samples with masks in {root}")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.inner[self._indices[idx]]
        assert sample.mask is not None
        return sample.design.float(), sample.mask.float()


def _build_dataset(cfg: TrainConfig) -> Dataset:
    if cfg.smoke_test or cfg.data_root is None:
        return _DummyPairs(n=max(cfg.batch_size, 4), size=64)
    if cfg.dataset == "lithobench":
        return _LithoBenchPairs(cfg.data_root)
    raise ValueError(f"Unsupported dataset: {cfg.dataset}")


def _forward(mask_continuous: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """Apply the configured forward model to (B, 1, H, W) continuous masks.

    Both forward models in `_utils` operate on 2D tensors, so iterate the
    batch dimension explicitly.
    """
    outs = []
    for b in range(mask_continuous.shape[0]):
        m = mask_continuous[b, 0]
        if cfg.forward_model == "gaussian":
            aerial = simulate_aerial_image(m, sigma_px=cfg.sigma_px)
        elif cfg.forward_model == "hopkins":
            params = HopkinsParams(num_kernels=8, pixel_size_nm=2.0)
            aerial = simulate_aerial_image_hopkins(m, params=params)
        else:
            raise ValueError(f"Unknown forward model: {cfg.forward_model}")
        outs.append(aerial)
    return torch.stack(outs).unsqueeze(1)


def _step(model: UNet, batch: tuple[torch.Tensor, torch.Tensor], cfg: TrainConfig) -> torch.Tensor:
    design, target_mask = batch
    design = design.to(cfg.device).unsqueeze(1)
    target_mask = target_mask.to(cfg.device).unsqueeze(1)

    logits = model(design)
    mask_continuous = torch.sigmoid(logits)

    bce = functional.binary_cross_entropy_with_logits(logits, target_mask)
    aerial = _forward(mask_continuous, cfg)
    consistency = functional.mse_loss(aerial, design)
    return bce + 0.1 * consistency


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(0)
    device = torch.device(cfg.device)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    dataset = _build_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    history: list[float] = []
    epochs = 1 if cfg.smoke_test else cfg.epochs

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad()
            loss = _step(model, batch, cfg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            if cfg.smoke_test:
                break  # one batch is enough to prove wiring works
        scheduler.step()
        mean = sum(epoch_losses) / max(1, len(epoch_losses))
        history.append(mean)
        print(f"epoch {epoch}: loss={mean:.4f}")
        if cfg.smoke_test:
            break

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.output)

    metadata = {
        "config": cfg.to_dict(),
        "history": history,
        "final_loss": history[-1] if history else math.nan,
    }
    metadata_path = cfg.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved checkpoint to {cfg.output}")
    print(f"Saved metadata to {metadata_path}")
    return metadata


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=None,
                   help="Path to LithoBench root. If omitted, uses the dummy generator.")
    p.add_argument("--dataset", default="lithobench")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sigma-px", type=float, default=4.5)
    p.add_argument("--forward-model", default="gaussian", choices=["gaussian", "hopkins"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=Path("checkpoints/neural_ilt.pt"))
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a single batch to verify wiring, then exit.")
    args = p.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sigma_px=args.sigma_px,
        forward_model=args.forward_model,
        device=args.device,
        output=args.output,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
