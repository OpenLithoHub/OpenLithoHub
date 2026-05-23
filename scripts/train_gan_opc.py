"""Reference training script for the GAN-OPC generator baseline.

This script trains the U-Net used by `openlithohub.models.gan_opc`. It mirrors
`scripts/train_neural_ilt.py` — same UNet, same BCE + forward-consistency
loss formulation — but ingests the GAN-OPC paired-PNG dataset
(`Yang2018_GANOPC`) instead of LithoBench.

Scope (v0.2, 2026-05-23 follow-up to v0.1): generator-only. Adds an MRC-aware
soft loss term (`curvilinear_mrc_loss`), switches default forward model to
Hopkins SOCS, and pre-FFTs SOCS kernels via the new
`precomputed_kernels_f` kwarg on `simulate_aerial_image_hopkins` to keep
training time within the overnight slot. The paper's discriminator side
(§IV.B) is still not trained here; the in-tree adapter `GanOpcModel` only
loads the generator, so a generator-only checkpoint matches the contract.

Usage:
    # Smoke test (one batch, no real training) — used by CI.
    python scripts/train_gan_opc.py --smoke-test

    # v0.2 real training:
    python scripts/train_gan_opc.py \\
        --data-root data/ganopc/extracted/ganopc-data \\
        --resize-to 256 \\
        --epochs 50 \\
        --batch-size 8 \\
        --device mps \\
        --forward-model hopkins \\
        --num-kernels 24 \\
        --lambda-mrc 1.0 \\
        --output checkpoints/gan_opc_v0_2.pt

After training, upload the checkpoint to HuggingFace Hub:

    huggingface-cli upload openlithohub/gan-opc-v0.1 \\
        checkpoints/gan_opc_v0_2.pt model.pt

then `GanOpcModel(pretrained=True)` will resolve to it (see
`src/openlithohub/models/gan_opc.py:64`).

Notes on resolution:
    GAN-OPC PNGs are 2048×2048; UNet at full resolution would not fit on
    an M-series MPS device with batch=8. The paper trains on 256×256
    crops (§IV.A "input image size"); pass `--resize-to 256` to match.
    The resize uses `mode="area"` (correct antialiasing for binary
    downsampling) followed by a `> 0.5` threshold so design and target
    stay in {0, 1}. (v0.1 used bilinear; that quietly eroded sub-4-px
    features before training saw them.)
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
from openlithohub._utils.hopkins import (
    HopkinsParams,
    compute_socs_kernels,
    simulate_aerial_image_hopkins,
)
from openlithohub.benchmark.metrics.mrc_loss import curvilinear_mrc_loss
from openlithohub.data.ganopc import GanOpcDataset
from openlithohub.models._unet import UNet


@dataclass
class TrainConfig:
    data_root: Path | None
    epochs: int
    batch_size: int
    lr: float
    sigma_px: float
    forward_model: str
    device: str
    output: Path
    resize_to: int | None
    consistency_weight: float
    smoke_test: bool
    num_kernels: int
    pixel_size_nm: float
    lambda_mrc: float
    lambda_mrc_warmup_epochs: int
    lambda_mrc_warmup_start: float
    mrc_min_width_nm: float
    mrc_min_spacing_nm: float
    mrc_weight_min_spacing: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_root"] = str(self.data_root) if self.data_root else None
        d["output"] = str(self.output)
        return d


class _DummyPairs(Dataset):
    """Fallback dataset for `--smoke-test`. Random binary patterns."""

    def __init__(self, n: int = 4, size: int = 64) -> None:
        self.n = n
        self.size = size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(idx)
        layout = (torch.rand(self.size, self.size, generator=gen) > 0.7).float()
        return layout, layout


class _GanOpcPairs(Dataset):
    """Adapter wrapping GanOpcDataset to yield (design, mask) tensors.

    Optionally resizes from native 2048×2048 down to ``resize_to`` so the
    training loop fits in unified memory on consumer Apple Silicon.
    """

    def __init__(self, root: Path, resize_to: int | None) -> None:
        self.inner = GanOpcDataset(root=root)
        self.resize_to = resize_to

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.inner[idx]
        assert sample.mask is not None
        design = sample.design.float()
        mask = sample.mask.float()
        if self.resize_to is not None and design.shape[-1] != self.resize_to:
            design = self._resize(design, self.resize_to)
            mask = self._resize(mask, self.resize_to)
        return design, mask

    @staticmethod
    def _resize(t: torch.Tensor, target: int) -> torch.Tensor:
        # B2 fix: ``mode="area"`` is the correct antialiasing filter for
        # downsampling binary masks; bilinear+threshold (v0.1) quietly
        # eroded sub-4-native-px features before training saw them.
        # ``align_corners`` has no meaning for ``mode="area"`` and must
        # be dropped.
        t4 = t.unsqueeze(0).unsqueeze(0)
        resized = functional.interpolate(t4, size=(target, target), mode="area")
        return (resized.squeeze(0).squeeze(0) > 0.5).float()


def _build_dataset(cfg: TrainConfig) -> Dataset:
    if cfg.smoke_test or cfg.data_root is None:
        return _DummyPairs(n=max(cfg.batch_size, 4), size=64)
    return _GanOpcPairs(cfg.data_root, resize_to=cfg.resize_to)


def _forward(
    mask_continuous: torch.Tensor,
    cfg: TrainConfig,
    kernels: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    kernels_f: torch.Tensor | None = None,
) -> torch.Tensor:
    if cfg.forward_model == "gaussian":
        return simulate_aerial_image(mask_continuous, sigma_px=cfg.sigma_px)
    if cfg.forward_model == "hopkins":
        # B1 fix: pixel_size_nm=8.0 (was 2.0, copy-pasted from neural-ilt).
        # The 2048→256 resize gives an effective pitch of 8 nm/px on
        # GAN-OPC's 2 µm-tile inputs; 2.0 nm/px built SOCS kernels for
        # a 4× finer grid than the mask actually represented.
        if kernels is not None and weights is not None:
            return simulate_aerial_image_hopkins(
                mask_continuous,
                kernels=kernels,
                weights=weights,
                precomputed_kernels_f=kernels_f,
            )
        # Fallback path (smoke test / no precompute): build kernels lazily.
        params = HopkinsParams(num_kernels=cfg.num_kernels, pixel_size_nm=cfg.pixel_size_nm)
        return simulate_aerial_image_hopkins(mask_continuous, params=params)
    raise ValueError(f"Unknown forward model: {cfg.forward_model}")


def _lambda_mrc_at(epoch: int, cfg: TrainConfig) -> float:
    """Linear warm-up from `lambda_mrc_warmup_start` to `lambda_mrc` over
    `lambda_mrc_warmup_epochs`, held thereafter."""
    if cfg.lambda_mrc == 0.0:
        return 0.0
    n = max(1, cfg.lambda_mrc_warmup_epochs)
    if epoch >= n:
        return cfg.lambda_mrc
    t = epoch / n
    return cfg.lambda_mrc_warmup_start + t * (cfg.lambda_mrc - cfg.lambda_mrc_warmup_start)


def _step(
    model: UNet,
    batch: tuple[torch.Tensor, torch.Tensor],
    cfg: TrainConfig,
    epoch: int,
    kernels: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    kernels_f: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    design, target_mask = batch
    design = design.to(cfg.device).unsqueeze(1)
    target_mask = target_mask.to(cfg.device).unsqueeze(1)

    logits = model(design)
    mask_continuous = torch.sigmoid(logits)

    bce = functional.binary_cross_entropy_with_logits(logits, target_mask)
    aerial = _forward(mask_continuous, cfg, kernels, weights, kernels_f)
    consistency = functional.mse_loss(aerial, design)

    if cfg.lambda_mrc > 0.0:
        mrc = curvilinear_mrc_loss(
            mask_continuous,
            min_width_nm=cfg.mrc_min_width_nm,
            min_spacing_nm=cfg.mrc_min_spacing_nm,
            pixel_size_nm=cfg.pixel_size_nm,
            weight_min_cd=1.0,
            weight_min_spacing=cfg.mrc_weight_min_spacing,
            weight_min_curvature=0.0,
        )
    else:
        mrc = torch.zeros((), device=design.device)

    lambda_mrc = _lambda_mrc_at(epoch, cfg)
    total = bce + cfg.consistency_weight * consistency + lambda_mrc * mrc
    return {
        "total": total,
        "bce": bce.detach(),
        "consistency": consistency.detach(),
        "mrc": mrc.detach(),
        "lambda_mrc": torch.tensor(lambda_mrc, device=design.device),
    }


def _bn_drift_log(
    model: torch.nn.Module,
    prev: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Per-epoch BatchNorm drift snapshot (v7 Bug H + v8 H3)."""
    eps = 1e-3
    max_dvar = 0.0
    max_dmean = 0.0
    flagged_var = 0
    flagged_mean = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d) and m.running_mean is not None:
            key = id(m)
            cur_mean = m.running_mean.detach().clone()
            cur_var = m.running_var.detach().clone()
            if key in prev:
                prev_mean, prev_var = prev[key]
                d_mean = float((cur_mean - prev_mean).abs().mean().item())
                d_var = float((cur_var - prev_var).abs().mean().item())
                mean_thresh = 0.5 * (1.0 + float(prev_mean.abs().mean().item()))
                var_thresh = 0.5 * max(float(prev_var.mean().item()), eps)
                if d_var > var_thresh:
                    flagged_var += 1
                if d_mean > mean_thresh:
                    flagged_mean += 1
                max_dvar = max(max_dvar, d_var)
                max_dmean = max(max_dmean, d_mean)
            prev[key] = (cur_mean, cur_var)
    return {
        "max_d_mean": max_dmean,
        "max_d_var": max_dvar,
        "flagged_layers_var": flagged_var,
        "flagged_layers_mean": flagged_mean,
    }


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(0)
    device = torch.device(cfg.device)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    dataset = _build_dataset(cfg)
    # num_workers=0 on MPS — multi-process loaders interact poorly with
    # the unified-memory model and the gain is marginal for 256×256.
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    # Pre-compute SOCS kernels + their FFT once per run (P2-a).
    kernels: torch.Tensor | None = None
    weights_t: torch.Tensor | None = None
    kernels_f: torch.Tensor | None = None
    grid_size: int | None = None
    if cfg.forward_model == "hopkins" and not cfg.smoke_test:
        grid_size = cfg.resize_to or 256
        params = HopkinsParams(num_kernels=cfg.num_kernels, pixel_size_nm=cfg.pixel_size_nm)
        kernels, weights_t = compute_socs_kernels(params, grid_size, device)
        kernels_c64 = kernels.to(torch.complex64)
        kernels_f = torch.fft.fft2(torch.fft.ifftshift(kernels_c64, dim=(-2, -1)))
        print(
            f"[setup] precomputed SOCS: K={cfg.num_kernels} grid={grid_size} "
            f"pixel_nm={cfg.pixel_size_nm}"
        )

    history: list[float] = []
    component_history: list[dict[str, float]] = []
    bn_drift_history: list[dict[str, float]] = []
    epochs = 1 if cfg.smoke_test else cfg.epochs
    prev_bn: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    first_step_logged = False

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        epoch_components: dict[str, list[float]] = {
            "bce": [],
            "consistency": [],
            "mrc": [],
        }
        for batch in loader:
            optimizer.zero_grad()
            losses = _step(model, batch, cfg, epoch, kernels, weights_t, kernels_f)
            total = losses["total"]
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {total.item()}")

            if not first_step_logged:
                bce_v = float(losses["bce"].item())
                cons_v = float(losses["consistency"].item())
                mrc_v = float(losses["mrc"].item())
                lam_v = float(losses["lambda_mrc"].item())
                print(
                    f"[step-1] bce={bce_v:.4f} consistency={cons_v:.4f} "
                    f"mrc={mrc_v:.4f} lambda_mrc={lam_v:.3f}"
                )
                # Sanity floor only — see plan v6 caveat.
                if bce_v > 0.0 and mrc_v >= 10.0 * bce_v and lam_v > 0.0:
                    print(
                        "[step-1] WARN: MRC penalty >= 10× BCE; consider "
                        "dropping --lambda-mrc-warmup-start to 0.01."
                    )
                first_step_logged = True

            total.backward()
            optimizer.step()
            epoch_losses.append(float(total.item()))
            epoch_components["bce"].append(float(losses["bce"].item()))
            epoch_components["consistency"].append(float(losses["consistency"].item()))
            epoch_components["mrc"].append(float(losses["mrc"].item()))
            if cfg.smoke_test:
                break
        scheduler.step()

        if device.type == "mps":
            torch.mps.empty_cache()

        bn_summary = _bn_drift_log(model, prev_bn)
        bn_drift_history.append(bn_summary)

        mean = sum(epoch_losses) / max(1, len(epoch_losses))
        history.append(mean)
        means = {k: sum(v) / max(1, len(v)) for k, v in epoch_components.items()}
        component_history.append(means)
        print(
            f"epoch {epoch}: loss={mean:.4f} bce={means['bce']:.4f} "
            f"consistency={means['consistency']:.4f} mrc={means['mrc']:.4f} "
            f"lambda_mrc={_lambda_mrc_at(epoch, cfg):.3f} "
            f"bn_max_dvar={bn_summary['max_d_var']:.3e}"
        )
        if cfg.smoke_test:
            break

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.output)

    metadata = {
        "config": cfg.to_dict(),
        "history": history,
        "component_history": component_history,
        "bn_drift_history": bn_drift_history,
        "final_loss": history[-1] if history else math.nan,
        "dataset": "ganopc",
        "paper": "Yang2018_GANOPC",
        "scope": "generator-only (no discriminator); see scripts/train_gan_opc.py docstring",
        "version": "v0.2",
        "v02_changes": [
            "B1: _forward Hopkins branch pixel_size_nm 2.0->8.0",
            "B2: _GanOpcPairs._resize bilinear+thresh -> mode='area'+thresh",
            "B3: _step returns dict; warm-up reads epoch; BN drift logged",
            "Change 1: curvilinear_mrc_loss term with lambda_mrc warm-up",
            "Change 2: --forward-model default flipped to hopkins; "
            "kernel_f precomputed via P2-a precomputed_kernels_f kwarg",
        ],
        "mrc_loss_min_width_nm_nominal": cfg.mrc_min_width_nm,
        "mrc_loss_actual_floor_nm": 16.0,
        "mrc_loss_actual_floor_note": (
            "radius=1 at 8 nm/px enforces 16 nm floor regardless of nominal "
            "min_width_nm in [16,23]; chose 24 for loss-checker radius parity"
        ),
        "bce_v01_v02_incomparable": True,
        "bce_v01_v02_note": (
            "B2 changed the target binarisation; absolute BCE values differ "
            "from v0.1. Eval-time PVB/MRC remain comparable."
        ),
        "dataset_resize_note": (
            "mode='area' antialiased downsample then >0.5 threshold; "
            "preserves areal mass before binarisation."
        ),
        "kernel_f_cache": cfg.forward_model == "hopkins" and not cfg.smoke_test,
        "lambda_mrc_schedule": {
            "start": cfg.lambda_mrc_warmup_start,
            "end": cfg.lambda_mrc,
            "warmup_epochs": cfg.lambda_mrc_warmup_epochs,
        },
        "reproducibility_note": (
            "torch.manual_seed(0); MPS kernel-launch nondeterminism "
            "limits bitwise reproducibility (~1e-5)."
        ),
    }
    metadata_path = cfg.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved checkpoint to {cfg.output}")
    print(f"Saved metadata to {metadata_path}")
    return metadata


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to ganopc-data/ (containing artitgt/ and artimsk/). Omit for dummy data.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sigma-px", type=float, default=4.5)
    p.add_argument(
        "--forward-model",
        default="hopkins",
        choices=["gaussian", "hopkins"],
        help="v0.2 default flipped to hopkins (Change 2). Pass gaussian to reproduce v0.1.",
    )
    p.add_argument("--device", default=_default_device())
    p.add_argument("--output", type=Path, default=Path("checkpoints/gan_opc_v0_2.pt"))
    p.add_argument(
        "--resize-to",
        type=int,
        default=256,
        help="Resize 2048×2048 PNGs to this side length. Paper §IV.A trains on 256.",
    )
    p.add_argument(
        "--consistency-weight",
        type=float,
        default=0.05,
        help="Weight on the forward-aerial-vs-design MSE term. v0.2 default 0.05 "
        "(was 0.1 for v0.1; Hopkins gradient is stronger than Gaussian).",
    )
    p.add_argument(
        "--num-kernels",
        type=int,
        default=24,
        help="SOCS kernel count for Hopkins forward model. Default 24 (LithoBench Table II); "
        "P1 sweep selects smaller N if energy spectrum tolerates it.",
    )
    p.add_argument(
        "--pixel-size-nm",
        type=float,
        default=8.0,
        help="Effective pixel pitch of the resized mask. 2048→256 of GAN-OPC's "
        "1 nm/px native gives 8 nm/px.",
    )
    p.add_argument(
        "--lambda-mrc",
        type=float,
        default=1.0,
        help="Final weight for the curvilinear_mrc_loss term (after warm-up). 0 disables.",
    )
    p.add_argument(
        "--lambda-mrc-warmup-epochs",
        type=int,
        default=5,
        help="Linear warm-up duration for lambda_mrc (epochs).",
    )
    p.add_argument(
        "--lambda-mrc-warmup-start",
        type=float,
        default=0.1,
        help="Initial lambda_mrc at epoch 0 of warm-up.",
    )
    p.add_argument(
        "--mrc-min-width-nm",
        type=float,
        default=24.0,
        help="MRC loss min-width target. v0.2 lock = 24 nm: only value where "
        "loss-radius == checker-radius == 1 at 8 nm/px (plan v6).",
    )
    p.add_argument(
        "--mrc-min-spacing-nm",
        type=float,
        default=24.0,
        help="MRC loss min-spacing target. Match width for radius parity.",
    )
    p.add_argument(
        "--mrc-weight-min-spacing",
        type=float,
        default=0.5,
        help="Weight on the min-spacing MRC term. v0.2 P0-probe finding: ~30% of "
        "v0.1 violations are spacing — enable a non-zero weight to address them.",
    )
    p.add_argument("--smoke-test", action="store_true", help="Single batch then exit.")
    args = p.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sigma_px=args.sigma_px,
        forward_model=args.forward_model,
        device=args.device,
        output=args.output,
        resize_to=args.resize_to,
        consistency_weight=args.consistency_weight,
        smoke_test=args.smoke_test,
        num_kernels=args.num_kernels,
        pixel_size_nm=args.pixel_size_nm,
        lambda_mrc=args.lambda_mrc,
        lambda_mrc_warmup_epochs=args.lambda_mrc_warmup_epochs,
        lambda_mrc_warmup_start=args.lambda_mrc_warmup_start,
        mrc_min_width_nm=args.mrc_min_width_nm,
        mrc_min_spacing_nm=args.mrc_min_spacing_nm,
        mrc_weight_min_spacing=args.mrc_weight_min_spacing,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
