"""Reference training script for the GAN-OPC generator baseline.

This script trains the U-Net used by `openlithohub.models.gan_opc`. It
mirrors `scripts/train_neural_ilt.py` — same UNet, same BCE +
forward-consistency loss formulation — but ingests the GAN-OPC paired-PNG
dataset (`Yang2018_GANOPC`) instead of LithoBench.

Scope (v0.3, 2026-05-23 follow-up to v0.2):

  v0.2 changed four things at once and three of them plausibly affected
  PVB. See ``out/plans/gan-opc-v0.3-improvements.md`` §1 for the full
  retrospective. v0.3 isolates one variable at a time via four ablation
  runs (D, A, B, C) defined below.

  Run-level config matrix (from plan §2.4):

    Run | px (nm) | resize mode | cons. wt | fwd model | MRC term | PVB term
    ----|---------|-------------|----------|-----------|----------|---------
    A   | 4.0     | bilinear    | 0.1      | gaussian  | yes      | yes
    B   | 4.0     | bilinear    | 0.1      | gaussian  | yes      | no
    C   | 4.0     | bilinear    | 0.1      | gaussian  | no       | yes
    D   | 8.0     | bilinear    | 0.1      | gaussian  | no       | no  (v0.1 replay; manual_seed(1))

  At px=4.0 train resolution is 512×512 (4× compute vs. v0.2). At px=8.0
  it stays 256×256 (matches v0.1 / v0.2).

Usage:
    # Smoke test (one batch, no real training).
    python scripts/train_gan_opc.py --smoke-test

    # Run D (v0.1 replay; falsification test):
    python scripts/train_gan_opc.py \\
        --data-root data/ganopc/extracted/ganopc-data \\
        --resize-to 256 --pixel-size-nm 8.0 \\
        --resize-mode bilinear --consistency-weight 0.1 \\
        --forward-model gaussian \\
        --lambda-mrc 0.0 --lambda-pvb 0.0 \\
        --epochs 50 --batch-size 8 --device mps \\
        --seed 1 \\
        --output checkpoints/gan_opc_v0_3_d.pt

    # Run A (v0.3 candidate):
    python scripts/train_gan_opc.py \\
        --data-root data/ganopc/extracted/ganopc-data \\
        --resize-to 512 --pixel-size-nm 4.0 \\
        --resize-mode bilinear --consistency-weight 0.1 \\
        --forward-model gaussian \\
        --lambda-mrc 0.5 --mrc-min-width-nm 20 --mrc-min-spacing-nm 20 \\
        --lambda-pvb 0.1 \\
        --epochs 50 --batch-size 8 --device mps \\
        --seed 0 \\
        --output checkpoints/gan_opc_v0_3_a.pt

  B and C are A with --lambda-pvb 0.0 / --lambda-mrc 0.0 respectively.
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
    resize_mode: str
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
    lambda_pvb: float
    lambda_pvb_warmup_epochs: int
    lambda_pvb_dose_delta: float
    lambda_pvb_defocus_range_nm: float
    lambda_pvb_sigma_nominal: float
    seed: int

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
    training loop fits in unified memory on consumer Apple Silicon. Resize
    mode is selectable: ``bilinear`` (v0.1, v0.3) or ``area`` (v0.2; kept
    for ablation reproducibility, not recommended — see plan §1.3 Cause 1).
    """

    def __init__(self, root: Path, resize_to: int | None, resize_mode: str) -> None:
        self.inner = GanOpcDataset(root=root)
        self.resize_to = resize_to
        if resize_mode not in {"bilinear", "area"}:
            raise ValueError(f"resize_mode must be bilinear|area, got {resize_mode!r}")
        self.resize_mode = resize_mode

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.inner[idx]
        assert sample.mask is not None
        design = sample.design.float()
        mask = sample.mask.float()
        if self.resize_to is not None and design.shape[-1] != self.resize_to:
            design = self._resize(design, self.resize_to, self.resize_mode)
            mask = self._resize(mask, self.resize_to, self.resize_mode)
        return design, mask

    @staticmethod
    def _resize(t: torch.Tensor, target: int, mode: str) -> torch.Tensor:
        t4 = t.unsqueeze(0).unsqueeze(0)
        if mode == "bilinear":
            resized = functional.interpolate(
                t4, size=(target, target), mode="bilinear", align_corners=False
            )
        else:
            resized = functional.interpolate(t4, size=(target, target), mode="area")
        return (resized.squeeze(0).squeeze(0) > 0.5).float()


def _build_dataset(cfg: TrainConfig) -> Dataset:
    if cfg.smoke_test or cfg.data_root is None:
        return _DummyPairs(n=max(cfg.batch_size, 4), size=64)
    return _GanOpcPairs(cfg.data_root, resize_to=cfg.resize_to, resize_mode=cfg.resize_mode)


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
        if kernels is not None and weights is not None:
            return simulate_aerial_image_hopkins(
                mask_continuous,
                kernels=kernels,
                weights=weights,
                precomputed_kernels_f=kernels_f,
            )
        params = HopkinsParams(num_kernels=cfg.num_kernels, pixel_size_nm=cfg.pixel_size_nm)
        return simulate_aerial_image_hopkins(mask_continuous, params=params)
    raise ValueError(f"Unknown forward model: {cfg.forward_model}")


def _pvb_loss(
    mask_continuous: torch.Tensor,
    target_design: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """4-corner (dose × defocus) printability term — v0.3 Change 5.

    Mirrors the eval-time ``compute_pvband`` (pvband.py:_gaussian_pw_envelopes)
    construction verbatim:
        sigma_def = defocus_range_nm / (2 * pixel_size_nm)
        sigma_hi  = sigma_nom + sigma_def
        sigma_lo  = max(0.5, sigma_nom - sigma_def * 0.5)  # asymmetric
    Four corners: (dose × defocus) ∈ {(1+δ)·{hi,lo}, (1-δ)·{hi,lo}}.

    Returns mean MSE across corners against ``target_design``. Same MSE
    scale as the consistency term so λ_pvb is comparable to consistency_weight.
    """
    delta = cfg.lambda_pvb_dose_delta
    sigma_nom = cfg.lambda_pvb_sigma_nominal
    sigma_def = cfg.lambda_pvb_defocus_range_nm / (2.0 * cfg.pixel_size_nm)
    sigma_hi = sigma_nom + sigma_def
    sigma_lo = max(0.5, sigma_nom - sigma_def * 0.5)

    corners = (
        (1.0 + delta, sigma_hi),
        (1.0 + delta, sigma_lo),
        (1.0 - delta, sigma_hi),
        (1.0 - delta, sigma_lo),
    )
    total = mask_continuous.new_zeros(())
    for dose, sigma in corners:
        aerial = simulate_aerial_image(mask_continuous, sigma_px=sigma, dose=dose)
        total = total + functional.mse_loss(aerial, target_design)
    return total / float(len(corners))


def _lambda_mrc_at(epoch: int, cfg: TrainConfig) -> float:
    if cfg.lambda_mrc == 0.0:
        return 0.0
    n = max(1, cfg.lambda_mrc_warmup_epochs)
    if epoch >= n:
        return cfg.lambda_mrc
    t = epoch / n
    return cfg.lambda_mrc_warmup_start + t * (cfg.lambda_mrc - cfg.lambda_mrc_warmup_start)


def _lambda_pvb_at(epoch: int, cfg: TrainConfig) -> float:
    if cfg.lambda_pvb == 0.0:
        return 0.0
    n = max(1, cfg.lambda_pvb_warmup_epochs)
    if epoch >= n:
        return cfg.lambda_pvb
    return (epoch / n) * cfg.lambda_pvb


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

    if cfg.lambda_pvb > 0.0:
        pvb = _pvb_loss(mask_continuous, design, cfg)
    else:
        pvb = torch.zeros((), device=design.device)

    lambda_mrc = _lambda_mrc_at(epoch, cfg)
    lambda_pvb = _lambda_pvb_at(epoch, cfg)
    total = (
        bce
        + cfg.consistency_weight * consistency
        + lambda_mrc * mrc
        + lambda_pvb * pvb
    )
    return {
        "total": total,
        "bce": bce.detach(),
        "consistency": consistency.detach(),
        "mrc": mrc.detach(),
        "pvb": pvb.detach(),
        "lambda_mrc": torch.tensor(lambda_mrc, device=design.device),
        "lambda_pvb": torch.tensor(lambda_pvb, device=design.device),
        "mask_mean": mask_continuous.detach().mean(),
    }


def _bn_drift_log(
    model: torch.nn.Module,
    prev: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
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


# v0.2 baseline bn_max_d_var per epoch (peaks ~10.38, decays to 1.24).
# Hard-coded so v0.3 abort guard fires at >1.5× v0.2 baseline at the same epoch.
# Source: out/baselines/iccad16/gan-opc-v0.2.json bn_drift_history (or
# gan_opc_v0_2.metadata.json:bn_drift_history). Conservative fill (10.38) for
# epoch 0 if the array is shorter than queried.
_V02_BN_BASELINE: tuple[float, ...] = (
    9.66, 10.38, 8.5, 7.0, 6.0, 5.04, 4.5, 4.2, 4.0, 3.8,
    3.6, 3.4, 3.3, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5,
    2.4, 2.3, 2.2, 2.1, 2.0, 1.95, 1.9, 1.85, 1.8, 1.75,
    1.7, 1.65, 1.6, 1.55, 1.5, 1.46, 1.42, 1.39, 1.36, 1.34,
    1.32, 1.31, 1.30, 1.29, 1.28, 1.27, 1.26, 1.25, 1.24, 1.24,
)


def _v02_bn_baseline(epoch: int) -> float:
    if epoch < 0:
        return _V02_BN_BASELINE[0]
    if epoch >= len(_V02_BN_BASELINE):
        return _V02_BN_BASELINE[-1]
    return _V02_BN_BASELINE[epoch]


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    dataset = _build_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

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
    mask_mean_history: list[float] = []
    epochs = 1 if cfg.smoke_test else cfg.epochs
    prev_bn: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    first_step_logged = False
    v01_mask_mean: float | None = None  # captured at first step; collapse guard reference
    plateau_streak = 0
    last_loss = math.inf

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        epoch_components: dict[str, list[float]] = {
            "bce": [],
            "consistency": [],
            "mrc": [],
            "pvb": [],
        }
        epoch_mask_means: list[float] = []
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
                pvb_v = float(losses["pvb"].item())
                lam_mrc = float(losses["lambda_mrc"].item())
                lam_pvb = float(losses["lambda_pvb"].item())
                print(
                    f"[step-1] bce={bce_v:.4f} consistency={cons_v:.4f} "
                    f"mrc={mrc_v:.4f} pvb={pvb_v:.4f} "
                    f"lambda_mrc={lam_mrc:.3f} lambda_pvb={lam_pvb:.3f}"
                )
                if pvb_v > 5.0 * max(bce_v, 1e-6) and cfg.lambda_pvb > 0.0:
                    print(
                        "[step-1] WARN: PVB step-1 > 5× BCE; consider lowering --lambda-pvb."
                    )
                v01_mask_mean = float(losses["mask_mean"].item())
                first_step_logged = True

            total.backward()
            optimizer.step()
            epoch_losses.append(float(total.item()))
            epoch_components["bce"].append(float(losses["bce"].item()))
            epoch_components["consistency"].append(float(losses["consistency"].item()))
            epoch_components["mrc"].append(float(losses["mrc"].item()))
            epoch_components["pvb"].append(float(losses["pvb"].item()))
            epoch_mask_means.append(float(losses["mask_mean"].item()))
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
        mask_mean_epoch = sum(epoch_mask_means) / max(1, len(epoch_mask_means))
        mask_mean_history.append(mask_mean_epoch)
        print(
            f"epoch {epoch}: loss={mean:.4f} bce={means['bce']:.4f} "
            f"consistency={means['consistency']:.4f} mrc={means['mrc']:.4f} "
            f"pvb={means['pvb']:.4f} "
            f"lambda_mrc={_lambda_mrc_at(epoch, cfg):.3f} "
            f"lambda_pvb={_lambda_pvb_at(epoch, cfg):.3f} "
            f"mask_mean={mask_mean_epoch:.4f} "
            f"bn_max_dvar={bn_summary['max_d_var']:.3e}"
        )

        # Stopping rules (plan §2.6).
        # 1. BN-drift relative to v0.2 baseline.
        if epoch >= 1 and len(bn_drift_history) >= 2 and not cfg.smoke_test:
            cur = bn_summary["max_d_var"]
            prev_bn_v = bn_drift_history[-2]["max_d_var"]
            base_cur = _v02_bn_baseline(epoch)
            base_prev = _v02_bn_baseline(epoch - 1)
            if cur > 1.5 * base_cur and prev_bn_v > 1.5 * base_prev:
                raise RuntimeError(
                    f"BN drift escalation: epoch {epoch} max_d_var={cur:.3f} > "
                    f"1.5×v0.2_baseline={1.5*base_cur:.3f} for two consecutive epochs."
                )
        # 2. Mask-mean collapse guard.
        if v01_mask_mean is not None and mask_mean_epoch < 0.1 * v01_mask_mean and not cfg.smoke_test:
            raise RuntimeError(
                f"Mask-mean collapsed at epoch {epoch}: "
                f"{mask_mean_epoch:.4f} < 0.1 * step1_mean={v01_mask_mean:.4f}. "
                "PVB term may be mis-constructed."
            )
        # 3. Total-loss plateau after warm-up.
        if epoch >= 15 and not cfg.smoke_test:
            if mean >= last_loss - 1e-6:
                plateau_streak += 1
            else:
                plateau_streak = 0
            if plateau_streak >= 5:
                raise RuntimeError(
                    f"Loss plateau: no decrease for 5 consecutive epochs after warm-up "
                    f"(epoch {epoch}, last_loss={last_loss:.4f})."
                )
        last_loss = mean

        if cfg.smoke_test:
            break

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.output)

    metadata = {
        "config": cfg.to_dict(),
        "history": history,
        "component_history": component_history,
        "bn_drift_history": bn_drift_history,
        "mask_mean_history": mask_mean_history,
        "final_loss": history[-1] if history else math.nan,
        "dataset": "ganopc",
        "paper": "Yang2018_GANOPC",
        "scope": "generator-only (no discriminator); see scripts/train_gan_opc.py docstring",
        "version": "v0.3",
        "v03_changes": [
            "Change 1: resize mode bilinear+thresh (revert v0.2 area)",
            "Change 2: consistency_weight 0.1 (revert v0.2 0.05)",
            "Change 3: forward model gaussian default (revert v0.2 hopkins)",
            "Change 4: MRC term at px=4, w=20 (radius-2 parity); lambda_mrc=0.5",
            "Change 5: 4-corner (dose × defocus) MSE PVB regulariser; lambda_pvb=0.1",
        ],
        "lambda_mrc_schedule": {
            "start": cfg.lambda_mrc_warmup_start,
            "end": cfg.lambda_mrc,
            "warmup_epochs": cfg.lambda_mrc_warmup_epochs,
        },
        "lambda_pvb_schedule": {
            "start": 0.0,
            "end": cfg.lambda_pvb,
            "warmup_epochs": cfg.lambda_pvb_warmup_epochs,
            "dose_delta": cfg.lambda_pvb_dose_delta,
            "defocus_range_nm": cfg.lambda_pvb_defocus_range_nm,
            "sigma_nominal": cfg.lambda_pvb_sigma_nominal,
        },
        "reproducibility_note": (
            f"torch.manual_seed({cfg.seed}); MPS kernel-launch nondeterminism "
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
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sigma-px", type=float, default=4.5)
    p.add_argument(
        "--forward-model",
        default="gaussian",
        choices=["gaussian", "hopkins"],
        help="v0.3 default: gaussian (Hopkins reverted per Change 3).",
    )
    p.add_argument("--device", default=_default_device())
    p.add_argument("--output", type=Path, default=Path("checkpoints/gan_opc_v0_3.pt"))
    p.add_argument(
        "--resize-to",
        type=int,
        default=512,
        help="Resize side. v0.3 default 512 (px=4.0). v0.1/D use 256 (px=8.0).",
    )
    p.add_argument(
        "--resize-mode",
        default="bilinear",
        choices=["bilinear", "area"],
        help="v0.3 default: bilinear+thresh (Change 1, revert v0.2 area).",
    )
    p.add_argument(
        "--consistency-weight",
        type=float,
        default=0.1,
        help="v0.3 default 0.1 (Change 2, revert v0.2 0.05).",
    )
    p.add_argument("--num-kernels", type=int, default=24)
    p.add_argument(
        "--pixel-size-nm",
        type=float,
        default=4.0,
        help="v0.3 default: 4.0 (Change 4 — eval-aligned). D uses 8.0.",
    )
    p.add_argument(
        "--lambda-mrc",
        type=float,
        default=0.5,
        help="v0.3 final lambda_mrc (warm-up to this over 10 epochs). 0 disables.",
    )
    p.add_argument("--lambda-mrc-warmup-epochs", type=int, default=10)
    p.add_argument("--lambda-mrc-warmup-start", type=float, default=0.0)
    p.add_argument(
        "--mrc-min-width-nm",
        type=float,
        default=20.0,
        help="v0.3 default 20 nm (radius-2 parity at px=4).",
    )
    p.add_argument("--mrc-min-spacing-nm", type=float, default=20.0)
    p.add_argument("--mrc-weight-min-spacing", type=float, default=0.5)
    p.add_argument(
        "--lambda-pvb",
        type=float,
        default=0.1,
        help="v0.3 final lambda_pvb (warm-up over 15 epochs). 0 disables.",
    )
    p.add_argument("--lambda-pvb-warmup-epochs", type=int, default=15)
    p.add_argument(
        "--lambda-pvb-dose-delta",
        type=float,
        default=0.05,
        help="Δ for dose perturbation; matches pvband.py default 0.05.",
    )
    p.add_argument(
        "--lambda-pvb-defocus-range-nm",
        type=float,
        default=20.0,
        help="defocus_range_nm for sigma_def; matches pvband.py default 20.0.",
    )
    p.add_argument("--lambda-pvb-sigma-nominal", type=float, default=2.0)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="torch.manual_seed. Run D requires --seed 1 (independent re-sample).",
    )
    p.add_argument("--smoke-test", action="store_true")
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
        resize_mode=args.resize_mode,
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
        lambda_pvb=args.lambda_pvb,
        lambda_pvb_warmup_epochs=args.lambda_pvb_warmup_epochs,
        lambda_pvb_dose_delta=args.lambda_pvb_dose_delta,
        lambda_pvb_defocus_range_nm=args.lambda_pvb_defocus_range_nm,
        lambda_pvb_sigma_nominal=args.lambda_pvb_sigma_nominal,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
