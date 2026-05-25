"""Reference training script for the GAN-OPC generator baseline.

This script trains the U-Net used by `openlithohub.models.gan_opc`. It
mirrors `scripts/train_neural_ilt.py` — same UNet, same BCE +
forward-consistency loss formulation — but ingests the GAN-OPC paired-PNG
dataset (`Yang2018_GANOPC`) instead of LithoBench.

Scope (v0.4, 2026-05-25):

  v0.3 ablation revealed that the PVB regularizer was structurally
  misaligned with the eval metric (MSE vs design vs envelope bandwidth).
  v0.4 fixes this by directly minimizing 4-corner envelope bandwidth
  using `differentiable_threshold` (sigmoid) instead of hard thresholding.

  Key changes from v0.3:
  1. PVB loss → bandwidth loss (metric-aligned, R2 core fix)
  2. Gradient accumulation support
  3. Mixed precision (AMP) support
  4. Memmap-backed dataset cache
  5. Thread control (OMP/MKL/OPENBLAS)
  6. UNetV2 (4-level, 64→512) architecture option
  7. Plateau early-stopping with best-checkpoint rollback
  8. BN-drift sliding baseline for >50 epoch runs

Usage:
    # Smoke test (one batch, no real training).
    python scripts/train_gan_opc.py --smoke-test

    # Stage 0 sanity check (256², 10 epochs):
    python scripts/train_gan_opc.py \\
        --data-root data/ganopc/extracted/ganopc-data \\
        --resize-to 256 --pixel-size-nm 8.0 \\
        --resize-mode bilinear --consistency-weight 0.1 \\
        --forward-model gaussian \\
        --lambda-mrc 0.5 --lambda-pvb 0.1 \\
        --epochs 10 --batch-size 4 --device cpu \\
        --seed 0 --mixed-precision \\
        --output checkpoints/gan_opc_v0_4_stage0.pt

    # Stage 1 Run E (v0.4-aerial-pvb):
    python scripts/train_gan_opc.py \\
        --data-root data/ganopc/extracted/ganopc-data \\
        --resize-to 512 --pixel-size-nm 4.0 \\
        --resize-mode bilinear --consistency-weight 0.1 \\
        --forward-model gaussian \\
        --lambda-mrc 0.5 --mrc-min-width-nm 20 --mrc-min-spacing-nm 20 \\
        --lambda-pvb 0.1 \\
        --epochs 50 --batch-size 4 --gradient-accumulation 2 \\
        --device cpu --seed 0 \\
        --output checkpoints/gan_opc_v0_4_e.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.hopkins import (
    HopkinsParams,
    compute_socs_kernels,
    simulate_aerial_image_hopkins,
)
from openlithohub._utils.resist_model import differentiable_threshold
from openlithohub.benchmark.metrics.mrc_loss import curvilinear_mrc_loss
from openlithohub.data.ganopc import GanOpcDataset
from openlithohub.models._unet import UNet, UNetV2

# Thread control (R3+R4): limit OMP/MKL/OPENBLAS threads to avoid contention.
# Must be set before torch imports take effect in worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")


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
    pvb_steepness: float
    seed: int
    gradient_accumulation: int
    mixed_precision: bool
    arch: str
    plateau_patience: int
    num_workers: int
    cache_dir: str

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
    training loop fits in unified memory on consumer hardware. Resize
    mode is selectable: ``bilinear`` (v0.1, v0.3) or ``area`` (v0.2).
    """

    def __init__(self, root: Path, resize_to: int | None, resize_mode: str) -> None:
        if resize_mode not in {"bilinear", "area"}:
            raise ValueError(f"resize_mode must be bilinear|area, got {resize_mode!r}")
        self.inner = GanOpcDataset(root=root)
        self.resize_to = resize_to
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


class _MemmapGanOpcPairs(Dataset):
    """Single-file memmap-backed dataset cache (R3 Q7).

    Stores the entire preprocessed dataset as a single NumPy memmap file,
    avoiding the I/O bottleneck of thousands of small .pt files.

    4875 samples × 2 (design+mask) × H² × 2 bytes (float16) ≈ 5.1 GB at 512².
    Uses numpy.memmap for on-demand loading, not occupying full RAM.
    Multi-worker DataLoader can safely read the same memmap file concurrently.
    """

    def __init__(
        self,
        root: Path,
        resize_to: int | None,
        resize_mode: str,
        cache_dir: str = "cache/ganopc/",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.inner_n = len(GanOpcDataset(root=root))
        self.resize_to = resize_to or 2048
        self.shape = (self.inner_n, 2, self.resize_to, self.resize_to)
        tag = f"ganopc_{self.resize_to}_{resize_mode}"
        self.path = self.cache_dir / f"{tag}.memmap"
        self._ensure_cache(root, resize_to, resize_mode)
        self._mmap: np.memmap | None = None

    def _ensure_cache(self, root: Path, resize_to: int | None, resize_mode: str) -> None:
        if self.path.exists():
            return
        print(f"[memmap] Building cache at {self.path} ({self.inner_n} samples)...")
        t0 = time.time()
        inner = _GanOpcPairs(root, resize_to, resize_mode)
        arr = np.memmap(self.path, dtype=np.float16, mode="w+", shape=self.shape)
        from concurrent.futures import ProcessPoolExecutor

        def _process(i: int) -> tuple[int, np.ndarray, np.ndarray]:
            design, mask = inner[i]
            return i, design.numpy().astype(np.float16), mask.numpy().astype(np.float16)

        with ProcessPoolExecutor(max_workers=8) as pool:
            for i, design, mask in pool.map(_process, range(self.inner_n)):
                arr[i, 0] = design
                arr[i, 1] = mask
        arr.flush()
        del arr
        elapsed = time.time() - t0
        print(f"[memmap] Cache built in {elapsed:.1f}s ({self.path.stat().st_size / 1e9:.2f} GB)")

    def __len__(self) -> int:
        return self.inner_n

    def _get_memmap(self) -> np.memmap:
        """Lazy singleton: open memmap once per worker, reuse across calls."""
        if self._mmap is None:
            self._mmap = np.memmap(self.path, dtype=np.float16, mode="r", shape=self.shape)
        return self._mmap

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        arr = self._get_memmap()
        design = torch.from_numpy(arr[idx, 0].astype(np.float32))
        mask = torch.from_numpy(arr[idx, 1].astype(np.float32))
        return design, mask


def _build_dataset(cfg: TrainConfig) -> Dataset:
    if cfg.smoke_test or cfg.data_root is None:
        return _DummyPairs(n=max(cfg.batch_size, 4), size=64)
    if cfg.cache_dir:
        return _MemmapGanOpcPairs(
            cfg.data_root,
            resize_to=cfg.resize_to,
            resize_mode=cfg.resize_mode,
            cache_dir=cfg.cache_dir,
        )
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


def _pvb_bandwidth_loss(
    mask_continuous: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """Metric-aligned bandwidth loss — structurally aligned with eval metric.

    Mirrors pvband.py:_gaussian_pw_envelopes construction verbatim,
    but replaces apply_resist_threshold (hard, non-diff) with
    differentiable_threshold (sigmoid, diff) to preserve gradients.

    Minimizes mean(outer_envelope - inner_envelope) across 4 corners.
    No reference target needed — pure self-robustness measure.
    """
    delta = cfg.lambda_pvb_dose_delta
    sigma_nom = cfg.lambda_pvb_sigma_nominal
    sigma_def = cfg.lambda_pvb_defocus_range_nm / (2.0 * cfg.pixel_size_nm)
    sigma_hi = sigma_nom + sigma_def
    sigma_lo = max(0.5, sigma_nom - sigma_def * 0.5)
    steepness = cfg.pvb_steepness

    corners = [
        (1.0 + delta, sigma_hi),
        (1.0 + delta, sigma_lo),
        (1.0 - delta, sigma_hi),
        (1.0 - delta, sigma_lo),
    ]

    outer = mask_continuous.new_zeros(mask_continuous.shape)
    inner = mask_continuous.new_ones(mask_continuous.shape)
    for dose, sigma in corners:
        aerial = simulate_aerial_image(mask_continuous, sigma_px=sigma, dose=dose)
        resist = differentiable_threshold(aerial, threshold=0.5, steepness=steepness)
        outer = torch.maximum(outer, resist)
        inner = torch.minimum(inner, resist)

    bandwidth = (outer - inner).clamp(min=0.0)
    return bandwidth.mean()


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


def _compute_loss(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    cfg: TrainConfig,
    epoch: int,
    kernels: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    kernels_f: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute loss and return (total, metrics_dict). No backward pass."""
    design, target_mask = batch
    design = design.to(cfg.device).unsqueeze(1)
    target_mask = target_mask.to(cfg.device).unsqueeze(1)

    logits = model(design)
    mask_continuous = torch.sigmoid(logits)

    # Ensure float32 for loss computations that don't support half precision
    mask_continuous = mask_continuous.float()
    logits_for_bce = logits.float()

    bce = functional.binary_cross_entropy_with_logits(logits_for_bce, target_mask.float())
    aerial = _forward(mask_continuous, cfg, kernels, weights, kernels_f)
    consistency = functional.mse_loss(aerial, design.float())

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
        pvb = _pvb_bandwidth_loss(mask_continuous, cfg)
    else:
        pvb = torch.zeros((), device=design.device)

    lambda_mrc = _lambda_mrc_at(epoch, cfg)
    lambda_pvb = _lambda_pvb_at(epoch, cfg)
    total = bce + cfg.consistency_weight * consistency + lambda_mrc * mrc + lambda_pvb * pvb

    metrics = {
        "total": total.detach(),
        "bce": bce.detach(),
        "consistency": consistency.detach(),
        "mrc": mrc.detach(),
        "pvb": pvb.detach(),
        "lambda_mrc": torch.tensor(lambda_mrc, device=design.device),
        "lambda_pvb": torch.tensor(lambda_pvb, device=design.device),
        "mask_mean": mask_continuous.detach().mean(),
        "target_mean": target_mask.detach().mean(),
    }
    return total, metrics


class _NullContext:
    """Minimal no-op context manager for when AMP is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


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


def _build_model(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.arch == "unetv2":
        return UNetV2()
    return UNet()


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    if cfg.device == "cpu":
        torch.set_num_threads(6)
    device = torch.device(cfg.device)
    model = _build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    dataset = _build_dataset(cfg)
    num_workers = cfg.num_workers if not cfg.smoke_test else 0
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    kernels: torch.Tensor | None = None
    weights_t: torch.Tensor | None = None
    kernels_f: torch.Tensor | None = None
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

    use_amp = cfg.mixed_precision and cfg.device == "cpu"
    scaler: torch.amp.GradScaler | None = None
    if use_amp:
        scaler = torch.amp.GradScaler("cpu")

    history: list[float] = []
    component_history: list[dict[str, float]] = []
    bn_drift_history: list[dict[str, float]] = []
    mask_mean_history: list[float] = []
    epochs = 1 if cfg.smoke_test else cfg.epochs
    prev_bn: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    first_step_logged = False
    plateau_streak = 0
    last_loss = math.inf
    best_loss = math.inf
    best_state: dict | None = None

    # Sliding baseline for BN-drift guard in long runs (R4).
    bn_sliding_window: list[float] = []

    epoch_time_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_losses: list[float] = []
        epoch_components: dict[str, list[float]] = {
            "bce": [],
            "consistency": [],
            "mrc": [],
            "pvb": [],
        }
        epoch_mask_means: list[float] = []
        epoch_target_means: list[float] = []
        optimizer.zero_grad()

        accum_steps = max(1, cfg.gradient_accumulation)

        for step_idx, batch in enumerate(loader):
            use_amp = cfg.mixed_precision and cfg.device == "cpu"
            amp_ctx = torch.amp.autocast("cpu", enabled=use_amp)

            with amp_ctx:
                total, losses = _compute_loss(
                    model, batch, cfg, epoch, kernels, weights_t, kernels_f
                )
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {total.item()}")

            if not first_step_logged:
                bce_v = float(losses["bce"].item())
                cons_v = float(losses["consistency"].item())
                mrc_v = float(losses["mrc"].item())
                pvb_v = float(losses["pvb"].item())
                lam_mrc = float(losses["lambda_mrc"].item())
                lam_pvb = float(losses["lambda_pvb"].item())
                tgt_mean = float(losses["target_mean"].item())
                print(
                    f"[step-1] bce={bce_v:.4f} consistency={cons_v:.4f} "
                    f"mrc={mrc_v:.4f} pvb={pvb_v:.4f} "
                    f"lambda_mrc={lam_mrc:.3f} lambda_pvb={lam_pvb:.3f} "
                    f"target_mean={tgt_mean:.4f}"
                )
                if pvb_v > 5.0 * max(bce_v, 1e-6) and cfg.lambda_pvb > 0.0:
                    print("[step-1] WARN: PVB step-1 > 5× BCE; consider lowering --lambda-pvb.")
                first_step_logged = True

            # Backward with optional gradient scaling
            scaled_loss = total / accum_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Step optimizer every accum_steps
            if (step_idx + 1) % accum_steps == 0 or (cfg.smoke_test and step_idx == 0):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            epoch_losses.append(float(losses["total"].item()))
            epoch_components["bce"].append(float(losses["bce"].item()))
            epoch_components["consistency"].append(float(losses["consistency"].item()))
            epoch_components["mrc"].append(float(losses["mrc"].item()))
            epoch_components["pvb"].append(float(losses["pvb"].item()))
            epoch_mask_means.append(float(losses["mask_mean"].item()))
            epoch_target_means.append(float(losses["target_mean"].item()))
            if cfg.smoke_test:
                break

        # Handle remaining accumulated gradients
        if not cfg.smoke_test and (len(loader) % accum_steps) != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        if device.type == "mps":
            torch.mps.empty_cache()

        bn_summary = _bn_drift_log(model, prev_bn)
        bn_drift_history.append(bn_summary)

        mean = sum(epoch_losses) / max(1, len(epoch_losses))
        history.append(mean)
        means = {k: sum(v) / max(1, len(v)) for k, v in epoch_components.items()}
        mask_mean_epoch = sum(epoch_mask_means) / max(1, len(epoch_mask_means))
        mask_mean_history.append(mask_mean_epoch)
        target_mean_epoch = sum(epoch_target_means) / max(1, len(epoch_target_means))

        epoch_elapsed = time.time() - epoch_start
        print(
            f"epoch {epoch}: loss={mean:.4f} bce={means['bce']:.4f} "
            f"consistency={means['consistency']:.4f} mrc={means['mrc']:.4f} "
            f"pvb={means['pvb']:.4f} "
            f"lambda_mrc={_lambda_mrc_at(epoch, cfg):.3f} "
            f"lambda_pvb={_lambda_pvb_at(epoch, cfg):.3f} "
            f"mask_mean={mask_mean_epoch:.4f} target_mean={target_mean_epoch:.4f} "
            f"bn_max_dvar={bn_summary['max_d_var']:.3e} "
            f"time={epoch_elapsed:.1f}s"
        )

        # Best state tracking (R4 split: plateau → 3 subtasks).
        if mean < best_loss:
            best_loss = mean
            best_state = copy.deepcopy(model.state_dict())

        # BN-drift guard.
        if (
            epoch >= 2
            and len(history) >= 2
            and len(bn_drift_history) >= 2
            and not cfg.smoke_test
            and cfg.pixel_size_nm <= 5.0
        ):
            cur = bn_summary["max_d_var"]
            prev_bn_v = bn_drift_history[-2]["max_d_var"]

            # R4: Sliding baseline for >50 epoch runs.
            if cfg.epochs > 50 and epoch >= 50:
                bn_sliding_window.append(cur)
                if len(bn_sliding_window) >= 10:
                    sliding_avg = sum(bn_sliding_window[-10:]) / 10.0
                    bn_threshold = 1.5 * sliding_avg
                else:
                    bn_threshold = 1.5 * cur  # not enough data yet, be lenient
                bn_escalating = cur > bn_threshold and prev_bn_v > bn_threshold
            else:
                base_cur = _v02_bn_baseline(epoch)
                base_prev = _v02_bn_baseline(epoch - 1)
                bn_escalating = cur > 1.5 * base_cur and prev_bn_v > 1.5 * base_prev

            loss_stalled = history[-1] >= history[-2]
            mask_drift = (
                target_mean_epoch > 0.0
                and abs(mask_mean_epoch - target_mean_epoch) > 0.5 * target_mean_epoch
            )
            if bn_escalating and (loss_stalled or mask_drift):
                raise RuntimeError(
                    f"BN drift escalation + divergence signal at epoch {epoch}: "
                    f"max_d_var={cur:.3f}, "
                    f"loss_stalled={loss_stalled} mask_drift={mask_drift}."
                )

        # Mask-mean collapse guard.
        if (
            epoch >= 5
            and target_mean_epoch > 0.0
            and mask_mean_epoch < 0.1 * target_mean_epoch
            and not cfg.smoke_test
        ):
            raise RuntimeError(
                f"Mask-mean collapsed at epoch {epoch}: "
                f"{mask_mean_epoch:.4f} < 0.1 * target_mean={target_mean_epoch:.4f}. "
                "PVB term may be mis-constructed."
            )

        # Plateau early-stopping (R4: parameterized patience).
        patience = cfg.plateau_patience
        if epoch >= max(15, patience) and not cfg.smoke_test:
            if mean >= last_loss - 1e-6:
                plateau_streak += 1
            else:
                plateau_streak = 0
            if plateau_streak >= patience:
                print(
                    f"[early-stop] Loss plateau: no decrease for {patience} epochs "
                    f"(epoch {epoch}, best_loss={best_loss:.4f}). Rolling back to best."
                )
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
        last_loss = mean

        # Bandwidth loss monitoring (warmup-independent).
        if (
            cfg.lambda_pvb > 0.0
            and epoch >= 20
            and not cfg.smoke_test
            and len(component_history) >= 5
        ):
            recent_pvb = [
                component_history[-i]["pvb"]
                for i in range(1, min(6, len(component_history) + 1))
            ]
            if all(p >= component_history[-5]["pvb"] - 1e-6 for p in recent_pvb):
                print(f"[warn] Bandwidth loss not decreasing for 5 epochs (epoch {epoch}).")

        if cfg.smoke_test:
            break

    # Save final checkpoint (best state if early-stopped, otherwise final).
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    final_state = best_state if best_state is not None else model.state_dict()
    torch.save(final_state, cfg.output)

    metadata = {
        "config": cfg.to_dict(),
        "history": history,
        "component_history": component_history,
        "bn_drift_history": bn_drift_history,
        "mask_mean_history": mask_mean_history,
        "final_loss": history[-1] if history else math.nan,
        "best_loss": best_loss,
        "early_stopped": plateau_streak >= cfg.plateau_patience,
        "total_time_s": time.time() - epoch_time_start,
        "dataset": "ganopc",
        "paper": "Yang2018_GANOPC",
        "scope": "generator-only (no discriminator); see scripts/train_gan_opc.py docstring",
        "version": "v0.4",
        "v04_changes": [
            "Change 1: PVB loss → metric-aligned bandwidth (mean(outer-inner))",
            "Change 2: differentiable_threshold (steepness=20) replaces hard threshold",
            "Change 3: Gradient accumulation support",
            "Change 4: Mixed precision (AMP) support",
            "Change 5: Memmap-backed dataset cache (single file)",
            "Change 6: Thread control (OMP/MKL/OPENBLAS=6)",
            "Change 7: UNetV2 (4-level, 64→512) architecture option",
            "Change 8: Plateau early-stopping with best-checkpoint rollback",
            "Change 9: BN-drift sliding baseline for >50 epoch runs",
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
            "steepness": cfg.pvb_steepness,
        },
        "reproducibility_note": (
            f"torch.manual_seed({cfg.seed}); device={cfg.device}; "
            f"arch={cfg.arch}; grad_accum={cfg.gradient_accumulation}; "
            f"amp={cfg.mixed_precision}"
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
    )
    p.add_argument("--device", default=_default_device())
    p.add_argument("--output", type=Path, default=Path("checkpoints/gan_opc_v0_4.pt"))
    p.add_argument(
        "--resize-to",
        type=int,
        default=512,
        help="Resize side. v0.4 default 512 (px=4.0). Stage 0 uses 256 (px=8.0).",
    )
    p.add_argument(
        "--resize-mode",
        default="bilinear",
        choices=["bilinear", "area"],
    )
    p.add_argument(
        "--consistency-weight",
        type=float,
        default=0.1,
    )
    p.add_argument("--num-kernels", type=int, default=24)
    p.add_argument(
        "--pixel-size-nm",
        type=float,
        default=4.0,
    )
    p.add_argument(
        "--lambda-mrc",
        type=float,
        default=0.5,
        help="MRC loss weight. 0 disables.",
    )
    p.add_argument("--lambda-mrc-warmup-epochs", type=int, default=10)
    p.add_argument("--lambda-mrc-warmup-start", type=float, default=0.0)
    p.add_argument("--mrc-min-width-nm", type=float, default=20.0)
    p.add_argument("--mrc-min-spacing-nm", type=float, default=20.0)
    p.add_argument("--mrc-weight-min-spacing", type=float, default=0.5)
    p.add_argument(
        "--lambda-pvb",
        type=float,
        default=0.1,
        help="PVB bandwidth loss weight. 0 disables.",
    )
    p.add_argument("--lambda-pvb-warmup-epochs", type=int, default=15)
    p.add_argument("--lambda-pvb-dose-delta", type=float, default=0.05)
    p.add_argument("--lambda-pvb-defocus-range-nm", type=float, default=20.0)
    p.add_argument("--lambda-pvb-sigma-nominal", type=float, default=2.0)
    p.add_argument(
        "--pvb-steepness",
        type=float,
        default=20.0,
        help="Steepness for differentiable_threshold in bandwidth loss. R3: fixed at 20.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps. v0.4 default 1 (use 2 for effective batch=8).",
    )
    p.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable AMP (torch.amp.autocast('cpu')). May slow down on AMD 5600G.",
    )
    p.add_argument(
        "--arch",
        default="unet",
        choices=["unet", "unetv2"],
        help="Architecture: unet (3-level, 32→256) or unetv2 (4-level, 64→512).",
    )
    p.add_argument(
        "--plateau-patience",
        type=int,
        default=5,
        help="Plateau early-stopping patience. Stage 1=5, Stage 2=10.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers. 0 for smoke test.",
    )
    p.add_argument(
        "--cache-dir",
        default="cache/ganopc/",
        help="Directory for memmap cache. Empty string disables caching.",
    )
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
        pvb_steepness=args.pvb_steepness,
        seed=args.seed,
        gradient_accumulation=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
        arch=args.arch,
        plateau_patience=args.plateau_patience,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
