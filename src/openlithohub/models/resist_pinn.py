"""Differentiable resist model via Physics-Informed Neural Networks.

Replaces hard-threshold resist with a learned model that respects
diffusion/development physics constraints.

References:
    - Physics-Informed Neural Networks, Raissi et al., 2019
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ResistPhysicsConstraints:
    """Physics constraints enforced on the resist PINN output.

    Attributes:
        monotonicity_weight: Scaling for monotonicity violation penalty.
        boundary_weight: Scaling for out-of-[0,1] violation penalty.
    """

    monotonicity_weight: float = 1.0
    boundary_weight: float = 1.0

    def monotonicity_loss(self, predictions: Tensor, doses: Tensor) -> Tensor:
        """Penalize non-monotonic predictions (more dose should yield more development).

        For each spatial location, sort by dose and penalize decreasing
        predictions.
        """
        flat_pred = predictions.flatten(1)
        flat_dose = doses.flatten(1)
        sorted_idx = flat_dose.argsort(dim=1)
        sorted_pred = flat_pred.gather(1, sorted_idx)
        diffs = sorted_pred[:, 1:] - sorted_pred[:, :-1]
        violations = torch.clamp(-diffs, min=0.0)
        return violations.mean() * self.monotonicity_weight

    def boundary_loss(self, predictions: Tensor) -> Tensor:
        """Penalize predictions outside [0, 1]."""
        below = torch.clamp(-predictions, min=0.0)
        above = torch.clamp(predictions - 1.0, min=0.0)
        violations = below + above
        return violations.mean() * self.boundary_weight


class _ResistBlock(nn.Module):
    """Conv-BN-ReLU block for the resist CNN."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)  # type: ignore[no-any-return]


class ResistPINN(nn.Module):
    """Physics-Informed Neural Network for differentiable resist modeling.

    Input:  dose field ``(B, 1, H, W)`` — aerial image intensity.
    Output: resist thickness ``(B, 1, H, W)`` in ``[0, 1]``.

    The network is a small encoder-decoder CNN with sigmoid output,
    regularised by monotonicity and boundary physics constraints.

    Attributes:
        physics: Physics constraint handler.
        data_weight: Weight for the data (MSE) loss term.
        physics_weight: Weight for the combined physics loss term.
    """

    def __init__(
        self,
        base_channels: int = 16,
        depth: int = 3,
        physics: ResistPhysicsConstraints | None = None,
        data_weight: float = 1.0,
        physics_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.physics = physics or ResistPhysicsConstraints()
        self.data_weight = data_weight
        self.physics_weight = physics_weight

        ch = base_channels
        encoder_layers: list[nn.Module] = [_ResistBlock(1, ch)]
        for _ in range(depth - 1):
            encoder_layers.append(nn.Sequential(nn.MaxPool2d(2), _ResistBlock(ch, ch * 2)))
            ch *= 2
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        dec_ch = ch
        for _ in range(depth - 1):
            decoder_layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            decoder_layers.append(_ResistBlock(dec_ch, dec_ch // 2))
            dec_ch //= 2
        self.decoder = nn.Sequential(*decoder_layers)

        self.head = nn.Conv2d(dec_ch, 1, kernel_size=1)

    def forward(self, dose: Tensor) -> Tensor:
        """Predict normalised resist thickness from dose field."""
        x = self.encoder(dose)
        x = self.decoder(x)
        return torch.sigmoid(self.head(x))

    def physics_loss(self, dose: Tensor, prediction: Tensor) -> Tensor:
        """Combined physics constraint loss (monotonicity + boundary)."""
        mono = self.physics.monotonicity_loss(prediction, dose)
        bnd = self.physics.boundary_loss(prediction)
        return mono + bnd

    def train_step(
        self,
        dose: Tensor,
        target_resist: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Execute one training step with data + physics loss.

        Returns:
            Dictionary with ``total``, ``data``, and ``physics`` loss values.
        """
        self.train()
        optimizer.zero_grad()
        prediction = self(dose)
        data_loss = nn.functional.mse_loss(prediction, target_resist) * self.data_weight
        phys_loss = self.physics_loss(dose, prediction) * self.physics_weight
        total = data_loss + phys_loss
        total.backward()
        optimizer.step()
        return {
            "total": total.item(),
            "data": data_loss.item(),
            "physics": phys_loss.item(),
        }


@dataclass
class _TrainHistory:
    losses: list[dict[str, float]] = field(default_factory=list)


class ResistCalibrator:
    """Calibrate a ResistPINN against measured/simulated resist profiles.

    Attributes:
        model: The PINN model being calibrated.
    """

    def __init__(self, model: ResistPINN | None = None) -> None:
        self.model = model or ResistPINN()

    def calibrate(
        self,
        train_doses: Tensor,
        train_resists: Tensor,
        n_epochs: int = 10,
        lr: float = 1e-3,
    ) -> list[dict[str, float]]:
        """Train the PINN on dose/resist pairs.

        Args:
            train_doses: Aerial image doses ``(N, 1, H, W)``.
            train_resists: Ground-truth resist thickness ``(N, 1, H, W)``.
            n_epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            Per-epoch loss history.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        history: list[dict[str, float]] = []
        for _ in range(n_epochs):
            metrics = self.model.train_step(train_doses, train_resists, optimizer)
            history.append(metrics)
        return history

    def evaluate(self, test_doses: Tensor, test_resists: Tensor) -> dict[str, float]:
        """Evaluate calibrated model against ground truth.

        Returns:
            Metrics dict with ``mse`` and ``edge_placement_error``.
        """
        self.model.eval()
        with torch.no_grad():
            pred = self.model(test_doses)
            mse = nn.functional.mse_loss(pred, test_resists).item()
            pred_binary = (pred > 0.5).float()
            gt_binary = (test_resists > 0.5).float()
            edge_err = (pred_binary != gt_binary).float().mean().item()
        return {"mse": mse, "edge_placement_error": edge_err}


class ResistBenchmark:
    """Compare hard threshold vs learned PINN resist."""

    def __init__(self, pinn: ResistPINN | None = None) -> None:
        self.pinn = pinn or ResistPINN()

    def compare(
        self,
        doses: Tensor,
        ground_truth: Tensor,
        threshold: float = 0.5,
    ) -> dict[str, dict[str, float]]:
        """Compare hard threshold and PINN resist predictions.

        Args:
            doses: Aerial image doses ``(N, 1, H, W)``.
            ground_truth: Ground-truth resist ``(N, 1, H, W)``.
            threshold: Hard threshold cutoff.

        Returns:
            ``{"hard_threshold": {...}, "pinn": {...}}`` each with
            ``mse``, ``edge_sharpness``, and ``process_window_accuracy``.
        """
        self.pinn.eval()
        with torch.no_grad():
            hard_pred = (doses >= threshold).float()
            pinn_pred = self.pinn(doses)

        hard_metrics = self._compute_metrics(hard_pred, ground_truth)
        pinn_metrics = self._compute_metrics(pinn_pred, ground_truth)

        return {"hard_threshold": hard_metrics, "pinn": pinn_metrics}

    @staticmethod
    def _compute_metrics(pred: Tensor, gt: Tensor) -> dict[str, float]:
        mse = nn.functional.mse_loss(pred, gt).item()
        grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        edge_sharpness = (grad_x.abs().mean() + grad_y.abs().mean()).item()
        gt_binary = (gt > 0.5).float()
        pred_binary = (pred > 0.5).float()
        process_window_accuracy = (pred_binary == gt_binary).float().mean().item()
        return {
            "mse": mse,
            "edge_sharpness": edge_sharpness,
            "process_window_accuracy": process_window_accuracy,
        }
