"""Lightweight U-Net proxy for thick-mask aerial image prediction.

Trains a compact U-Net to predict aerial images from (mask, thickness)
pairs, using Born-series forward data as ground truth. Intended as a
fast surrogate during gradient-based ILT optimisation where repeatedly
calling the full Born forward model is too expensive.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from openlithohub._utils.forward_model import simulate_aerial_image_born
from openlithohub.models._unet import UNet


class ThickMaskProxy(nn.Module):
    """U-Net proxy that maps (mask, thickness) → aerial image.

    Takes a 2-channel input: the mask and a spatially-broadcast thickness
    parameter. Lightweight by design — uses the 3-level UNet with 32-base
    channels, not the full 4-level UNetV2.

    Args:
        base_channels: Base channel count for the U-Net encoder.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.unet = UNet(in_channels=2, out_channels=1)

    def forward(self, mask: torch.Tensor, thickness_param: torch.Tensor) -> torch.Tensor:
        """Predict aerial image from mask and thickness.

        Args:
            mask: ``(B, 1, H, W)`` layout mask.
            thickness_param: ``(B, 1)`` or ``(B,)`` scalar thickness value
                per batch element. Broadcast across spatial dimensions.

        Returns:
            ``(B, 1, H, W)`` predicted aerial image.
        """
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"Expected mask shape (B,1,H,W); got {tuple(mask.shape)}")

        if thickness_param.ndim == 1:
            thickness_param = thickness_param.unsqueeze(1)

        t_map = thickness_param.unsqueeze(-1).unsqueeze(-1).expand_as(mask)
        x = torch.cat([mask, t_map], dim=1)
        return self.unet(x)

    def train_from_born(
        self,
        n_samples: int = 64,
        image_size: int = 64,
        sigma_px: float = 2.0,
        dose: float = 1.0,
        thickness_range: tuple[float, float] = (0.0, 140.0),
        n_born_terms: int = 2,
        reflectivity: float = 0.1,
        n_epochs: int = 50,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> list[float]:
        """Generate Born-series training data and train the proxy.

        Creates random masks and thickness values, computes ground-truth
        aerial images via :func:`simulate_aerial_image_born`, and trains
        the U-Net to predict them.

        Args:
            n_samples: Number of training samples to generate.
            image_size: Spatial size of each sample (square).
            sigma_px: PSF width in pixels for Born simulation.
            dose: Exposure dose multiplier.
            thickness_range: (min, max) thickness in nm for random sampling.
            n_born_terms: Number of Born scattering terms.
            reflectivity: Born scattering coupling strength.
            n_epochs: Training epochs.
            lr: Learning rate.
            device: Compute device; defaults to CPU.

        Returns:
            List of per-epoch MSE losses.
        """
        if device is None:
            device = torch.device("cpu")

        self.to(device)
        self.train()

        masks = torch.rand(n_samples, 1, image_size, image_size, device=device)
        thicknesses = torch.rand(n_samples, device=device)
        thicknesses = thicknesses * (thickness_range[1] - thickness_range[0]) + thickness_range[0]

        targets = torch.zeros(n_samples, 1, image_size, image_size, device=device)
        for i in range(n_samples):
            m = masks[i, 0]
            targets[i, 0] = simulate_aerial_image_born(
                m,
                sigma_px=sigma_px,
                dose=dose,
                n_born_terms=n_born_terms,
                reflectivity=reflectivity,
            )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        losses: list[float] = []

        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred = self.forward(masks, thicknesses)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses
