"""LevelSet-ILT: Iterative mask optimization via gradient descent."""

from __future__ import annotations

from typing import Any

import torch

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    """Compute isotropic total variation for a 2D tensor."""
    diff_h = (x[1:, :] - x[:-1, :]).pow(2)
    diff_w = (x[:, 1:] - x[:, :-1]).pow(2)
    return diff_h.sum() + diff_w.sum()


@registry.register
class LevelSetILTModel(LithographyModel):
    """Inverse Lithography Technology via level-set gradient descent.

    Optimizes a continuous mask representation to minimize the difference
    between the simulated resist image and the target design pattern.
    Uses the built-in Gaussian PSF forward model with autograd.
    """

    def __init__(
        self,
        iterations: int = 200,
        lr: float = 0.1,
        sigma_px: float = 2.0,
        tv_weight: float = 0.01,
        dose: float = 1.0,
        resist_steepness: float = 50.0,
    ) -> None:
        self._iterations = iterations
        self._lr = lr
        self._sigma_px = sigma_px
        self._tv_weight = tv_weight
        self._dose = dose
        self._resist_steepness = resist_steepness

    @property
    def name(self) -> str:
        return "levelset-ilt"

    @property
    def supports_curvilinear(self) -> bool:
        return True

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        """Optimize a mask to reproduce the target design under lithography simulation.

        Args:
            design: Target design pattern (H, W), binary.
            **kwargs: Optional overrides — iterations, lr, sigma_px, tv_weight.
        """
        target = design.detach().float()
        if target.ndim > 2:
            target = target.squeeze()

        iterations = kwargs.get("iterations", self._iterations)
        lr = kwargs.get("lr", self._lr)
        sigma_px = kwargs.get("sigma_px", self._sigma_px)
        tv_weight = kwargs.get("tv_weight", self._tv_weight)

        mask_logit = torch.zeros_like(target, requires_grad=True)
        # Initialize closer to target to speed convergence
        with torch.no_grad():
            mask_logit.copy_(target * 4.0 - 2.0)
        mask_logit = mask_logit.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([mask_logit], lr=lr)

        best_loss = float("inf")
        best_mask: torch.Tensor = target.clone()

        for _ in range(iterations):
            optimizer.zero_grad()

            mask_continuous = torch.sigmoid(mask_logit)
            aerial = simulate_aerial_image(mask_continuous, sigma_px=sigma_px, dose=self._dose)

            # Differentiable resist threshold via steep sigmoid
            resist = torch.sigmoid(self._resist_steepness * (aerial - 0.5))

            fidelity_loss = torch.nn.functional.mse_loss(resist, target)
            tv_loss = _total_variation(mask_continuous)
            loss = fidelity_loss + tv_weight * tv_loss

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_mask = (mask_continuous > 0.5).float().detach()

        return PredictionResult(
            mask=best_mask,
            metadata={
                "final_loss": best_loss,
                "iterations": iterations,
                "sigma_px": sigma_px,
            },
        )
