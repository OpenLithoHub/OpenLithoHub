"""LevelSet-ILT: Iterative mask optimization via gradient descent."""

from __future__ import annotations

from typing import Any, Literal

import torch

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.hopkins import (
    HopkinsParams,
    compute_socs_kernels,
    simulate_aerial_image_hopkins,
)
from openlithohub._utils.resist_model import differentiable_threshold
from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry

ForwardModelKind = Literal["gaussian", "hopkins"]


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
    Supports two forward models:

    - ``gaussian`` (default): a single Gaussian PSF — fast, used in tests.
    - ``hopkins``: SOCS-truncated partial-coherence Hopkins imaging — physically
      faithful, suitable for end-to-end AI-OPC research.
    """

    def __init__(
        self,
        iterations: int = 200,
        lr: float = 0.1,
        sigma_px: float = 2.0,
        tv_weight: float = 0.01,
        dose: float = 1.0,
        resist_steepness: float = 50.0,
        forward_model: ForwardModelKind = "gaussian",
        hopkins_params: HopkinsParams | None = None,
    ) -> None:
        self._iterations = iterations
        self._lr = lr
        self._sigma_px = sigma_px
        self._tv_weight = tv_weight
        self._dose = dose
        self._resist_steepness = resist_steepness
        self._forward_model = forward_model
        self._hopkins_params = hopkins_params or HopkinsParams()
        self._cached_kernels: torch.Tensor | None = None
        self._cached_weights: torch.Tensor | None = None
        self._cached_grid: int | None = None
        self._compiled_hopkins_cache: dict[tuple[Any, ...], Any] = {}

    @property
    def name(self) -> str:
        return "levelset-ilt"

    @property
    def supports_curvilinear(self) -> bool:
        return True

    def _ensure_hopkins_kernels(
        self, grid_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cached_kernels is None
            or self._cached_weights is None
            or self._cached_grid != grid_size
            or self._cached_kernels.device != device
        ):
            kernels, weights = compute_socs_kernels(self._hopkins_params, grid_size, device)
            self._cached_kernels = kernels
            self._cached_weights = weights
            self._cached_grid = grid_size
        return self._cached_kernels, self._cached_weights

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        """Optimize a mask to reproduce the target design under lithography simulation.

        Args:
            design: Target design pattern (H, W), binary.
            **kwargs: Optional overrides — iterations, lr, sigma_px, tv_weight,
                forward_model, hopkins_params.
        """
        target = design.detach().float()
        if target.ndim > 2:
            target = target.squeeze()

        iterations = kwargs.get("iterations", self._iterations)
        lr = kwargs.get("lr", self._lr)
        sigma_px = kwargs.get("sigma_px", self._sigma_px)
        tv_weight = kwargs.get("tv_weight", self._tv_weight)
        forward_model = kwargs.get("forward_model", self._forward_model)
        hopkins_params = kwargs.get("hopkins_params", self._hopkins_params)
        dtype = kwargs.get("dtype", torch.float32)
        compile_forward = kwargs.get("compile_forward", False)

        if forward_model == "hopkins":
            if hopkins_params is not self._hopkins_params:
                self._hopkins_params = hopkins_params
                self._cached_kernels = None
                self._cached_weights = None
                self._cached_grid = None
            kernels, weights = self._ensure_hopkins_kernels(target.shape[0], target.device)
            hopkins_fn = simulate_aerial_image_hopkins
            if compile_forward:
                cache_key = (
                    target.shape[0],
                    str(target.device),
                    str(dtype),
                    forward_model,
                )
                compiled = self._compiled_hopkins_cache.get(cache_key)
                if compiled is None:
                    compiled = torch.compile(
                        hopkins_fn, mode="reduce-overhead", dynamic=False
                    )
                    self._compiled_hopkins_cache[cache_key] = compiled
                hopkins_fn = compiled
        else:
            kernels = None
            weights = None
            hopkins_fn = simulate_aerial_image_hopkins

        mask_logit = torch.zeros_like(target, requires_grad=True)
        with torch.no_grad():
            mask_logit.copy_(target * 4.0 - 2.0)
        mask_logit = mask_logit.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([mask_logit], lr=lr)

        best_loss = float("inf")
        best_mask: torch.Tensor = target.clone()

        for _ in range(iterations):
            optimizer.zero_grad()

            mask_continuous = torch.sigmoid(mask_logit)
            if forward_model == "hopkins":
                aerial = hopkins_fn(
                    mask_continuous,
                    kernels=kernels,
                    weights=weights,
                    dose=self._dose,
                    dtype=dtype,
                )
            else:
                aerial = simulate_aerial_image(mask_continuous, sigma_px=sigma_px, dose=self._dose)
            if aerial.dtype != torch.float32:
                aerial = aerial.float()

            resist = differentiable_threshold(
                aerial, threshold=0.5, steepness=self._resist_steepness
            )

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
                "forward_model": forward_model,
            },
        )
