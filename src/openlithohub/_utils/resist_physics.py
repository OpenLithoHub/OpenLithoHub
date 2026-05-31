"""Physically-motivated resist model with gradient fidelity verification.

Provides:
- ``PhysicalResistModel``: nn.Module implementing acid-generation, acid-diffusion,
  quencher neutralization, and sigmoid development — all fully differentiable.
- ``GradientFidelityGate``: verifies that a surrogate forward's analytical gradient
  agrees with finite-difference reference to a configurable tolerance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as functional

from openlithohub._utils.forward_model import _build_gaussian_kernel, _circular_pad_clamped


class PhysicalResistModel(nn.Module):
    """Minimum viable physically-motivated resist model.

    Models the key resist effects:
    1. Acid generation (proportional to dose)
    2. Acid diffusion (Gaussian blur)
    3. Development (sigmoid threshold with quencher)

    All steps are differentiable for end-to-end ILT optimization.
    """

    def __init__(
        self,
        pixel_size_nm: float = 1.0,
        acid_diffusion_nm: float = 5.0,
        development_steepness: float = 10.0,
        quencher_concentration: float = 0.1,
    ):
        super().__init__()
        self.pixel_size_nm = pixel_size_nm
        self.acid_diffusion_nm = acid_diffusion_nm
        self.steepness = nn.Parameter(
            torch.tensor(development_steepness, dtype=torch.float32), requires_grad=False
        )
        self.quencher = nn.Parameter(
            torch.tensor(quencher_concentration, dtype=torch.float32), requires_grad=False
        )

    def forward(self, aerial_image: torch.Tensor) -> torch.Tensor:
        """Forward resist model: aerial image -> resist contour.

        Args:
            aerial_image: (H, W) tensor of intensity values.

        Returns:
            resist_contour: (H, W) tensor in [0, 1].
        """
        # 1. Acid generation: dose-dependent
        acid = self._acid_generation(aerial_image)
        # 2. Acid diffusion: Gaussian blur
        acid_diffused = self._acid_diffusion(acid)
        # 3. Quencher reaction
        net_acid = acid_diffused - self.quencher
        # 4. Development: sigmoid
        resist = torch.sigmoid(self.steepness * net_acid)
        return resist

    # ------------------------------------------------------------------
    # Internal sub-steps
    # ------------------------------------------------------------------

    def _acid_generation(self, dose: torch.Tensor) -> torch.Tensor:
        """Acid concentration from dose (proportional, clamped to [0, inf))."""
        return dose.clamp(min=0.0)

    def _acid_diffusion(self, acid: torch.Tensor) -> torch.Tensor:
        """Acid diffusion via depthwise convolution with Gaussian kernel.

        Uses circular padding consistent with the Hopkins forward model.
        """
        sigma_px = self.acid_diffusion_nm / max(self.pixel_size_nm, 1e-6)
        if sigma_px < 0.1:
            return acid
        kernel = _build_gaussian_kernel(sigma_px, acid.device)
        padding = kernel.shape[-1] // 2
        inp = acid.unsqueeze(0).unsqueeze(0)
        inp_padded = _circular_pad_clamped(inp, padding)
        return functional.conv2d(inp_padded, kernel).squeeze(0).squeeze(0)


@dataclass
class FidelityResult:
    """Result from a single gradient fidelity verification."""

    surrogate_gradient_cosine: float
    finite_diff_gradient_cosine: float
    max_component_error: float
    passed: bool
    details: dict[str, object] = field(default_factory=dict)


class GradientFidelityGate:
    """Verify gradient fidelity of surrogate against high-fidelity forward.

    Computes gradient of resist output w.r.t. mask using:
    1. Surrogate forward (fast approximation)
    2. High-fidelity forward (Born/Abbe)
    3. Finite difference reference

    Verifies that surrogate gradient is consistent with high-fidelity.
    """

    def __init__(
        self,
        surrogate_fn: Callable[[torch.Tensor], torch.Tensor],
        high_fidelity_fn: Callable[[torch.Tensor], torch.Tensor],
        eps: float = 1e-4,
        cosine_threshold: float = 0.99,
        max_component_threshold: float = 0.05,
    ):
        self.surrogate_fn = surrogate_fn
        self.high_fidelity_fn = high_fidelity_fn
        self.eps = eps
        self.cosine_threshold = cosine_threshold
        self.max_component_threshold = max_component_threshold

    def verify(self, mask: torch.Tensor, target: torch.Tensor) -> FidelityResult:
        """Run gradient fidelity verification.

        Args:
            mask: Input mask tensor (H, W) with requires_grad capability.
            target: Target wafer pattern (H, W) for loss computation.

        Returns:
            FidelityResult with cosine similarities, component error, and pass/fail.
        """
        # --- Analytical gradient via surrogate ---
        mask_s = mask.detach().clone().requires_grad_(True)
        out_s = self.surrogate_fn(mask_s)
        loss_s = _mse_loss(out_s, target)
        loss_s.backward()  # type: ignore[no-untyped-call]
        assert mask_s.grad is not None
        grad_surrogate = mask_s.grad.detach().clone().flatten()

        # --- Analytical gradient via high-fidelity ---
        mask_h = mask.detach().clone().requires_grad_(True)
        out_h = self.high_fidelity_fn(mask_h)
        loss_h = _mse_loss(out_h, target)
        loss_h.backward()  # type: ignore[no-untyped-call]
        assert mask_h.grad is not None
        grad_hf = mask_h.grad.detach().clone().flatten()

        # --- Finite-difference gradient (high-fidelity) ---
        grad_fd = self._finite_diff_gradient(mask, target)

        # --- Metrics ---
        cos_surrogate = _cosine_similarity(grad_surrogate, grad_hf)
        cos_fd = _cosine_similarity(grad_fd, grad_hf)

        denom = grad_hf.abs().clamp(min=1e-8)
        max_comp_err = (grad_surrogate - grad_hf).abs().div(denom).max().item()

        passed = (
            cos_surrogate >= self.cosine_threshold and max_comp_err <= self.max_component_threshold
        )

        return FidelityResult(
            surrogate_gradient_cosine=cos_surrogate,
            finite_diff_gradient_cosine=cos_fd,
            max_component_error=max_comp_err,
            passed=passed,
            details={
                "surrogate_grad_norm": grad_surrogate.norm().item(),
                "hf_grad_norm": grad_hf.norm().item(),
                "fd_grad_norm": grad_fd.norm().item(),
            },
        )

    def benchmark(self, n_masks: int = 10, size: int = 16) -> dict[str, object]:
        """Run verification across multiple random masks.

        Returns aggregate statistics.
        """
        results: list[FidelityResult] = []
        for _ in range(n_masks):
            mask = torch.rand(size, size)
            target = torch.bernoulli(torch.full((size, size), 0.5))
            results.append(self.verify(mask, target))

        cosines_s = [r.surrogate_gradient_cosine for r in results]
        cosines_fd = [r.finite_diff_gradient_cosine for r in results]
        max_errs = [r.max_component_error for r in results]

        return {
            "n_masks": n_masks,
            "surrogate_cosine_mean": sum(cosines_s) / len(cosines_s),
            "surrogate_cosine_min": min(cosines_s),
            "fd_cosine_mean": sum(cosines_fd) / len(cosines_fd),
            "fd_cosine_min": min(cosines_fd),
            "max_component_error_mean": sum(max_errs) / len(max_errs),
            "max_component_error_max": max(max_errs),
            "all_passed": all(r.passed for r in results),
            "n_passed": sum(1 for r in results if r.passed),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _finite_diff_gradient(self, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Central finite-difference gradient of MSE loss w.r.t. mask."""
        flat = mask.detach().clone().flatten()
        grad = torch.zeros_like(flat)
        for i in range(flat.numel()):
            mask_plus = flat.clone()
            mask_plus[i] += self.eps
            mask_minus = flat.clone()
            mask_minus[i] -= self.eps
            loss_plus = _mse_loss(
                self.high_fidelity_fn(mask_plus.reshape(mask.shape)), target
            ).item()
            loss_minus = _mse_loss(
                self.high_fidelity_fn(mask_minus.reshape(mask.shape)), target
            ).item()
            grad[i] = (loss_plus - loss_minus) / (2.0 * self.eps)
        return grad


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


def _mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((output - target) ** 2).mean()


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm().clamp(min=1e-8) * b.norm().clamp(min=1e-8)
    val: float = (a.dot(b) / denom).item()
    return val
