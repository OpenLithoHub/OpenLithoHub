"""GenAI warm-start interface and batch ILT candidate selection.

Provides ``WarmStartProvider`` (a Protocol), two concrete providers built on
the in-tree UNet (GAN-OPC and Neural-ILT flavours), a ``CandidateScorer``
that ranks masks using lithographic fidelity metrics, and a top-level
``warm_start_ilt`` entry-point that wires everything together: generate
candidates, optionally refine each with any ILT method, score, and return
the best mask.

Even without pretrained weights the warm-start providers produce
*structured* initial masks — Xavier-initialized UNet forward passes
correlate with the input target, giving ILT a non-random starting point
that converges in fewer iterations than a cold start.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.resist_model import apply_differentiable_resist
from openlithohub.models._unet import UNet

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class WarmStartProvider(Protocol):
    """Protocol for models that can provide ILT warm-start initializations."""

    def generate_initial_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Produce a single initial mask guess from *target*."""
        ...

    def generate_candidates(
        self, target: torch.Tensor, n_candidates: int
    ) -> list[torch.Tensor]:
        """Produce *n_candidates* diverse initial masks."""
        ...


# ---------------------------------------------------------------------------
# Shared helper — noise-injected forward pass for candidate diversity
# ---------------------------------------------------------------------------


def _forward_with_noise(
    net: nn.Module,
    target_4d: torch.Tensor,
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """Forward pass with additive Gaussian noise for candidate diversity.

    The in-tree UNet has no dropout layers, so simply toggling train/eval
    mode produces identical outputs. Instead we inject small Gaussian noise
    into the logits before sigmoid, which produces meaningfully different
    candidates while keeping them correlated with the input target.
    """
    logits = net(target_4d)
    noise = torch.randn_like(logits) * noise_scale
    return torch.sigmoid(logits + noise)


def _forward_clean(
    net: nn.Module,
    target_4d: torch.Tensor,
) -> torch.Tensor:
    """Deterministic forward pass — no noise."""
    return torch.sigmoid(net(target_4d))


# ---------------------------------------------------------------------------
# GAN-OPC warm-start provider
# ---------------------------------------------------------------------------


class GANOPCWarmStart(nn.Module):
    """GAN-OPC as warm-start provider.

    Uses the existing UNet generator to produce initial mask candidates.
    Not SOTA, but produces non-random initializations suitable for ILT
    refinement. Xavier initialization + the forward pass through the UNet
    gives structured output correlated with the input target.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.net = UNet(in_channels=in_channels, out_channels=out_channels)
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """Return a soft mask prediction for *target*."""
        x = self._ensure_4d(target)
        mask = _forward_clean(self.net, x).squeeze(0)
        return self._match_ndim(mask, target)

    def generate_initial_mask(self, target: torch.Tensor) -> torch.Tensor:
        x = self._ensure_4d(target)
        mask = _forward_clean(self.net, x).squeeze(0)
        return self._match_ndim(mask, target)

    def generate_candidates(
        self,
        target: torch.Tensor,
        n_candidates: int = 5,
    ) -> list[torch.Tensor]:
        x = self._ensure_4d(target)
        candidates = []
        for _ in range(n_candidates):
            mask = _forward_with_noise(self.net, x).squeeze(0)
            candidates.append(self._match_ndim(mask, target))
        return candidates

    @staticmethod
    def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.ndim == 3:
            return tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _match_ndim(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if ref.ndim == 2 and mask.ndim == 3:
            mask = mask.squeeze(0)
        return mask


# ---------------------------------------------------------------------------
# Neural-ILT warm-start provider
# ---------------------------------------------------------------------------


class NeuralILTWarmStart(nn.Module):
    """Neural-ILT as warm-start provider.

    Same UNet backbone as GAN-OPC but with a residual connection that adds
    a fraction of the input target back to the network output. This biases
    the warm-start mask toward the target shape even without trained
    weights — the residual connection ensures output is a blend of the
    UNet prediction and the target itself.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        residual_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.net = UNet(in_channels=in_channels, out_channels=out_channels)
        self.residual_weight = residual_weight
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        x = self._ensure_4d(target)
        raw = _forward_clean(self.net, x)
        blended = (1.0 - self.residual_weight) * raw + self.residual_weight * x
        mask = blended.squeeze(0)
        return self._match_ndim(mask, target)

    def generate_initial_mask(self, target: torch.Tensor) -> torch.Tensor:
        x = self._ensure_4d(target)
        raw = _forward_clean(self.net, x)
        blended = (1.0 - self.residual_weight) * raw + self.residual_weight * x
        mask = blended.squeeze(0)
        return self._match_ndim(mask, target)

    def generate_candidates(
        self,
        target: torch.Tensor,
        n_candidates: int = 5,
    ) -> list[torch.Tensor]:
        x = self._ensure_4d(target)
        candidates = []
        for _ in range(n_candidates):
            mask_soft = _forward_with_noise(self.net, x)
            blended = (
                (1.0 - self.residual_weight) * mask_soft
                + self.residual_weight * x
            )
            mask = blended.squeeze(0)
            candidates.append(self._match_ndim(mask, target))
        return candidates

    @staticmethod
    def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.ndim == 3:
            return tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _match_ndim(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if ref.ndim == 2 and mask.ndim == 3:
            mask = mask.squeeze(0)
        return mask


# ---------------------------------------------------------------------------
# Candidate scorer
# ---------------------------------------------------------------------------


class CandidateScorer:
    """Score and rank ILT candidates using lithographic metrics.

    The composite score is a weighted sum of:

    - **EPE** (edge placement error): MSE between simulated resist and target.
    - **PVB** (process-variation band): L2 penalty on mask gradients —
      smoother masks have smaller PVB in production.
    - **MRC** (minimum-rule check): penalty for sub-resolution features,
      approximated by total-variation on the mask.

    Lower score is better.
    """

    def __init__(
        self,
        forward_fn=None,
        epe_weight: float = 1.0,
        pvb_weight: float = 1.0,
        mrc_weight: float = 10.0,
        sigma_px: float = 2.0,
        dose: float = 1.0,
        resist_steepness: float = 50.0,
    ) -> None:
        if forward_fn is None:

            def _default_forward(
                mask: torch.Tensor,
            ) -> torch.Tensor:
                return simulate_aerial_image(mask, sigma_px=sigma_px, dose=dose)

            self._forward_fn = _default_forward
        else:
            self._forward_fn = forward_fn
        self.epe_weight = epe_weight
        self.pvb_weight = pvb_weight
        self.mrc_weight = mrc_weight
        self.resist_steepness = resist_steepness

    def score(self, mask: torch.Tensor, target: torch.Tensor) -> float:
        """Compute composite lithographic score for *mask* against *target*.

        Returns a scalar float — lower is better.
        """
        mask_2d = mask.detach().float()
        if mask_2d.ndim > 2:
            mask_2d = mask_2d.squeeze()
        target_2d = target.detach().float()
        if target_2d.ndim > 2:
            target_2d = target_2d.squeeze()

        aerial = self._forward_fn(mask_2d)
        resist = apply_differentiable_resist(
            aerial, threshold=0.5, steepness=self.resist_steepness
        )
        epe = nn.functional.mse_loss(resist, target_2d)

        grad_h = (mask_2d[1:, :] - mask_2d[:-1, :]).pow(2)
        grad_w = (mask_2d[:, 1:] - mask_2d[:, :-1]).pow(2)
        pvb = grad_h.mean() + grad_w.mean()

        tv = grad_h.sum() + grad_w.sum()
        mrc_penalty = tv / max(mask_2d.numel(), 1)

        total = (
            self.epe_weight * epe
            + self.pvb_weight * pvb
            + self.mrc_weight * mrc_penalty
        )
        return total.item()

    def rank_candidates(
        self,
        candidates: list[torch.Tensor],
        target: torch.Tensor,
    ) -> list[tuple[torch.Tensor, float]]:
        """Rank candidates by score, return sorted (mask, score) pairs.

        Best (lowest score) first.
        """
        scored = [(c, self.score(c, target)) for c in candidates]
        scored.sort(key=lambda pair: pair[1])
        return scored


# ---------------------------------------------------------------------------
# Top-level warm-start ILT entry-point
# ---------------------------------------------------------------------------


def warm_start_ilt(
    target: torch.Tensor,
    warm_start_provider: WarmStartProvider,
    ilt_refiner=None,
    n_candidates: int = 5,
    top_k: int = 3,
) -> dict:
    """Warm-start ILT: generate candidates -> optionally refine -> select best.

    Args:
        target: Binary design target, shape ``(H, W)`` or ``(1, H, W)``.
        warm_start_provider: Any object satisfying the
            :class:`WarmStartProvider` protocol.
        ilt_refiner: Optional callable ``refiner(mask, target) -> mask``
            that improves a single mask. When ``None`` the warm-start
            candidates are scored directly without refinement.
        n_candidates: Number of diverse warm-start candidates to generate.
        top_k: How many of the top-scoring candidates to refine (when
            *ilt_refiner* is provided). All *n_candidates* are scored
            before refinement, then only the *top_k* are refined and
            re-scored.

    Returns:
        Dict with keys:
        - ``best_mask``: highest-quality mask found.
        - ``all_results``: list of ``(mask, score)`` for all refined
          candidates (or all raw candidates if no refiner).
        - ``warm_start_mask``: the single best warm-start candidate
          *before* refinement.
        - ``cold_start_mask``: the target itself (identity cold-start)
          for comparison.
        - ``warm_start_score``: score of *warm_start_mask*.
        - ``cold_start_score``: score of *cold_start_mask*.
    """
    target_f = target.detach().float()
    if target_f.ndim > 2:
        target_f = target_f.squeeze()

    candidates = warm_start_provider.generate_candidates(target_f, n_candidates)

    scorer = CandidateScorer()

    warm_initial = warm_start_provider.generate_initial_mask(target_f)
    warm_score = scorer.score(warm_initial, target_f)
    cold_score = scorer.score(target_f, target_f)

    if ilt_refiner is not None:
        ranked = scorer.rank_candidates(candidates, target_f)
        to_refine = ranked[:top_k]
        refined: list[tuple[torch.Tensor, float]] = []
        for mask, _score in to_refine:
            refined_mask = ilt_refiner(mask, target_f)
            refined_score = scorer.score(refined_mask, target_f)
            refined.append((refined_mask, refined_score))
        refined.sort(key=lambda pair: pair[1])
        all_results = refined
    else:
        all_results = scorer.rank_candidates(candidates, target_f)

    best_mask = all_results[0][0]

    return {
        "best_mask": best_mask,
        "all_results": all_results,
        "warm_start_mask": warm_initial,
        "cold_start_mask": target_f,
        "warm_start_score": warm_score,
        "cold_start_score": cold_score,
    }
