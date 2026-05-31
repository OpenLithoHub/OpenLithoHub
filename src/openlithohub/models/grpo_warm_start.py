"""GRPO fine-tuning for generative mask warm-start.

References:
    - Pushing the Limits of Inverse Lithography with Generative RL, arXiv:2602.19027, 2026-02
    - Clean-room implementation of GRPO mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from openlithohub.models.posterior_warm_start import PosteriorWarmStart
from openlithohub.models.warm_start import CandidateScorer
from openlithohub.workflow.layer_purpose import LayerPurpose


@dataclass
class GRPOConfig:
    group_size: int = 4
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    ilt_guidance_weight: float = 0.5
    n_candidates: int = 16
    style_aware: bool = True


class StyleConditioning(nn.Module):
    """Embed layer-purpose pair info into a conditioning vector."""

    _PURPOSE_VOCAB = {
        "drawing": 0, "net": 1, "pin": 2, "label": 3,
        "boundary": 4, "blockage": 5, "fill": 6, "fillopc": 7,
        "track": 8, "slot": 9, "annotation": 10, "warning": 11,
        "redundant": 12, "notch": 13, "cutsom": 14,
    }
    _UNKNOWN_IDX = len(_PURPOSE_VOCAB)

    def __init__(self, embed_dim: int = 16, max_layers: int = 256) -> None:
        super().__init__()
        self.layer_embed = nn.Embedding(max_layers, embed_dim)
        self.purpose_embed = nn.Embedding(self._UNKNOWN_IDX + 1, embed_dim)
        self.proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, layer_purpose: LayerPurpose) -> torch.Tensor:
        lp = layer_purpose
        layer_id = torch.tensor([lp.layer % self.layer_embed.num_embeddings])
        purpose_name = lp.purpose if lp.purpose is not None else "drawing"
        purpose_id = torch.tensor([self._PURPOSE_VOCAB.get(purpose_name, self._UNKNOWN_IDX)])
        le = self.layer_embed(layer_id)
        pe = self.purpose_embed(purpose_id)
        return self.proj(torch.cat([le, pe], dim=-1)).squeeze(0)


class GRPOWarmStart(nn.Module):
    """GRPO-finetuned mask generator wrapping a PosteriorWarmStart VAE."""

    def __init__(
        self,
        vae: Optional[PosteriorWarmStart] = None,
        config: Optional[GRPOConfig] = None,
        style_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.config = config or GRPOConfig()
        self.vae = vae or PosteriorWarmStart()
        self.style_conditioning = StyleConditioning(embed_dim=style_embed_dim)
        self._old_log_probs: dict[int, torch.Tensor] = {}

    def generate_group(
        self,
        target: torch.Tensor,
        layer_purpose: Optional[LayerPurpose] = None,
    ) -> list[torch.Tensor]:
        """Generate group_size candidates for the same target."""
        candidates: list[torch.Tensor] = []
        for _ in range(self.config.group_size):
            mask = self.vae(target)
            candidates.append(mask)
        return candidates

    def compute_reward(
        self,
        candidates: list[torch.Tensor],
        target: torch.Tensor,
        scorer: CandidateScorer,
    ) -> torch.Tensor:
        """Score candidates; lower score = better, so negate for reward."""
        rewards = []
        for c in candidates:
            score = scorer.score(c, target)
            rewards.append(-score)
        return torch.tensor(rewards, dtype=torch.float32)

    def compute_ilt_guidance_loss(
        self,
        candidates: list[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """L2 loss between generated masks and ILT-optimized reference.

        When no ILT reference is available, uses the target as a proxy.
        """
        total = torch.tensor(0.0)
        for c in candidates:
            total = total + nn.functional.mse_loss(c, target)
        return total / max(len(candidates), 1)

    def style_aware_conditioning(self, layer_purpose: LayerPurpose) -> torch.Tensor:
        """Return conditioning vector from LayerPurpose."""
        return self.style_conditioning(layer_purpose)

    def _compute_log_prob(
        self,
        candidates: list[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-probability proxy for each candidate via VAE ELBO."""
        log_probs = []
        for c in candidates:
            loss = self.vae.loss(target, c)
            log_probs.append(-loss)
        return torch.stack(log_probs)

    def grpo_step(
        self,
        target: torch.Tensor,
        scorer: CandidateScorer,
        layer_purpose: Optional[LayerPurpose] = None,
        ilt_reference: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """One GRPO update step."""
        self.vae.train()

        candidates = self.generate_group(target, layer_purpose)
        log_probs_new = self._compute_log_prob(candidates, target)

        rewards = self.compute_reward(candidates, target, scorer)

        # Group-relative advantages
        adv_mean = rewards.mean()
        adv_std = rewards.std()
        if adv_std < 1e-8:
            adv_std = torch.tensor(1.0)
        advantages = (rewards - adv_mean) / adv_std

        # Old log-probs (detached snapshot)
        with torch.no_grad():
            log_probs_old = log_probs_new.detach()
        self._old_log_probs[id(target)] = log_probs_old

        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus (exploration)
        entropy = -(torch.sigmoid(log_probs_new) * log_probs_new).mean()
        entropy_bonus = -self.config.entropy_coeff * entropy

        # ILT imitation loss
        ref = ilt_reference if ilt_reference is not None else target
        ilt_loss = self.compute_ilt_guidance_loss(candidates, ref)

        loss = policy_loss + entropy_bonus + self.config.ilt_guidance_weight * ilt_loss

        # Backprop through VAE
        for p in self.vae.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()

        with torch.no_grad():
            for p in self.vae.parameters():
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-1e-3)

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "ilt_loss": ilt_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }

    def train_loop(
        self,
        targets: list[torch.Tensor],
        scorer: CandidateScorer,
        n_epochs: int = 10,
        layer_purposes: Optional[list[Optional[LayerPurpose]]] = None,
        ilt_references: Optional[list[Optional[torch.Tensor]]] = None,
    ) -> list[dict[str, float]]:
        """Full GRPO training loop."""
        lps = layer_purposes if layer_purposes is not None else [None] * len(targets)
        refs = ilt_references if ilt_references is not None else [None] * len(targets)

        history: list[dict[str, float]] = []
        for _epoch in range(n_epochs):
            for i, tgt in enumerate(targets):
                step_info = self.grpo_step(tgt, scorer, lps[i], refs[i])
                history.append(step_info)
        return history

    def evaluate_escape(
        self,
        target: torch.Tensor,
        n_attempts: int = 16,
        baseline_model: Optional[PosteriorWarmStart] = None,
        scorer: Optional[CandidateScorer] = None,
    ) -> dict[str, object]:
        """Compare with VAE-only baseline on escaping local optima."""
        if scorer is None:
            scorer = CandidateScorer()

        baseline = baseline_model or PosteriorWarmStart()

        grpo_scores: list[float] = []
        for _ in range(n_attempts):
            mask = self.vae(target)
            grpo_scores.append(scorer.score(mask, target))

        baseline_scores: list[float] = []
        for _ in range(n_attempts):
            mask = baseline(target)
            baseline_scores.append(scorer.score(mask, target))

        grpo_t = torch.tensor(grpo_scores)
        baseline_t = torch.tensor(baseline_scores)

        return {
            "grpo_mean": grpo_t.mean().item(),
            "grpo_std": grpo_t.std().item(),
            "baseline_mean": baseline_t.mean().item(),
            "baseline_std": baseline_t.std().item(),
            "improvement": (baseline_t.mean() - grpo_t.mean()).item(),
            "grpo_scores": grpo_scores,
            "baseline_scores": baseline_scores,
        }

    @staticmethod
    def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.ndim == 3:
            return tensor.unsqueeze(0)
        return tensor
