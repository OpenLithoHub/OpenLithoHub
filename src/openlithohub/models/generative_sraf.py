"""Generative SRAF insertion using reinforcement learning.

Learns to place sub-resolution assist features that improve lithographic
quality, built on top of the GRPO warm-start framework.

References:
    - SRAF insertion via deep RL, SPIE 2024
    - O9.1 GRPO framework (arXiv:2602.19027)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.resist_model import differentiable_threshold


@dataclass
class SRAFConfig:
    mask_size: int = 64
    max_sraf_count: int = 20
    sraf_size_range: tuple[int, int] = (2, 6)
    improvement_threshold: float = 0.01


class SRAFGenerator(nn.Module):
    """Generate SRAF placement candidates from a mask layout.

    Encoder-decoder architecture: a CNN processes the input mask into a
    feature map, then the decoder produces ``(x, y, w, h, confidence)``
    tuples for up to ``max_sraf_count`` SRAF rectangles.
    """

    def __init__(self, config: SRAFConfig | None = None) -> None:
        super().__init__()
        self.config = config or SRAFConfig()
        s = self.config.mask_size
        n = self.config.max_sraf_count

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n * 5),
        )

        self._s = s

    def forward(self, mask: Tensor) -> Tensor:
        """Return SRAF placements ``(B, max_sraf_count, 5)``.

        Each row is ``(x, y, w, h, confidence)`` normalised to ``[0, 1]``.
        """
        x = mask.float()
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(1)

        bsz = x.shape[0]
        feat = self.encoder(x)
        raw = self.decoder(feat)
        placements = raw.view(bsz, self.config.max_sraf_count, 5)

        placements[..., :4] = torch.sigmoid(placements[..., :4])
        placements[..., 4] = torch.sigmoid(placements[..., 4])

        lo, hi = self.config.sraf_size_range
        s = self._s
        placements[..., 2] = placements[..., 2] * (hi - lo) / s + lo / s
        placements[..., 3] = placements[..., 3] * (hi - lo) / s + lo / s

        return cast(Tensor, placements)

    def render_srafs(self, mask: Tensor, placements: Tensor) -> Tensor:
        """Overlay SRAF rectangles onto *mask*, returning a new mask tensor.

        SRAFs are rendered as 0.5-valued patches (sub-resolution — they
        should not print at wafer level but improve main-feature contrast).
        """
        result = mask.float().clone()
        if result.ndim == 2:
            result = result.unsqueeze(0)
        if placements.ndim == 2:
            placements = placements.unsqueeze(0)

        s = self._s
        lo, hi = self.config.sraf_size_range
        for b in range(result.shape[0]):
            for i in range(placements.shape[1]):
                conf = placements[b, i, 4].item()
                if conf < 0.3:
                    continue
                cx = int(placements[b, i, 0].item() * s)
                cy = int(placements[b, i, 1].item() * s)
                w = int(placements[b, i, 2].item() * s)
                h = int(placements[b, i, 3].item() * s)
                w = max(lo, min(w, hi))
                h = max(lo, min(h, hi))
                x0 = max(0, cx - w // 2)
                y0 = max(0, cy - h // 2)
                x1 = min(s, cx + w // 2)
                y1 = min(s, cy + h // 2)
                if x1 > x0 and y1 > y0:
                    result[b, y0:y1, x0:x1] = torch.maximum(
                        result[b, y0:y1, x0:x1],
                        torch.tensor(0.5),
                    )
        if mask.ndim == 2:
            result = result.squeeze(0)
        return cast(Tensor, result)


class LithographicImprovementScorer:
    """Score SRAF placements by measuring lithographic improvement.

    Compares NILS (Normalized Image Log Slope) of the aerial image with and
    without SRAFs. Higher score means the SRAFs improved printability.
    """

    def __init__(
        self,
        sigma_px: float = 2.0,
        dose: float = 1.0,
        steepness: float = 50.0,
    ) -> None:
        self.sigma_px = sigma_px
        self.dose = dose
        self.steepness = steepness

    def _nils(self, aerial: Tensor, target: Tensor) -> Tensor:
        """Approximate NILS at feature edges."""
        target_f = target.float()
        if target_f.ndim > 2:
            target_f = target_f.squeeze()
        aerial_f = aerial.float()
        if aerial_f.ndim > 2:
            aerial_f = aerial_f.squeeze()

        grad_x = aerial_f[:, 1:] - aerial_f[:, :-1]
        grad_y = aerial_f[1:, :] - aerial_f[:-1, :]

        edge_mask = torch.zeros_like(target_f)
        edge_mask[:, 1:] = torch.maximum(
            edge_mask[:, 1:], (target_f[:, 1:] != target_f[:, :-1]).float()
        )
        edge_mask[1:, :] = torch.maximum(
            edge_mask[1:, :], (target_f[1:, :] != target_f[:-1, :]).float()
        )

        if edge_mask.sum() < 1e-6:
            return torch.tensor(0.0)

        slope = torch.sqrt(grad_x[:-1, :] ** 2 + grad_y[:, :-1] ** 2 + 1e-8)
        edge_crop = edge_mask[:-1, :-1]
        nils = (slope * edge_crop).sum() / (edge_crop.sum() + 1e-8)
        return cast(Tensor, nils)

    def score(self, mask_with_sraf: Tensor, mask_without_sraf: Tensor, target: Tensor) -> float:
        """Return improvement score (positive = SRAFs helped)."""
        m_ws = mask_with_sraf.float()
        m_wo = mask_without_sraf.float()
        tgt = target.float()

        aerial_ws = simulate_aerial_image(m_ws, sigma_px=self.sigma_px, dose=self.dose)
        aerial_wo = simulate_aerial_image(m_wo, sigma_px=self.sigma_px, dose=self.dose)

        nils_ws = self._nils(aerial_ws, tgt)
        nils_wo = self._nils(aerial_wo, tgt)

        resist_ws = differentiable_threshold(aerial_ws, threshold=0.5, steepness=self.steepness)
        resist_wo = differentiable_threshold(aerial_wo, threshold=0.5, steepness=self.steepness)

        epe_ws = nn.functional.mse_loss(resist_ws, tgt)
        epe_wo = nn.functional.mse_loss(resist_wo, tgt)

        improvement = (nils_ws - nils_wo) + 0.5 * (epe_wo - epe_ws)
        return improvement.item()


class SRAFRLTrainer:
    """Train SRAFGenerator using GRPO-style reinforcement learning.

    Reuses the group-relative advantage formulation from O9.1 GRPO.
    """

    def __init__(
        self,
        generator: SRAFGenerator | None = None,
        config: SRAFConfig | None = None,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
    ) -> None:
        self.config = config or SRAFConfig()
        self.generator = generator or SRAFGenerator(self.config)
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)

    def _generate_noisy(self, mask: Tensor, n: int) -> list[Tensor]:
        self.generator.train()
        placements_list: list[Tensor] = []
        for _ in range(n):
            p = self.generator(mask)
            placements_list.append(p)
        return placements_list

    def train_step(
        self,
        masks: Tensor,
        targets: Tensor,
        scorer: LithographicImprovementScorer,
    ) -> dict[str, float]:
        """One GRPO-style RL step on a batch of (mask, target) pairs."""
        self.generator.train()
        self.optimizer.zero_grad()

        group_size = 4
        total_loss = torch.tensor(0.0, requires_grad=True)
        total_improvement = 0.0
        n_pairs = masks.shape[0] if masks.ndim >= 3 else 1

        for idx in range(n_pairs):
            m = masks[idx] if masks.ndim >= 3 else masks
            t = targets[idx] if targets.ndim >= 3 else targets

            candidates = self._generate_noisy(m, group_size)
            rewards_list: list[float] = []
            for c in candidates:
                rendered = self.generator.render_srafs(m, c)
                r = scorer.score(rendered, m, t)
                rewards_list.append(r)
            rewards = torch.tensor(rewards_list, dtype=torch.float32)

            mean_r = rewards.mean()
            std_r = rewards.std()
            if std_r < 1e-8:
                std_r = torch.tensor(1.0)
            advantages = (rewards - mean_r) / std_r

            placement_logits_list: list[Tensor] = []
            for _c in candidates:
                p = self.generator(m)
                placement_logits_list.append(p)
            log_probs = torch.stack([p.sum() for p in placement_logits_list])

            entropy = -(torch.sigmoid(log_probs) * log_probs).mean()
            entropy_bonus = -self.entropy_coeff * entropy

            adv_weighted = (advantages * log_probs).mean()
            loss = -adv_weighted + entropy_bonus
            total_loss = total_loss + loss
            total_improvement += mean_r.item()

        avg_loss = total_loss / max(n_pairs, 1)
        avg_loss.backward()
        self.optimizer.step()

        return {
            "loss": avg_loss.item(),
            "improvement": total_improvement / max(n_pairs, 1),
        }

    def train(
        self,
        train_data: list[tuple[Tensor, Tensor]],
        n_epochs: int = 5,
        lr: float = 1e-3,
    ) -> list[dict[str, float]]:
        """Full training loop returning per-epoch history."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        scorer = LithographicImprovementScorer()
        history: list[dict[str, float]] = []

        for _epoch in range(n_epochs):
            for mask, target in train_data:
                masks_b = mask.unsqueeze(0) if mask.ndim == 2 else mask
                targets_b = target.unsqueeze(0) if target.ndim == 2 else target
                step = self.train_step(masks_b, targets_b, scorer)
                history.append(step)
        return history


class SRAFPipeline:
    """End-to-end SRAF insertion pipeline.

    Generates multiple SRAF placement candidates, scores them by
    lithographic improvement, and returns the best result.
    """

    def __init__(
        self,
        generator: SRAFGenerator | None = None,
        scorer: LithographicImprovementScorer | None = None,
        config: SRAFConfig | None = None,
    ) -> None:
        self.config = config or SRAFConfig()
        self.generator = generator or SRAFGenerator(self.config)
        self.scorer = scorer or LithographicImprovementScorer()

    @torch.no_grad()
    def insert_srafs(
        self,
        mask: Tensor,
        target: Tensor,
        n_candidates: int = 4,
    ) -> Tensor:
        """Return the best mask with SRAFs from *n_candidates* trials."""
        self.generator.eval()
        mask_f = mask.float()
        target_f = target.float()

        best_mask = mask_f.clone()
        best_score = -float("inf")

        for _ in range(n_candidates):
            placements = self.generator(mask_f)
            rendered = self.generator.render_srafs(mask_f, placements)
            score = self.scorer.score(rendered, mask_f, target_f)
            if score > best_score:
                best_score = score
                best_mask = rendered.clone()

        if best_score < self.config.improvement_threshold:
            return mask_f

        return best_mask

    def evaluate(
        self,
        mask_with_sraf: Tensor,
        mask_without_sraf: Tensor,
        target: Tensor,
    ) -> dict[str, float]:
        """Return improvement metrics dict."""
        imp = self.scorer.score(mask_with_sraf, mask_without_sraf, target)
        return {
            "improvement": imp,
            "improved": imp > 0.0,
        }

    @torch.no_grad()
    def benchmark(
        self,
        masks: list[Tensor],
        targets: list[Tensor],
    ) -> dict[str, float]:
        """Run full SRAF insertion benchmark on (mask, target) pairs."""
        total_improvement = 0.0
        n_improved = 0

        for mask, target in zip(masks, targets, strict=True):
            mask_with = self.insert_srafs(mask, target)
            result = self.evaluate(mask_with, mask, target)
            total_improvement += result["improvement"]
            if result["improved"]:
                n_improved += 1

        n = max(len(masks), 1)
        return {
            "avg_improvement": total_improvement / n,
            "improvement_rate": n_improved / n,
            "n_samples": float(len(masks)),
        }
