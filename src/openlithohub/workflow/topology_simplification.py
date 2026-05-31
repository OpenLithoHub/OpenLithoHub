"""Mask topology simplification with MBMW-aware post-processing.

Reduces curvilinear mask complexity to multi-beam-compatible geometry
while preserving lithographic quality via differentiable simplification.

References:
    - MBMW-aware mask optimization, SPIE 2024
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from openlithohub._utils.morphology import (
    connected_components,
    morphological_opening,
    mrc_projection,
)


@dataclass
class SimplificationConfig:
    target_edge_count: int = 64
    min_feature_size: int = 3
    litho_quality_weight: float = 0.5
    complexity_weight: float = 0.5


class TopologyAnalyzer:
    """Analyzes mask topology complexity."""

    @staticmethod
    def count_edges(mask: Tensor) -> int:
        """Count polygon edges via gradient-based edge detection."""
        m = mask.float()
        if m.ndim > 2:
            m = m.squeeze()
        binary_soft = torch.sigmoid(20.0 * (m - 0.5))
        gy = (binary_soft[1:, :] - binary_soft[:-1, :]).abs()
        gx = (binary_soft[:, 1:] - binary_soft[:, :-1]).abs()
        gy = functional.pad(gy, (0, 0, 0, 1))
        gx = functional.pad(gx, (0, 1, 0, 0))
        edge_map = (gy + gx) > 0.3
        return int(edge_map.sum().item())

    @staticmethod
    def count_connected_components(mask: Tensor) -> int:
        """Count the number of connected foreground components."""
        binary = (mask > 0.5).float()
        if binary.ndim > 2:
            binary = binary.squeeze()
        _, n = connected_components(binary, connectivity=8)
        return n

    @staticmethod
    def complexity_score(mask: Tensor) -> Tensor:
        """Return a scalar complexity score in [0, 1].

        Higher means more complex (more edges relative to area).
        """
        m = mask.float()
        if m.ndim > 2:
            m = m.squeeze()
        h, w = m.shape
        area = float(m.clamp(0, 1).sum().item())
        if area < 1.0:
            return torch.tensor(0.0)
        n_edges = TopologyAnalyzer.count_edges(mask)
        perimeter_ratio = n_edges / (2.0 * (h + w))
        return torch.tensor(min(1.0, perimeter_ratio))

    @staticmethod
    def mbmw_compatible(mask: Tensor, target_edges: int = 64) -> bool:
        """Check if the mask is MBMW-compatible (edge count within target)."""
        return TopologyAnalyzer.count_edges(mask) <= target_edges


class CurvilinearSimplifier(nn.Module):
    """Simplifies curvilinear mask geometry while preserving lithographic quality.

    Uses differentiable morphological operations to smooth and simplify
    curvilinear masks for multi-beam mask writer compatibility.
    """

    def __init__(self, smoothing_sigma: float = 1.5) -> None:
        super().__init__()
        self._sigma = smoothing_sigma
        self._gaussian_kernel: Tensor | None = None

    def _get_gaussian_kernel(self, size: int, device: torch.device) -> Tensor:
        if (
            self._gaussian_kernel is None
            or self._gaussian_kernel.shape[-1] != size
            or self._gaussian_kernel.device != device
        ):
            ax = torch.arange(size, dtype=torch.float32, device=device) - size // 2
            gauss_1d = torch.exp(-0.5 * (ax / self._sigma) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()
            kernel = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
            self._gaussian_kernel = kernel.unsqueeze(0).unsqueeze(0)
        return self._gaussian_kernel

    def forward(self, mask: Tensor, config: SimplificationConfig) -> Tensor:
        """Simplify mask through smoothing, grid snapping, and feature removal."""
        m = mask.float()
        if m.ndim > 2:
            m = m.squeeze()

        smoothed = self._smooth_edges(m)
        snapped = _ste_snap_to_grid(smoothed, grid_size=1.0)
        cleaned = self._remove_small_features(snapped, config.min_feature_size)
        return cleaned.clamp(0.0, 1.0)

    def _smooth_edges(self, mask: Tensor) -> Tensor:
        """Smooth edges via Gaussian filter."""
        k_size = int(6 * self._sigma + 1)
        if k_size % 2 == 0:
            k_size += 1
        kernel = self._get_gaussian_kernel(k_size, mask.device)
        pad = k_size // 2
        inp = mask.unsqueeze(0).unsqueeze(0)
        out = functional.conv2d(inp, kernel, padding=pad)
        return out.squeeze(0).squeeze(0)

    def _remove_small_features(self, mask: Tensor, min_size: int) -> Tensor:
        """Remove features smaller than min_size pixels using morphological opening."""
        if min_size < 1:
            return mask
        radius = max(0.5, min_size / 2.0)
        return morphological_opening(mask, radius=radius)

    def simplification_loss(
        self,
        original: Tensor,
        simplified: Tensor,
        config: SimplificationConfig,
    ) -> Tensor:
        """Combined loss for complexity reduction and litho quality preservation."""
        o = original.float()
        s = simplified.float()
        if o.ndim > 2:
            o = o.squeeze()
        if s.ndim > 2:
            s = s.squeeze()

        quality_loss = functional.mse_loss(s, o.detach())

        orig_complexity = TopologyAnalyzer.complexity_score(o)
        simp_complexity = TopologyAnalyzer.complexity_score(s)
        complexity_reduction = simp_complexity - orig_complexity
        complexity_loss = complexity_reduction.clamp(min=0.0)

        return (
            config.litho_quality_weight * quality_loss + config.complexity_weight * complexity_loss
        )


def _ste_snap_to_grid(mask: Tensor, grid_size: float = 1.0) -> Tensor:
    """Straight-through estimator grid snapping (STE from quantization).

    Forward pass rounds to the nearest grid point; backward pass passes
    gradients through as if the rounding never happened.
    """
    if grid_size <= 0:
        return mask
    snapped = mask / grid_size
    rounded = torch.round(snapped) * grid_size
    return mask + (rounded - mask).detach()


class MBMWPostProcessor:
    """Post-processes simplified masks for multi-beam writer compatibility."""

    @staticmethod
    def apply_mbmw_constraints(
        mask: Tensor,
        min_feature_size: int = 3,
    ) -> Tensor:
        """Apply all MBMW constraints to a mask."""
        m = mask.float()
        if m.ndim > 2:
            m = m.squeeze()
        constrained = MBMWPostProcessor.remove_small_features(m, min_feature_size)
        constrained = MBMWPostProcessor.snap_to_grid(constrained, grid_size=1.0)
        return constrained.clamp(0.0, 1.0)

    @staticmethod
    def snap_to_grid(mask: Tensor, grid_size: float = 1.0) -> Tensor:
        """Grid-align mask using STE."""
        return _ste_snap_to_grid(mask, grid_size)

    @staticmethod
    def remove_small_features(mask: Tensor, min_size: int = 3) -> Tensor:
        """Remove connected components smaller than min_size pixels."""
        if min_size < 1:
            return mask
        binary = (mask > 0.5).float()
        if binary.ndim > 2:
            binary = binary.squeeze()
        labels, n_components = connected_components(binary, connectivity=8)
        if n_components == 0:
            return mask.clamp(0.0, 1.0)
        result = binary.clone()
        fg = binary > 0.5
        unique_labels = torch.unique(labels[fg])
        for label_val in unique_labels:
            component_mask = labels == label_val
            component_pixels = int(component_mask.sum().item())
            if component_pixels < min_size:
                result[component_mask] = 0.0
        return result * mask.clamp(0.0, 1.0)

    @staticmethod
    def ensure_mrc_compliance(
        mask: Tensor,
        min_space: int = 3,
        min_width: int = 3,
    ) -> Tensor:
        """Enforce minimum spacing and width rules via MRC projection."""
        min_feature = max(min_space, min_width)
        return mrc_projection(mask, min_feature_px=float(min_feature)).clamp(0.0, 1.0)


@dataclass
class SimplificationResult:
    mask: Tensor
    original_edges: int
    simplified_edges: int
    complexity_reduction: float
    mbmw_compatible: bool


@dataclass
class ComparisonMetrics:
    edge_placement_error: float
    original_edge_count: int
    simplified_edge_count: int
    mbmw_compatible: bool
    area_preservation: float


class TopologyPipeline:
    """End-to-end simplification pipeline."""

    def __init__(self) -> None:
        self.analyzer = TopologyAnalyzer()
        self.simplifier = CurvilinearSimplifier()
        self.post_processor = MBMWPostProcessor()

    def simplify(
        self,
        mask: Tensor,
        config: SimplificationConfig,
    ) -> SimplificationResult:
        """Simplify mask and return result with metrics."""
        original_edges = self.analyzer.count_edges(mask)

        simplified = self.simplifier(mask, config)
        simplified = self.post_processor.apply_mbmw_constraints(simplified, config.min_feature_size)

        simplified_edges = self.analyzer.count_edges(simplified)
        complexity_reduction = 0.0
        if original_edges > 0:
            complexity_reduction = (original_edges - simplified_edges) / original_edges

        return SimplificationResult(
            mask=simplified,
            original_edges=original_edges,
            simplified_edges=simplified_edges,
            complexity_reduction=complexity_reduction,
            mbmw_compatible=self.analyzer.mbmw_compatible(simplified, config.target_edge_count),
        )

    def compare(
        self,
        original: Tensor,
        simplified: Tensor,
        target: Tensor | None = None,
    ) -> ComparisonMetrics:
        """Compare original and simplified masks, returning quality metrics."""
        o = original.float()
        s = simplified.float()
        if o.ndim > 2:
            o = o.squeeze()
        if s.ndim > 2:
            s = s.squeeze()

        diff = (o - s).abs()
        epe = float(diff.mean().item()) * 100.0

        orig_area = float(o.sum().item())
        simp_area = float(s.sum().item())
        area_preservation = 1.0 - abs(orig_area - simp_area) / max(orig_area, 1.0)

        return ComparisonMetrics(
            edge_placement_error=epe,
            original_edge_count=self.analyzer.count_edges(o),
            simplified_edge_count=self.analyzer.count_edges(s),
            mbmw_compatible=self.analyzer.mbmw_compatible(s),
            area_preservation=area_preservation,
        )

    def benchmark(
        self,
        masks: list[Tensor],
        targets: list[Tensor] | None,
        configs: list[SimplificationConfig],
    ) -> list[list[ComparisonMetrics]]:
        """Sweep multiple masks and configs, returning comparison results."""
        results: list[list[ComparisonMetrics]] = []
        for i, mask in enumerate(masks):
            mask_results: list[ComparisonMetrics] = []
            for config in configs:
                result = self.simplify(mask, config)
                target = targets[i] if targets is not None else None
                comparison = self.compare(mask, result.mask, target)
                mask_results.append(comparison)
            results.append(mask_results)
        return results
