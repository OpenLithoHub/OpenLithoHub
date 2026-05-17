"""Mask Rule Check (MRC) — minimum width/spacing for mask manufacturing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from openlithohub._utils.morphology import binary_dilation, binary_erosion, distance_transform
from openlithohub._utils.tensor_ops import ensure_2d


@dataclass
class MRCResult:
    """Result of a Mask Rule Check."""

    passed: bool
    violation_count: int
    violation_rate: float
    violations: list[dict[str, float]]


def check_mrc(
    mask: torch.Tensor,
    min_width_nm: float = 40.0,
    min_spacing_nm: float = 40.0,
    pixel_size_nm: float = 1.0,
) -> MRCResult:
    """Check mask against minimum width and spacing rules.

    MRC violations are a hard-fail metric — a mask that violates these rules
    cannot be manufactured regardless of optical performance.

    Width check: perform morphological opening (erosion then dilation) with
    radius = floor(min_width / (2 * pixel_size)). Features that survive opening
    are wide enough. Foreground pixels that disappear after opening are width
    violation pixels.

    Spacing check: same logic on the inverted mask — gaps between features that
    disappear under opening are too narrow.

    Args:
        mask: Binary mask tensor (H, W) or (B, C, H, W).
        min_width_nm: Minimum allowed feature width.
        min_spacing_nm: Minimum allowed spacing between features.
        pixel_size_nm: Physical pixel size for unit conversion.

    Returns:
        MRCResult with pass/fail status and violation details.
    """
    m = ensure_2d(mask)
    binary = (m > 0.5).float()

    h, w = binary.shape
    total_pixels = h * w
    has_foreground = binary.sum() > 0
    has_background = (1.0 - binary).sum() > 0

    violations: list[dict[str, float]] = []

    radius_width = int(math.floor(min_width_nm / (2.0 * pixel_size_nm)))
    radius_spacing = int(math.floor(min_spacing_nm / (2.0 * pixel_size_nm)))

    width_violation_count = 0
    spacing_violation_count = 0

    if has_foreground and radius_width >= 1:
        opened = binary_dilation(binary_erosion(binary, radius=radius_width), radius=radius_width)
        width_violation_mask = (binary > 0.5) & (opened < 0.5)
        width_violation_count = int(width_violation_mask.sum().item())

        if width_violation_count > 0:
            fg_dist = distance_transform(binary)
            ys, xs = torch.where(width_violation_mask)
            _add_violations(
                violations, "width", ys, xs, fg_dist, pixel_size_nm, min_width_nm
            )

    if has_foreground and has_background and radius_spacing >= 1:
        bg = (binary < 0.5).float()
        eroded_bg = binary_erosion(bg, radius=radius_spacing)
        opened_bg = binary_dilation(eroded_bg, radius=radius_spacing)
        spacing_violation_mask = (bg > 0.5) & (opened_bg < 0.5)
        spacing_violation_count = int(spacing_violation_mask.sum().item())

        if spacing_violation_count > 0:
            bg_dist = distance_transform(bg)
            ys, xs = torch.where(spacing_violation_mask)
            _add_violations(
                violations, "spacing", ys, xs, bg_dist, pixel_size_nm, min_spacing_nm
            )

    violation_count = width_violation_count + spacing_violation_count
    violation_rate = violation_count / total_pixels if total_pixels > 0 else 0.0

    return MRCResult(
        passed=violation_count == 0,
        violation_count=violation_count,
        violation_rate=violation_rate,
        violations=violations,
    )


def _add_violations(
    violations: list[dict[str, float]],
    vtype: str,
    ys: torch.Tensor,
    xs: torch.Tensor,
    dist_map: torch.Tensor,
    pixel_size_nm: float,
    threshold_nm: float,
    max_reports: int = 100,
) -> None:
    """Add sampled violation reports (limited to max_reports)."""
    n = min(len(ys), max_reports)
    step = max(1, len(ys) // n)
    for idx in range(0, len(ys), step):
        if len(violations) >= max_reports:
            break
        y_px = int(ys[idx].item())
        x_px = int(xs[idx].item())
        actual_nm = float(dist_map[y_px, x_px].item()) * 2.0 * pixel_size_nm
        violations.append({
            "type_code": 0.0 if vtype == "width" else 1.0,
            "x_nm": float(x_px) * pixel_size_nm,
            "y_nm": float(y_px) * pixel_size_nm,
            "actual_nm": actual_nm,
            "required_nm": threshold_nm,
        })
