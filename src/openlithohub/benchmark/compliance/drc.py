"""Design Rule Check (DRC) — layout-level geometric constraint validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from openlithohub._utils.morphology import binary_dilation, binary_erosion, connected_components
from openlithohub._utils.tensor_ops import ensure_2d


@dataclass
class DRCRuleDeck:
    """Configuration for DRC rules."""

    min_width_nm: float = 40.0
    min_spacing_nm: float = 40.0
    min_area_nm2: float = 100.0
    min_notch_nm: float = 30.0


@dataclass
class DRCResult:
    """Result of a Design Rule Check."""

    passed: bool
    violation_count: int
    violations: list[dict[str, float]]
    rule_summary: dict[str, int] = field(default_factory=dict)


_DEFAULT_RULES = DRCRuleDeck()

_RULE_DECKS: dict[str, DRCRuleDeck] = {
    "default": _DEFAULT_RULES,
    "aggressive": DRCRuleDeck(
        min_width_nm=20.0, min_spacing_nm=20.0, min_area_nm2=50.0, min_notch_nm=15.0
    ),
}


def check_drc(
    mask: torch.Tensor,
    rule_deck: str | DRCRuleDeck = "default",
    pixel_size_nm: float = 1.0,
) -> DRCResult:
    """Run Design Rule Check on a mask layout.

    Checks: minimum width, minimum spacing, minimum area, notch detection.
    """
    m = ensure_2d(mask)
    binary = (m > 0.5).float()

    if isinstance(rule_deck, str):
        if rule_deck not in _RULE_DECKS:
            raise ValueError(f"Unknown rule deck {rule_deck!r}. Available: {sorted(_RULE_DECKS)}")
        rules = _RULE_DECKS[rule_deck]
    else:
        rules = rule_deck

    violations: list[dict[str, float]] = []
    rule_summary: dict[str, int] = {}

    width_violations = _check_width(binary, rules.min_width_nm, pixel_size_nm)
    rule_summary["min_width"] = len(width_violations)
    violations.extend(width_violations)

    spacing_violations = _check_spacing(binary, rules.min_spacing_nm, pixel_size_nm)
    rule_summary["min_spacing"] = len(spacing_violations)
    violations.extend(spacing_violations)

    area_violations = _check_min_area(binary, rules.min_area_nm2, pixel_size_nm)
    rule_summary["min_area"] = len(area_violations)
    violations.extend(area_violations)

    notch_violations = _check_notch(binary, rules.min_notch_nm, pixel_size_nm)
    rule_summary["notch"] = len(notch_violations)
    violations.extend(notch_violations)

    violation_count = len(violations)
    return DRCResult(
        passed=violation_count == 0,
        violation_count=violation_count,
        violations=violations,
        rule_summary=rule_summary,
    )


def _check_width(
    binary: torch.Tensor, min_width_nm: float, pixel_size_nm: float
) -> list[dict[str, float]]:
    radius = int(math.floor(min_width_nm / (2.0 * pixel_size_nm)))
    if radius < 1 or binary.sum() == 0:
        return []

    opened = binary_dilation(binary_erosion(binary, radius=radius), radius=radius)
    violation_mask = (binary > 0.5) & (opened < 0.5)
    return _sample_violations(violation_mask, "width", min_width_nm, pixel_size_nm)


def _check_spacing(
    binary: torch.Tensor, min_spacing_nm: float, pixel_size_nm: float
) -> list[dict[str, float]]:
    radius = int(math.floor(min_spacing_nm / (2.0 * pixel_size_nm)))
    bg = (binary < 0.5).float()
    if radius < 1 or bg.sum() == 0 or binary.sum() == 0:
        return []

    opened_bg = binary_dilation(binary_erosion(bg, radius=radius), radius=radius)
    violation_mask = (bg > 0.5) & (opened_bg < 0.5)
    return _sample_violations(violation_mask, "spacing", min_spacing_nm, pixel_size_nm)


def _check_min_area(
    binary: torch.Tensor, min_area_nm2: float, pixel_size_nm: float
) -> list[dict[str, float]]:
    pixel_area_nm2 = pixel_size_nm * pixel_size_nm
    min_area_px = min_area_nm2 / pixel_area_nm2

    labels, num = connected_components(binary, connectivity=4)
    if num == 0:
        return []

    fg = labels >= 0
    flat_labels = labels[fg]
    ys, xs = torch.where(fg)
    unique_labels, inverse = torch.unique(flat_labels, return_inverse=True)
    n_comp = unique_labels.numel()

    counts = torch.zeros(n_comp, dtype=torch.float64, device=binary.device)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float64))
    sum_y = torch.zeros(n_comp, dtype=torch.float64, device=binary.device)
    sum_y.scatter_add_(0, inverse, ys.to(torch.float64))
    sum_x = torch.zeros(n_comp, dtype=torch.float64, device=binary.device)
    sum_x.scatter_add_(0, inverse, xs.to(torch.float64))

    counts_cpu = counts.tolist()
    cy_cpu = (sum_y / counts).tolist()
    cx_cpu = (sum_x / counts).tolist()

    violations: list[dict[str, float]] = []
    for i in range(n_comp):
        if len(violations) >= 50:
            break
        area_px = counts_cpu[i]
        if area_px >= min_area_px:
            continue
        violations.append(
            {
                "rule": 2.0,
                "type": 2.0,
                "x_nm": cx_cpu[i] * pixel_size_nm,
                "y_nm": cy_cpu[i] * pixel_size_nm,
                "actual_nm2": area_px * pixel_area_nm2,
                "required_nm2": min_area_nm2,
            }
        )
    return violations


def _check_notch(
    binary: torch.Tensor, min_notch_nm: float, pixel_size_nm: float
) -> list[dict[str, float]]:
    """Detect notches: narrow concavities in the background adjacent to features."""
    radius = int(math.floor(min_notch_nm / (2.0 * pixel_size_nm)))
    if radius < 1:
        return []

    bg = (binary < 0.5).float()
    if bg.sum() == 0:
        return []

    closed_fg = binary_dilation(binary_erosion(binary, radius=radius), radius=radius)
    notch_fill = (closed_fg > 0.5) & (binary < 0.5)

    spacing_radius = int(math.floor(min_notch_nm / (2.0 * pixel_size_nm)))
    opened_bg = binary_dilation(binary_erosion(bg, radius=spacing_radius), radius=spacing_radius)
    spacing_violation = (bg > 0.5) & (opened_bg < 0.5)

    notch_mask = notch_fill & ~spacing_violation
    return _sample_violations(notch_mask, "notch", min_notch_nm, pixel_size_nm)


def _sample_violations(
    violation_mask: torch.Tensor,
    rule_name: str,
    threshold_nm: float,
    pixel_size_nm: float,
    max_reports: int = 50,
) -> list[dict[str, float]]:
    if not violation_mask.any():
        return []

    ys, xs = torch.where(violation_mask)
    n = min(len(ys), max_reports)
    step = max(1, len(ys) // n)

    rule_code = {"width": 0.0, "spacing": 1.0, "area": 2.0, "notch": 3.0}.get(rule_name, 9.0)
    violations: list[dict[str, float]] = []

    for idx in range(0, len(ys), step):
        if len(violations) >= max_reports:
            break
        violations.append(
            {
                "rule": rule_code,
                "type": rule_code,
                "x_nm": float(xs[idx].item()) * pixel_size_nm,
                "y_nm": float(ys[idx].item()) * pixel_size_nm,
                "threshold_nm": threshold_nm,
            }
        )

    return violations
