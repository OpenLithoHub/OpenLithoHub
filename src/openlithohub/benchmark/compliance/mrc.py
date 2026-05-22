"""Mask Rule Check (MRC) — minimum width/spacing for mask manufacturing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from openlithohub._utils.contour_trace import trace_contour
from openlithohub._utils.morphology import (
    binary_dilation,
    binary_erosion,
    connected_components,
    distance_transform,
)
from openlithohub._utils.sampling import evenly_spaced_indices
from openlithohub._utils.tensor_ops import ensure_2d


@dataclass
class MRCResult:
    """Result of a Mask Rule Check.

    .. note::
        ``violation_count`` is the count of **violating pixels**, the
        sum of per-pixel boolean masks for ``width`` and ``spacing``
        rules. It is unclipped and scales with the feature area of the
        layout — a 4096² mask with 1% violation density reports a
        bigger number than a 256² one with the same fractional rate.
        Use ``violation_rate`` (count / total pixels) for area-
        independent comparison.

        MRC ``violation_count`` is **not directly comparable** to DRC
        ``violation_count`` — DRC counts connected components and is
        clipped at the rule's ``max_reports`` cap, while MRC counts
        pixels. ``passed`` / ``passed`` comparisons are well-defined;
        magnitude comparisons are not.

        ``violations`` is a per-violation sample list (capped at
        ``max_reports``, evenly spaced) used for visualisation and
        debug; do not derive counts from it — use ``violation_count``
        directly.
    """

    passed: bool
    violation_count: int
    violation_rate: float
    violations: list[dict[str, float]]
    width_violation_count: int = 0
    spacing_violation_count: int = 0

    def _repr_html_(self) -> str:
        from openlithohub.jupyter._html import (
            kv_table,
            panel,
            pass_fail_badge,
            violation_table,
        )

        rows = [
            ("Total violations", str(self.violation_count)),
            ("Violation rate", f"{self.violation_rate:.4%}"),
            ("Width violations", str(self.width_violation_count)),
            ("Spacing violations", str(self.spacing_violation_count)),
        ]
        body = kv_table(rows) + violation_table(self.violations)
        return panel(title="MRC", header_html=pass_fail_badge(self.passed), body_html=body)


@dataclass
class CurvilinearMRCResult:
    """Result of a curvilinear-specific Mask Rule Check.

    Curvilinear masks (post-ILT, EUV) cannot be validated with Manhattan-only
    rules. This adds two checks aimed at MBMW writability:
    - Minimum curvature radius (sharp cusps cannot be written).
    - Minimum feature area (sub-resolution dots cannot be reliably exposed).
    """

    passed: bool
    violation_count: int
    curvature_violations: list[dict[str, float]] = field(default_factory=list)
    area_violations: list[dict[str, float]] = field(default_factory=list)
    min_radius_observed_nm: float | None = None
    min_area_observed_nm2: float | None = None

    def _repr_html_(self) -> str:
        from openlithohub.jupyter._html import (
            kv_table,
            panel,
            pass_fail_badge,
            violation_table,
        )

        def _fmt(v: float | None, suffix: str) -> str:
            return f"{v:.3g} {suffix}" if v is not None else "—"

        rows = [
            ("Total violations", str(self.violation_count)),
            ("Curvature violations", str(len(self.curvature_violations))),
            ("Area violations", str(len(self.area_violations))),
            ("Min radius observed", _fmt(self.min_radius_observed_nm, "nm")),
            ("Min area observed", _fmt(self.min_area_observed_nm2, "nm²")),
        ]
        body = kv_table(rows)
        if self.curvature_violations:
            body += '<div style="margin-top:6px;font-size:90%;color:#555;">Curvature</div>'
            body += violation_table(self.curvature_violations)
        if self.area_violations:
            body += '<div style="margin-top:6px;font-size:90%;color:#555;">Area</div>'
            body += violation_table(self.area_violations)
        return panel(
            title="Curvilinear MRC",
            header_html=pass_fail_badge(self.passed),
            body_html=body,
        )


def check_mrc(
    mask: torch.Tensor,
    min_width_nm: float = 40.0,
    min_spacing_nm: float = 40.0,
    pixel_size_nm: float = 1.0,
) -> MRCResult:
    """Check mask against minimum width and spacing rules.

    MRC violations are a hard-fail metric — a mask that violates these rules
    cannot be manufactured regardless of optical performance.

    Width check: morphological opening with structuring element of size
    ``kernel = floor(min_width_nm / pixel_size_nm)`` (i.e. the largest disk
    that physically fits inside a feature of exactly ``min_width_nm``). The
    kernel half-width passed to ``binary_erosion`` is therefore
    ``(kernel - 1) // 2``. Features that disappear under this opening are
    width violations. A feature exactly ``min_width_nm`` wide passes.

    Spacing check: same logic on the inverted mask — gaps that disappear
    under opening are too narrow.

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

    radius_width = max(0, (int(math.floor(min_width_nm / pixel_size_nm)) - 1) // 2)
    radius_spacing = max(0, (int(math.floor(min_spacing_nm / pixel_size_nm)) - 1) // 2)

    width_violation_count = 0
    spacing_violation_count = 0

    if has_foreground and radius_width >= 1:
        opened = binary_dilation(binary_erosion(binary, radius=radius_width), radius=radius_width)
        width_violation_mask = (binary > 0.5) & (opened < 0.5)
        width_violation_count = int(width_violation_mask.sum().item())

        if width_violation_count > 0:
            fg_dist = distance_transform(binary)
            ys, xs = torch.where(width_violation_mask)
            _add_violations(violations, "width", ys, xs, fg_dist, pixel_size_nm, min_width_nm)

    if has_foreground and has_background and radius_spacing >= 1:
        bg = (binary < 0.5).float()
        eroded_bg = binary_erosion(bg, radius=radius_spacing)
        opened_bg = binary_dilation(eroded_bg, radius=radius_spacing)
        spacing_violation_mask = (bg > 0.5) & (opened_bg < 0.5)
        spacing_violation_count = int(spacing_violation_mask.sum().item())

        if spacing_violation_count > 0:
            bg_dist = distance_transform(bg)
            ys, xs = torch.where(spacing_violation_mask)
            _add_violations(violations, "spacing", ys, xs, bg_dist, pixel_size_nm, min_spacing_nm)

    violation_count = width_violation_count + spacing_violation_count
    violation_rate = violation_count / total_pixels if total_pixels > 0 else 0.0

    return MRCResult(
        passed=violation_count == 0,
        violation_count=violation_count,
        violation_rate=violation_rate,
        violations=violations,
        width_violation_count=width_violation_count,
        spacing_violation_count=spacing_violation_count,
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
    """Add up to ``max_reports`` evenly spaced violation samples.

    ``actual_nm`` reports the *local feature width* (or spacing) at each
    sample, derived from the chessboard distance transform: the distance
    map at a violation pixel equals half the local feature thickness, so
    the feature width is ``dist * 2 * pixel_size_nm``. To make the
    reported width robust against off-spine samples (a 10 nm line's edge
    pixel sits 1 px from background and would otherwise report 2 nm), we
    look up the *maximum* distance within each sample's local
    chessboard neighbourhood (radius = half the violating-region
    extent). For typical MRC violation rectangles this recovers the
    spine value.
    """
    total = int(len(ys))
    if total == 0:
        return
    indices = evenly_spaced_indices(total, max_reports)
    h, w = dist_map.shape
    # Walk the violation point list to a small max neighbourhood so the
    # spine of even a thin feature is hit. The neighbourhood radius is
    # bounded by the violating-region extent in pixel units (worst case:
    # the half-width of the threshold rule).
    nbhd_radius = max(1, int(math.ceil(threshold_nm / (2 * pixel_size_nm))))
    for idx in indices:
        y_px = int(ys[idx].item())
        x_px = int(xs[idx].item())
        y0 = max(0, y_px - nbhd_radius)
        y1 = min(h, y_px + nbhd_radius + 1)
        x0 = max(0, x_px - nbhd_radius)
        x1 = min(w, x_px + nbhd_radius + 1)
        local_max = float(dist_map[y0:y1, x0:x1].max().item())
        actual_nm = local_max * 2.0 * pixel_size_nm
        violations.append(
            {
                "type_code": 0.0 if vtype == "width" else 1.0,
                "x_nm": float(x_px) * pixel_size_nm,
                "y_nm": float(y_px) * pixel_size_nm,
                "actual_nm": actual_nm,
                "required_nm": threshold_nm,
            }
        )


def _smooth_loop(loop: np.ndarray, window: int) -> np.ndarray:
    """Periodic moving-average smoother for closed contour loops.

    Removes single-pixel rasterization aliasing before curvature estimation.
    """
    if window <= 1 or len(loop) < window:
        return loop
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.concatenate([loop[-pad:], loop, loop[:pad]], axis=0)
    sy = np.convolve(padded[:, 0], kernel, mode="valid")
    sx = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.stack([sy[: len(loop)], sx[: len(loop)]], axis=1)


def _connected_component_areas(binary: torch.Tensor) -> list[tuple[int, float, float]]:
    """4-connected components of a binary mask. Returns (area_px, cy, cx) per component.

    Uses GPU-vectorized labeling — orders of magnitude faster than per-pixel
    BFS for large masks.
    """
    labels, num = connected_components(binary, connectivity=4)
    if num == 0:
        return []

    fg = labels >= 0
    flat_labels = labels[fg]
    h, w = binary.shape
    ys, xs = torch.where(fg)

    unique_labels, inverse = torch.unique(flat_labels, return_inverse=True)
    counts = torch.zeros(unique_labels.numel(), dtype=torch.float64, device=binary.device)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float64))
    sum_y = torch.zeros(unique_labels.numel(), dtype=torch.float64, device=binary.device)
    sum_y.scatter_add_(0, inverse, ys.to(torch.float64))
    sum_x = torch.zeros(unique_labels.numel(), dtype=torch.float64, device=binary.device)
    sum_x.scatter_add_(0, inverse, xs.to(torch.float64))

    counts_cpu = counts.tolist()
    cy_cpu = (sum_y / counts).tolist()
    cx_cpu = (sum_x / counts).tolist()
    return [
        (int(counts_cpu[i]), float(cy_cpu[i]), float(cx_cpu[i]))
        for i in range(unique_labels.numel())
    ]


def check_curvilinear_mrc(
    mask: torch.Tensor,
    min_curvature_radius_nm: float = 20.0,
    min_feature_area_nm2: float = 1600.0,
    pixel_size_nm: float = 1.0,
    smoothing_window: int = 5,
    max_reports: int = 100,
) -> CurvilinearMRCResult:
    """Check curvilinear-specific manufacturing rules on a binary mask.

    Two rules, both targeting MBMW writability of post-ILT curvilinear shapes:

    1. Minimum curvature radius. The contour is traced, smoothed with a periodic
       moving average to suppress rasterization aliasing, then discrete curvature
       is computed at each point via the Menger (three-point circumscribed
       circle) formula. A point violates if its radius (1/|kappa|) falls below
       ``min_curvature_radius_nm``. The smoothing offset (``smoothing_window // 2``)
       skips evaluation near sharp 90 degree corners typical of Manhattan input,
       so right-angled designs do not falsely fail.
    2. Minimum feature area. 4-connected components below
       ``min_feature_area_nm2`` are flagged as sub-resolution dots.

    Args:
        mask: Binary mask tensor (H, W) or (B, C, H, W).
        min_curvature_radius_nm: Minimum allowed local radius of curvature.
        min_feature_area_nm2: Minimum allowed area for a connected feature.
        pixel_size_nm: Physical pixel size for unit conversion.
        smoothing_window: Window size for the periodic moving-average smoother.
            Set to 1 to disable smoothing. Larger values relax the curvature
            check; the default suits 1 nm/pixel ILT outputs.
        max_reports: Cap on per-category violation reports.

    Returns:
        CurvilinearMRCResult with pass/fail status and violation details.
    """
    m = ensure_2d(mask)
    binary_torch = (m > 0.5).float()
    binary_np = binary_torch.detach().cpu().numpy().astype(np.int8)

    curvature_violations: list[dict[str, float]] = []
    area_violations: list[dict[str, float]] = []
    min_radius_observed: float | None = None
    min_area_observed: float | None = None

    pixel_area_nm2 = pixel_size_nm * pixel_size_nm
    components = _connected_component_areas(binary_torch)
    for area_px, cy, cx in components:
        area_nm2 = area_px * pixel_area_nm2
        if min_area_observed is None or area_nm2 < min_area_observed:
            min_area_observed = area_nm2
        if area_nm2 < min_feature_area_nm2 and len(area_violations) < max_reports:
            area_violations.append(
                {
                    "x_nm": cx * pixel_size_nm,
                    "y_nm": cy * pixel_size_nm,
                    "actual_nm2": area_nm2,
                    "required_nm2": min_feature_area_nm2,
                }
            )

    if min_curvature_radius_nm > 0.0:
        loops = trace_contour(binary_np)
        threshold_kappa = 1.0 / max(min_curvature_radius_nm, 1e-9)
        # Curvature stencil span. A 3-point Menger estimate is only reliable
        # when the sampled arc length is comparable to the radius being
        # measured. Span ~ R / pi keeps the chord/sagitta ratio sane and
        # prevents rasterization aliasing on smooth curves from flagging
        # spurious tight radii.
        stencil = max(2, int(round(min_curvature_radius_nm / max(pixel_size_nm, 1e-9) / 3.0)))
        skip = max(stencil, smoothing_window // 2)
        for loop in loops:
            if len(loop) < max(2 * skip + 3, 5):
                continue
            loop_nm = _smooth_loop(loop, smoothing_window) * pixel_size_nm
            n = len(loop_nm)
            # Vectorized Menger curvature over the whole closed loop.
            idx = np.arange(n)
            p0 = loop_nm[(idx - skip) % n]
            p1 = loop_nm[idx]
            p2 = loop_nm[(idx + skip) % n]
            a = np.linalg.norm(p1 - p0, axis=1)
            b = np.linalg.norm(p2 - p1, axis=1)
            c = np.linalg.norm(p2 - p0, axis=1)
            cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
                p2[:, 0] - p0[:, 0]
            )
            denom = a * b * c
            valid = (a >= 1e-9) & (b >= 1e-9) & (c >= 1e-9) & (np.abs(cross) >= 1e-12)
            kappa = np.zeros(n, dtype=np.float64)
            np.divide(2.0 * np.abs(cross), denom, out=kappa, where=valid)

            valid_kappa = kappa[valid]
            if valid_kappa.size > 0:
                loop_min_radius = 1.0 / float(valid_kappa.max())
                if min_radius_observed is None or loop_min_radius < min_radius_observed:
                    min_radius_observed = loop_min_radius

            (violator_indices,) = np.nonzero(kappa > threshold_kappa)
            for vi in violator_indices:
                if len(curvature_violations) >= max_reports:
                    break
                radius_nm = 1.0 / float(kappa[vi])
                curvature_violations.append(
                    {
                        "x_nm": float(p1[vi, 1]),
                        "y_nm": float(p1[vi, 0]),
                        "actual_radius_nm": radius_nm,
                        "required_radius_nm": min_curvature_radius_nm,
                    }
                )

    violation_count = len(curvature_violations) + len(area_violations)
    return CurvilinearMRCResult(
        passed=violation_count == 0,
        violation_count=violation_count,
        curvature_violations=curvature_violations,
        area_violations=area_violations,
        min_radius_observed_nm=min_radius_observed,
        min_area_observed_nm2=min_area_observed,
    )
