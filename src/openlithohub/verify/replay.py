"""Replay helpers for theorem-facing B04 proof artifacts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExpandedBandReplay:
    active_cells: int
    worst_cell_yx: tuple[int, int]
    kappa_lower_per_nm: float
    hausdorff_upper_nm: float
    pairwise_contour_diameter_upper_nm: float


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _minabs(lo: float, hi: float) -> float:
    return 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))


def _grad_bounds(gx: tuple[float, float], gy: tuple[float, float]) -> tuple[float, float]:
    gl = math.hypot(_minabs(*gx), _minabs(*gy))
    gu = math.hypot(max(abs(gx[0]), abs(gx[1])), max(abs(gy[0]), abs(gy[1])))
    return math.nextafter(gl, -math.inf), math.nextafter(gu, math.inf)


def replay_expanded_band(
    grid_npz: str | Path,
    *,
    threshold: float,
    pixel_nm: float,
    spatial_third_derivative_upper_per_nm3: float,
    delta_total: float,
) -> ExpandedBandReplay:
    rows = np.load(grid_npz)["rows"]
    if rows.ndim != 2 or rows.shape[1] != 14:
        raise ValueError(f"expected rows shape (n, 14); got {rows.shape}")
    d = pixel_nm / math.sqrt(2.0)
    active: list[tuple[int, int]] = []
    worst: tuple[int, int] | None = None
    worst_kappa = math.inf

    for row in rows:
        y, x = int(row[0]), int(row[1])
        vals = [(float(row[2 + 2 * i]), float(row[3 + 2 * i])) for i in range(6)]
        jb, gxb, gyb, hxxb, hxyb, hyyb = vals
        flo, fhi = jb[0] - threshold, jb[1] - threshold
        gl, gu = _grad_bounds(gxb, gyb)
        hxxu = max(abs(hxxb[0]), abs(hxxb[1]))
        hxyu = max(abs(hxyb[0]), abs(hxyb[1]))
        hyyu = max(abs(hyyb[0]), abs(hyyb[1]))
        hcenter = math.sqrt(hxxu * hxxu + 2.0 * hxyu * hxyu + hyyu * hyyu)
        hcell = math.nextafter(hcenter + spatial_third_derivative_upper_per_nm3 * d, math.inf)
        radius = math.nextafter(gu * d + 0.5 * hcell * d * d, math.inf)
        kappa = math.nextafter(gl - hcell * d, -math.inf)
        if flo - radius <= delta_total and fhi + radius >= -delta_total:
            active.append((y, x))
            if kappa < worst_kappa:
                worst_kappa = kappa
                worst = (y, x)

    if not active or worst is None:
        raise ValueError("expanded band is empty; certificate semantics unresolved")
    if worst_kappa <= 0.0:
        raise ValueError("expanded-band transversality is not certified")

    radius = delta_total / worst_kappa
    return ExpandedBandReplay(
        active_cells=len(active),
        worst_cell_yx=worst,
        kappa_lower_per_nm=worst_kappa,
        hausdorff_upper_nm=radius,
        pairwise_contour_diameter_upper_nm=2.0 * radius,
    )
