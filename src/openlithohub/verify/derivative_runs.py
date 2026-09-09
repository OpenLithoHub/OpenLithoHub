"""Interval derivative run-prefix artifacts for B04 known-layout correction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .interface_runs import HorizontalRun, split_runs_outside_chebyshev


class JetComponent(str, Enum):
    VALUE = "value"
    DX = "dx"
    DY = "dy"
    DXX = "dxx"
    DXY = "dxy"
    DYY = "dyy"


@dataclass(frozen=True)
class ComplexIntervalVector:
    real_lo: np.ndarray
    real_hi: np.ndarray
    imag_lo: np.ndarray
    imag_hi: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.real_lo)
        if not all(len(v) == n for v in (self.real_hi, self.imag_lo, self.imag_hi)):
            raise ValueError("complex interval vector shapes do not match")
        if np.any(self.real_lo > self.real_hi) or np.any(self.imag_lo > self.imag_hi):
            raise ValueError("invalid complex interval vector")


@dataclass
class DerivativeRunPrefixArtifact:
    arrays: Mapping[str, np.ndarray]
    grid_n: int
    pixel_nm: float

    @classmethod
    def load(cls, path: str | Path) -> DerivativeRunPrefixArtifact:
        z = np.load(Path(path), allow_pickle=False)
        n = int(z["grid_n"])
        return cls(
            arrays={k: z[k] for k in z.files},
            grid_n=n,
            pixel_nm=float(z["pixel_nm"]),
        )

    def sum_far_runs(
        self,
        *,
        component: JetComponent,
        runs: Sequence[HorizontalRun],
        center_y: int,
        center_x: int,
        halo_px: int,
    ) -> ComplexIntervalVector:
        far = split_runs_outside_chebyshev(
            runs,
            center_y=center_y,
            center_x=center_x,
            halo_px=halo_px,
        )
        n = self.grid_n
        c = n // 2
        key = component.value
        rlo = self.arrays[f"{key}_real_lo"]
        rhi = self.arrays[f"{key}_real_hi"]
        ilo = self.arrays[f"{key}_imag_lo"]
        ihi = self.arrays[f"{key}_imag_hi"]
        k = rlo.shape[0]

        orl = np.zeros(k)
        orh = np.zeros(k)
        oil = np.zeros(k)
        oih = np.zeros(k)
        for run in far:
            yi = (c + center_y - run.y) % n
            length = run.length
            rstart = (c + center_x - run.x0) % n
            s0 = (-rstart) % n

            sl = np.nextafter(rlo[:, yi, s0 + length] - rhi[:, yi, s0], -np.inf)
            sh = np.nextafter(rhi[:, yi, s0 + length] - rlo[:, yi, s0], np.inf)
            il = np.nextafter(ilo[:, yi, s0 + length] - ihi[:, yi, s0], -np.inf)
            ih = np.nextafter(ihi[:, yi, s0 + length] - ilo[:, yi, s0], np.inf)

            orl = np.nextafter(orl + sl, -np.inf)
            orh = np.nextafter(orh + sh, np.inf)
            oil = np.nextafter(oil + il, -np.inf)
            oih = np.nextafter(oih + ih, np.inf)

        return ComplexIntervalVector(orl, orh, oil, oih)


@dataclass(frozen=True)
class CenterJetCertificate:
    status: str
    min_center_gradient_lower_per_nm: float
    continuous_cell_status: str
    generic_l3_upper_per_nm3: float


def center_jet_certificate_from_json(path: str | Path, *, halo_px: int) -> CenterJetCertificate:
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {int(r["h_px"]): r for r in raw["queries"]["halos"]}
    row = rows[halo_px]
    return CenterJetCertificate(
        status="PASS",
        min_center_gradient_lower_per_nm=float(row["min_center_gradient_lower_per_nm"]),
        continuous_cell_status="OPEN",
        generic_l3_upper_per_nm3=float(raw["new_first_failure"]["generic_K24_L3_upper_per_nm3"]),
    )
