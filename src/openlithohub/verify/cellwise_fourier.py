"""Layout-conditioned finite-Fourier cell envelopes for B04."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LayoutFourierEnvelope:
    coeff_mid: np.ndarray
    coeff_rad: np.ndarray
    frequency_index: np.ndarray
    grid_n: int
    pixel_nm: float

    @classmethod
    def load(cls, path: str | Path) -> LayoutFourierEnvelope:
        z = np.load(Path(path), allow_pickle=False)
        return cls(
            coeff_mid=z["intensity_coeff_mid"].astype(np.complex128),
            coeff_rad=z["intensity_coeff_rad"].astype(np.float64),
            frequency_index=z["frequency_index"].astype(int),
            grid_n=int(z["grid_n"]),
            pixel_nm=float(z["pixel_nm"]),
        )

    @property
    def domain_nm(self) -> float:
        return self.grid_n * self.pixel_nm

    def _qabs(self) -> tuple[np.ndarray, np.ndarray]:
        q = 2.0 * np.pi * self.frequency_index / self.domain_nm
        qabs = np.nextafter(np.abs(q), np.inf)
        return (
            np.broadcast_to(qabs[None, :], self.coeff_mid.shape),
            np.broadcast_to(qabs[:, None], self.coeff_mid.shape),
        )

    def hessian_frobenius_upper_per_nm2(self) -> float:
        qx, qy = self._qabs()
        aa = np.nextafter(np.abs(self.coeff_mid) + self.coeff_rad, np.inf)
        mxx = np.nextafter(np.sum(aa * qx * qx), np.inf)
        mxy = np.nextafter(np.sum(aa * qx * qy), np.inf)
        myy = np.nextafter(np.sum(aa * qy * qy), np.inf)
        return float(np.nextafter(math.sqrt(mxx * mxx + 2.0 * mxy * mxy + myy * myy), np.inf))

    def third_derivative_tensor_upper_per_nm3(self) -> float:
        qx, qy = self._qabs()
        aa = np.nextafter(np.abs(self.coeff_mid) + self.coeff_rad, np.inf)
        a = np.nextafter(np.sum(aa * qx**3), np.inf)
        b = np.nextafter(np.sum(aa * qx * qx * qy), np.inf)
        c = np.nextafter(np.sum(aa * qx * qy * qy), np.inf)
        d = np.nextafter(np.sum(aa * qy**3), np.inf)
        return float(np.nextafter(math.sqrt(a * a + 3.0 * b * b + 3.0 * c * c + d * d), np.inf))


@dataclass(frozen=True)
class CellTransversalityTransfer:
    center_gradient_lower_per_nm: float
    hessian_upper_per_nm2: float
    cell_radius_nm: float
    cell_gradient_lower_per_nm: float

    @property
    def passes(self) -> bool:
        return self.cell_gradient_lower_per_nm > 0.0


def transfer_center_gradient_to_cell(
    *,
    center_gradient_lower_per_nm: float,
    hessian_upper_per_nm2: float,
    cell_radius_nm: float,
) -> CellTransversalityTransfer:
    if min(center_gradient_lower_per_nm, hessian_upper_per_nm2, cell_radius_nm) < 0:
        raise ValueError("gradient/Hessian/radius inputs must be non-negative")
    lower = float(
        np.nextafter(
            center_gradient_lower_per_nm - hessian_upper_per_nm2 * cell_radius_nm,
            -np.inf,
        )
    )
    return CellTransversalityTransfer(
        center_gradient_lower_per_nm=center_gradient_lower_per_nm,
        hessian_upper_per_nm2=hessian_upper_per_nm2,
        cell_radius_nm=cell_radius_nm,
        cell_gradient_lower_per_nm=lower,
    )


@dataclass(frozen=True)
class LocalCellGeometry:
    center_gradient_lower_per_nm: float
    center_hessian_upper_per_nm2: float
    third_derivative_upper_per_nm3: float
    cell_radius_nm: float
    cell_hessian_upper_per_nm2: float
    cell_gradient_lower_per_nm: float
    hidden_loop_curvature_scale_nm: float

    @property
    def passes_transversality(self) -> bool:
        return self.cell_gradient_lower_per_nm > 0.0

    def excludes_hidden_loop(self, *, cell_diameter_nm: float) -> bool:
        return self.passes_transversality and self.hidden_loop_curvature_scale_nm > cell_diameter_nm


def local_cell_geometry(
    *,
    center_gradient_lower_per_nm: float,
    center_hessian_upper_per_nm2: float,
    third_derivative_upper_per_nm3: float,
    cell_radius_nm: float,
) -> LocalCellGeometry:
    vals = (
        center_gradient_lower_per_nm,
        center_hessian_upper_per_nm2,
        third_derivative_upper_per_nm3,
        cell_radius_nm,
    )
    if min(vals) < 0:
        raise ValueError("local geometry inputs must be non-negative")
    h = float(
        np.nextafter(
            center_hessian_upper_per_nm2 + third_derivative_upper_per_nm3 * cell_radius_nm,
            np.inf,
        )
    )
    k = float(np.nextafter(center_gradient_lower_per_nm - h * cell_radius_nm, -np.inf))
    scale = float("-inf") if k <= 0 else float(np.nextafter(2.0 * k / h, -np.inf))
    return LocalCellGeometry(
        center_gradient_lower_per_nm=center_gradient_lower_per_nm,
        center_hessian_upper_per_nm2=center_hessian_upper_per_nm2,
        third_derivative_upper_per_nm3=third_derivative_upper_per_nm3,
        cell_radius_nm=cell_radius_nm,
        cell_hessian_upper_per_nm2=h,
        cell_gradient_lower_per_nm=k,
        hidden_loop_curvature_scale_nm=scale,
    )
