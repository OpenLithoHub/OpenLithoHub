"""Proof-facing exact-vector mask semantics for B04.

The theorem chain uses a single canonical mask representation:

    GDS/OASIS
      -> exact parser-aligned vector geometry
      -> canonical horizontal occupancy runs
      -> normalized discrete Fourier coefficients
      -> Hopkins/SOCS optical coefficients.

The legacy dense/PIL loader is intentionally absent from this path.  Therefore
there is no *representation bridge term* between the theorem mask and the
optical mask: both are the same exact-vector pixel-center indicator.

This does **not** claim zero error to an arbitrary continuous polygon model.
Coordinate quantization and the pixel-center indicator convention remain
explicit theorem assumptions/provenance.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from openlithohub.verify.interface_runs import HorizontalRun
from openlithohub.verify.run_spectrum import RunSpectrumAccumulator


class ExactRunSource(Protocol):
    @property
    def layout_hash(self) -> str: ...

    @property
    def pixel_size_nm(self) -> float: ...

    @property
    def backend_kind(self) -> str: ...

    def iter_runs_for_bbox(
        self,
        bbox_px: tuple[int, int, int, int],
    ) -> Iterable[HorizontalRun]: ...


@dataclass(frozen=True)
class ExactVectorMaskContract:
    layout_hash: str
    backend_kind: str
    pixel_size_nm: float
    bbox_xyxy: tuple[int, int, int, int]
    semantics: str = "EXACT_VECTOR_PIXEL_CENTER_INDICATOR"
    dense_loader_used: bool = False
    representation_bridge_upper: float = 0.0

    def __post_init__(self) -> None:
        if self.pixel_size_nm <= 0:
            raise ValueError("pixel_size_nm must be positive")
        x0, y0, x1, y1 = self.bbox_xyxy
        if x1 <= x0 or y1 <= y0:
            raise ValueError("invalid theorem bbox")
        if self.dense_loader_used:
            raise ValueError("proof-facing exact-vector contract forbids the legacy dense loader")
        if self.representation_bridge_upper != 0.0:
            raise ValueError("exact-vector optical path has zero internal representation bridge")


@dataclass(frozen=True)
class ExactVectorSpectrum:
    normalized_spectrum: np.ndarray
    run_count: int
    occupied_pixels: int
    contract: ExactVectorMaskContract

    @property
    def grid_n(self) -> int:
        return int(self.normalized_spectrum.shape[0])


def exact_vector_normalized_spectrum(
    source: ExactRunSource,
    *,
    bbox_xyxy: tuple[int, int, int, int],
) -> ExactVectorSpectrum:
    """Build normalized DFT coefficients directly from canonical source runs.

    The supplied bbox must be square.  Global runs are translated into the
    local Fourier tile before accumulation; no dense mask is materialized.
    """
    x0, y0, x1, y1 = bbox_xyxy
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("invalid bbox")
    if width != height:
        raise ValueError("Fourier proof tile must be square")

    raw_runs = tuple(source.iter_runs_for_bbox(bbox_xyxy))
    local_runs: list[HorizontalRun] = []
    occupied = 0
    for run in raw_runs:
        local = HorizontalRun(
            y=int(run.y) - y0,
            x0=int(run.x0) - x0,
            x1=int(run.x1) - x0,
        )
        if not (0 <= local.y < height):
            raise ValueError("source returned a run outside requested y-range")
        if not (0 <= local.x0 < local.x1 <= width):
            raise ValueError("source returned a run outside requested x-range")
        local_runs.append(local)
        occupied += local.x1 - local.x0

    acc = RunSpectrumAccumulator.create(width)
    acc.extend(local_runs)
    coeff = acc.normalized_spectrum()

    contract = ExactVectorMaskContract(
        layout_hash=source.layout_hash,
        backend_kind=source.backend_kind,
        pixel_size_nm=source.pixel_size_nm,
        bbox_xyxy=bbox_xyxy,
    )
    return ExactVectorSpectrum(
        normalized_spectrum=coeff,
        run_count=len(local_runs),
        occupied_pixels=occupied,
        contract=contract,
    )


def materialized_reference_spectrum(mask: np.ndarray) -> np.ndarray:
    """Diagnostic/reference DFT of the *same* exact-vector indicator mask.

    This is permitted only as an algebraic test oracle.  It is not the legacy
    GDS/OASIS dense loader.
    """
    arr = np.asarray(mask, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("mask must be square")
    n = arr.shape[0]
    out: np.ndarray = np.fft.fft2(arr) / (n * n)
    return out


def exact_vector_representation_bridge_upper(
    contract: ExactVectorMaskContract,
) -> float:
    """Return the internal representation-bridge term for the theorem path."""
    return contract.representation_bridge_upper
