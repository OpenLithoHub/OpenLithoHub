"""Streaming tile sinks (RFC 0008, prompt §3).

A ``TileSink`` receives each tile's trusted-core result.  Implementations
decide the output materialisation:

- :class:`TensorTileSink` — assemble a full dense tensor (small layouts,
  backward-compatible behaviour).
- :class:`MemmapTileSink` — page the output to an on-disk memmap
  (out-of-core big rasters).
- :class:`MetricOnlyTileSink` — keep only per-tile metrics/aggregates and
  never materialise a raster at all (the QDM-critical mode).
- :class:`StreamingManhattanTileSink` — convert each trusted binary core
  to Manhattan rectangles exactly once and insert them into a KLayout
  cell (bounded-memory mask-writer output; O(core + rectangle count),
  never O(full-chip raster)).

Because only trusted cores are ever written, no weight map / blend pass
is required: cores are an exact partition of the output domain.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from .geometry import BoundingBox


@runtime_checkable
class TileSink(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    def write_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def finalize(self) -> Any: ...


class CertifiedCommitNotRepresentableError(ValueError):
    """A sink was asked to commit a certified core it cannot represent.

    R17 C2: a verifier PASS (or a screening fact) without a known output
    tensor is a *verification-only* disposition.  Metric-only sinks can
    record it; tensor-materializing sinks must reject it, because
    ``verifier PASS does not imply a known output tensor``.
    """


class TensorTileSink:
    """Assemble the full dense output tensor in host memory."""

    def __init__(self, shape: tuple[int, int], dtype: torch.dtype = torch.float32) -> None:
        self._shape = shape
        self.output = torch.zeros(shape, dtype=dtype)
        self._written: set[str] = set()

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def write_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if tile_id in self._written:
            raise ValueError(f"duplicate core write for {tile_id!r}")
        self._written.add(tile_id)
        expected = (bbox.height, bbox.width)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"core tensor shape {tuple(tensor.shape)} != bbox {expected}")
        self.output[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1] = tensor

    def record_certified_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        *,
        exact_fill: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """R17 C2 certified commit: synthesize the exact fill, or reject."""
        if exact_fill is None:
            raise CertifiedCommitNotRepresentableError(
                "verification-only skip cannot be represented in a tensor "
                "output sink; verifier PASS does not imply a known tensor"
            )
        fill = torch.full((bbox.height, bbox.width), float(exact_fill), dtype=self.output.dtype)
        self.write_core(tile_id, bbox, fill, metadata)

    def finalize(self) -> torch.Tensor:
        return self.output


class MemmapTileSink:
    """Page trusted cores straight into an on-disk memmap raster.

    With ``npy=True`` the backing file is created via
    ``np.lib.format.open_memmap`` so the artifact is a standard,
    self-describing ``.npy`` that downstream readers can re-open with
    ``mmap_mode="r"``; the default keeps the historical raw raster.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        path: str | Path,
        dtype: np.dtype[Any] | None = None,
        *,
        npy: bool = False,
    ) -> None:
        self._shape = shape
        self.path = Path(path)
        if dtype is None:
            dtype = np.dtype(np.float32)
        if npy:
            self._map: np.memmap = np.lib.format.open_memmap(
                self.path, mode="w+", dtype=dtype, shape=shape
            )
        else:
            self._map = np.memmap(self.path, dtype=dtype, mode="w+", shape=shape)
        self._written: set[str] = set()

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def write_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if tile_id in self._written:
            raise ValueError(f"duplicate core write for {tile_id!r}")
        self._written.add(tile_id)
        window = tensor.detach().cpu().numpy().astype(self._map.dtype, copy=False)
        self._map[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1] = window

    def record_certified_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        *,
        exact_fill: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """R17 C2 certified commit: synthesize the exact fill, or reject."""
        if exact_fill is None:
            raise CertifiedCommitNotRepresentableError(
                "verification-only skip cannot be represented in a memmap "
                "output sink; verifier PASS does not imply a known tensor"
            )
        fill = torch.full((bbox.height, bbox.width), float(exact_fill), dtype=torch.float32)
        self.write_core(tile_id, bbox, fill, metadata)

    def finalize(self) -> np.memmap:
        self._map.flush()
        return self._map


class MetricOnlyTileSink:
    """Aggregate per-tile results without ever materialising a raster.

    Reducers receive one tile at a time, so peak memory is O(1) beyond the
    running aggregates.  This is the sink mode QDM-style verification needs
    (certificates, worst bounds, violation lists, provenance).
    """

    def __init__(self, shape: tuple[int, int]) -> None:
        self._shape = shape
        self.tiles: list[str] = []
        self.covered_pixels = 0
        self.metrics: dict[str, float] = {}
        self.metadata: dict[str, Any] = {}
        self.certified_records: dict[str, str] = {}
        self.certified_pixels = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def write_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.tiles.append(tile_id)
        self.covered_pixels += bbox.area
        self.metadata[tile_id] = metadata or {}
        for key, value in (metadata or {}).items():
            if isinstance(value, (int, float)):
                prev = self.metrics.get(f"max_{key}", float("-inf"))
                self.metrics[f"max_{key}"] = max(prev, float(value))

    def record_certified_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        *,
        exact_fill: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """R17 C2 tensor-free certified commit (the QDM-critical mode).

        No raster is materialised for a certified core — with or without an
        exact output fill.  ``exact_fill`` is recorded as provenance only.
        """
        disposition = "EXACT_OUTPUT_SKIP" if exact_fill is not None else "VERIFICATION_ONLY_SKIP"
        meta = dict(metadata or {})
        meta.setdefault("screen_disposition", disposition)
        if exact_fill is not None:
            meta.setdefault("certified_exact_fill", float(exact_fill))
        self.write_core(tile_id, bbox, torch.empty(0), meta)
        self.certified_records[tile_id] = disposition
        self.certified_pixels += bbox.area

    def finalize(self) -> dict[str, Any]:
        return {
            "n_tiles": len(self.tiles),
            "covered_pixels": self.covered_pixels,
            "metrics": dict(self.metrics),
        }


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(x0, x1)`` foreground runs of one binary row."""
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev + 1))
            start = prev = i
    runs.append((start, prev + 1))
    return runs


def _dbu_for_pixel(pixel_size_nm: float) -> tuple[int, float]:
    """Pick ``(pixels_per_dbu, dbu_um)`` so pixel edges land on integer DBU.

    KLayout stores geometry in integer database units; the sink needs
    every raster pixel edge at an exact DBU coordinate.  Scaling by
    powers of ten keeps ``pixels_per_dbu`` integral in both directions
    and keeps the database unit inside KLayout's sane range.
    """
    if pixel_size_nm <= 0:
        raise ValueError(f"pixel_size_nm must be positive, got {pixel_size_nm}")
    pixels_per_dbu = 1
    dbu_nm = float(pixel_size_nm)
    while dbu_nm > 10.0:  # keep dbu <= 0.01 um
        dbu_nm /= 10.0
        pixels_per_dbu *= 10
    while dbu_nm < 0.01:  # keep dbu >= 1e-5 um; pixel spans more DBU instead
        dbu_nm *= 10.0
        pixels_per_dbu *= 10
    return pixels_per_dbu, dbu_nm / 1000.0


class StreamingManhattanTileSink:
    """Bounded-memory Manhattan mask-writer output (PR-C Gate C2).

    Each trusted core arrives already binarised; the sink run-length
    encodes it row by row, vertically merges identical runs into boxes
    and inserts them into a KLayout cell.  The semantic property that
    makes this exact: cores are an exact partition of the output domain,
    so **each trusted core is converted to owned Manhattan geometry
    exactly once** — there is no blending pass and no full-chip raster
    anywhere in the pipeline.  Peak memory is O(core + rectangle count),
    never O(full-layout area).

    The raster-to-geometry mapping is y-flipped (row 0 is the top of the
    raster, KLayout's y axis points up) and quantised on integer DBU via
    :func:`_dbu_for_pixel`.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        path: str | Path,
        *,
        pixel_size_nm: float = 1.0,
        layer: tuple[int, int] = (1, 0),
    ) -> None:
        try:
            import klayout.db as db  # noqa: PLC0415 — optional dependency
        except ImportError:
            raise ImportError(
                "klayout is required for streaming Manhattan OASIS output. "
                "Install with: pip install openlithohub[workflow]"
            ) from None
        self._db = db
        self._shape = shape
        self.path = Path(path)
        self._pixels_per_dbu, self._dbu_um = _dbu_for_pixel(pixel_size_nm)
        self._pixel_size_nm = float(pixel_size_nm)
        layout = db.Layout()
        layout.dbu = self._dbu_um
        self._layout = layout
        self._cell = layout.create_cell("TOP")
        self._shapes = self._cell.shapes(layout.layer(int(layer[0]), int(layer[1])))
        self._written: set[str] = set()
        self.n_rectangles = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def _insert_box(self, x0: int, y_start: int, x1: int, y_end: int) -> None:
        ppd = self._pixels_per_dbu
        height = self._shape[0]
        box = self._db.Box(
            x0 * ppd,
            (height - y_end) * ppd,
            x1 * ppd,
            (height - y_start) * ppd,
        )
        self._shapes.insert(box)
        self.n_rectangles += 1

    def write_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if tile_id in self._written:
            raise ValueError(f"duplicate core write for {tile_id!r}")
        self._written.add(tile_id)
        arr = tensor.detach().cpu().numpy()
        expected = (bbox.height, bbox.width)
        if arr.shape != expected:
            raise ValueError(f"core tensor shape {arr.shape} != bbox {expected}")
        vals = np.unique(arr)
        if vals.size and not bool(np.isin(vals, (0.0, 1.0)).all()):
            raise ValueError(
                "StreamingManhattanTileSink needs binarised cores (values in {0.0, 1.0}); "
                "apply the threshold upstream"
            )
        # Vertical run merging: identical x-runs on consecutive rows extend
        # the open box; a changed row flushes it.  Open runs are bounded by
        # the core width, closed boxes by the mask geometry — both O(core).
        open_runs: dict[tuple[int, int], int] = {}
        x0_global = bbox.x0
        y0_global = bbox.y0
        for row in range(arr.shape[0]):
            runs = _row_runs(arr[row])
            run_set = set(runs)
            for key in list(open_runs):
                if key not in run_set:
                    y_end = y0_global + row
                    self._insert_box(
                        x0_global + key[0], open_runs.pop(key), x0_global + key[1], y_end
                    )
            for key in runs:
                if key not in open_runs:
                    open_runs[key] = y0_global + row
        for key, y_start in open_runs.items():
            self._insert_box(
                x0_global + key[0],
                y_start,
                x0_global + key[1],
                y0_global + arr.shape[0],
            )

    def record_certified_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        *,
        exact_fill: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """R17 C2 certified commit: constant cores become 0 or 1 rectangles."""
        if tile_id in self._written:
            raise ValueError(f"duplicate core write for {tile_id!r}")
        self._written.add(tile_id)
        if exact_fill is None:
            raise CertifiedCommitNotRepresentableError(
                "verification-only skip cannot be represented in Manhattan "
                "geometry; verifier PASS does not imply a known output tensor"
            )
        if float(exact_fill) == 1.0:
            self._insert_box(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        elif float(exact_fill) != 0.0:
            raise CertifiedCommitNotRepresentableError(
                f"exact fill {exact_fill!r} is not representable in a binary mask"
            )

    def finalize(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._layout.write(str(self.path))
        return self.path


def covered_pixels(sink: TileSink) -> int:
    """Total trusted-core pixels written (for gap/overlap audits)."""
    written = getattr(sink, "covered_pixels", None)
    if written is not None:
        return int(written)
    written = getattr(sink, "_written", None)
    if isinstance(written, Iterable):
        return len(written)  # type: ignore[arg-type]
    return -1
