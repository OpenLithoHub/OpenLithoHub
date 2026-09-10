"""Finite crop view over an exact-vector streaming source.

This adapter exposes a rectangular region of interest as its own local finite
layout without materializing the parent raster.  Local pixel coordinates start
at (0,0); all exact run queries are translated to the parent domain and then
translated back.

The crop is useful for large real-layout verification experiments where the
parent GDS block is much larger than the benchmark/verification region.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from openlithohub.verify.interface_runs import HorizontalRun

from .geometry import BoundingBox


class ExactRunTileSource(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def pixel_size_nm(self) -> float: ...

    @property
    def layout_hash(self) -> str: ...

    @property
    def backend_kind(self) -> str: ...

    def iter_runs_for_bbox(
        self,
        bbox_px: tuple[int, int, int, int],
    ) -> Iterable[HorizontalRun]: ...

    def read_window(self, bbox: BoundingBox) -> torch.Tensor: ...


@dataclass(frozen=True)
class CropProvenance:
    parent_layout_hash: str
    parent_backend_kind: str
    parent_shape_px: tuple[int, int]
    crop_bbox_parent_px: tuple[int, int, int, int]
    pixel_size_nm: float


class ExactVectorCropSource:
    """Local-coordinate finite crop over a parent exact run source."""

    backend_kind = "exact-vector-crop"

    def __init__(
        self,
        parent: ExactRunTileSource,
        crop_bbox_parent: BoundingBox,
    ) -> None:
        h, w = parent.shape
        if (
            crop_bbox_parent.x0 < 0
            or crop_bbox_parent.y0 < 0
            or crop_bbox_parent.x1 > w
            or crop_bbox_parent.y1 > h
        ):
            raise ValueError("crop escapes parent finite layout")
        if crop_bbox_parent.width <= 0 or crop_bbox_parent.height <= 0:
            raise ValueError("crop must have positive area")
        self._parent = parent
        self._crop = crop_bbox_parent
        self._shape = (
            crop_bbox_parent.height,
            crop_bbox_parent.width,
        )
        payload = (
            f"{parent.layout_hash}|"
            f"{crop_bbox_parent.x0},{crop_bbox_parent.y0},"
            f"{crop_bbox_parent.x1},{crop_bbox_parent.y1}"
        ).encode()
        self._layout_hash = hashlib.sha256(payload).hexdigest()

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def pixel_size_nm(self) -> float:
        return float(self._parent.pixel_size_nm)

    @property
    def layout_hash(self) -> str:
        return self._layout_hash

    @property
    def crop_bbox_parent(self) -> BoundingBox:
        return self._crop

    @property
    def provenance(self) -> CropProvenance:
        return CropProvenance(
            parent_layout_hash=self._parent.layout_hash,
            parent_backend_kind=self._parent.backend_kind,
            parent_shape_px=self._parent.shape,
            crop_bbox_parent_px=(
                self._crop.x0,
                self._crop.y0,
                self._crop.x1,
                self._crop.y1,
            ),
            pixel_size_nm=self.pixel_size_nm,
        )

    def _validate_local(self, bbox: BoundingBox) -> None:
        h, w = self._shape
        if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > w or bbox.y1 > h:
            raise ValueError("local bbox escapes crop")

    def _to_parent(self, bbox: BoundingBox) -> BoundingBox:
        self._validate_local(bbox)
        return BoundingBox(
            bbox.x0 + self._crop.x0,
            bbox.y0 + self._crop.y0,
            bbox.x1 + self._crop.x0,
            bbox.y1 + self._crop.y0,
        )

    def iter_runs_for_bbox(
        self,
        bbox_px: tuple[int, int, int, int],
    ) -> Iterable[HorizontalRun]:
        local = BoundingBox(*bbox_px)
        parent_box = self._to_parent(local)
        raw = self._parent.iter_runs_for_bbox(
            (
                parent_box.x0,
                parent_box.y0,
                parent_box.x1,
                parent_box.y1,
            )
        )
        return tuple(
            HorizontalRun(
                y=int(run.y) - self._crop.y0,
                x0=int(run.x0) - self._crop.x0,
                x1=int(run.x1) - self._crop.x0,
            )
            for run in raw
        )

    def read_window(self, bbox: BoundingBox) -> torch.Tensor:
        parent_box = self._to_parent(bbox)
        return self._parent.read_window(parent_box)

    def verification_metadata(
        self,
        bbox: BoundingBox,
        *,
        tile_id: str,
        core_bbox: BoundingBox,
    ) -> dict[str, Any]:
        parent_box = self._to_parent(bbox)
        parent_core = self._to_parent(core_bbox)
        metadata: dict[str, Any] = {
            "crop_layout_hash": self.layout_hash,
            "crop_bbox_parent_px": (
                self._crop.x0,
                self._crop.y0,
                self._crop.x1,
                self._crop.y1,
            ),
            "read_bbox_parent_px": (
                parent_box.x0,
                parent_box.y0,
                parent_box.x1,
                parent_box.y1,
            ),
            "core_bbox_parent_px": (
                parent_core.x0,
                parent_core.y0,
                parent_core.x1,
                parent_core.y1,
            ),
        }
        parent_meta = getattr(self._parent, "verification_metadata", None)
        if callable(parent_meta):
            metadata["parent"] = parent_meta(
                parent_box,
                tile_id=tile_id,
                core_bbox=parent_core,
            )
        return metadata
