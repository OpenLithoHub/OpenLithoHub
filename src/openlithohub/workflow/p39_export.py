"""SEMI P39 integration for OASIS and GDS export paths."""

from __future__ import annotations

from pathlib import Path

import torch

from openlithohub.workflow.export import export_gds, export_oasis
from openlithohub.workflow.layer_purpose import LayerPurpose
from openlithohub.workflow.semi_p39 import P39Mapper


def apply_p39_to_oasis(
    mask: torch.Tensor,
    output_path: str | Path,
    mapper: P39Mapper,
    *,
    layer: int = 1,
    purpose_name: str = "drawing",
    mode: str = "curvilinear",
    pixel_size_nm: float = 1.0,
    min_area_nm2: float = 0.0,
) -> None:
    mapper.to_semi(LayerPurpose.from_name(layer, purpose_name))
    export_oasis(
        mask,
        output_path,
        mode=mode,
        pixel_size_nm=pixel_size_nm,
        min_area_nm2=min_area_nm2,
    )


def apply_p39_to_gds(
    mask: torch.Tensor,
    output_path: str | Path,
    mapper: P39Mapper,
    *,
    layer: int = 1,
    purpose_name: str = "drawing",
    mode: str = "curvilinear",
    pixel_size_nm: float = 1.0,
    samples_per_curve: int = 64,
    min_area_nm2: float = 0.0,
) -> None:
    mapper.to_semi(LayerPurpose.from_name(layer, purpose_name))
    export_gds(
        mask,
        output_path,
        mode=mode,
        pixel_size_nm=pixel_size_nm,
        samples_per_curve=samples_per_curve,
        min_area_nm2=min_area_nm2,
    )
