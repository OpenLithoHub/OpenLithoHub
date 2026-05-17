"""OASIS/GDSII export coordination."""

from __future__ import annotations

from pathlib import Path

import torch


def export_oasis(
    mask: torch.Tensor,
    output_path: str | Path,
    *,
    mode: str = "curvilinear",
    pixel_size_nm: float = 1.0,
) -> None:
    """Export an optimized mask tensor to OASIS format.

    Args:
        mask: Optimized mask tensor (H, W).
        output_path: Destination path for the .oas file.
        mode: 'manhattan' for VSB writers or 'curvilinear' for MBMW.
        pixel_size_nm: Physical pixel size for coordinate scaling.
    """
    raise NotImplementedError(
        "OASIS export not yet implemented. "
        "Planned: extract contours (manhattan or curvilinear mode), "
        "serialize via appropriate writer (KLayout for manhattan, "
        "custom OASIS.MBW serializer for curvilinear)."
    )
