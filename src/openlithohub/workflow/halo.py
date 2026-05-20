"""Process-node-aware tile halo sizing (RFC 0005).

The halo is the per-tile guard band of overlapping pixels that lets the
forward lithography model see real layout context at tile boundaries
instead of zero-padded artefacts. Two physical phenomena set the lower
bound:

1. **Optical interaction radius (OIR)** — light at a tile boundary sees
   neighbours through the imaging kernel; carried by
   ``ProcessNodeConfig.optical_radius_nm``.
2. **Model receptive field** — convolutional models also propagate
   information across pixels; carried by
   ``LithographyModel.RECEPTIVE_FIELD_PX``.

This module centralises the math that picks ``max(OIR_px, RF_px)``,
rounds up to a stride-friendly multiple, and clamps to fit inside the
caller's tile size.
"""

from __future__ import annotations

import math

from openlithohub.models.base import LithographyModel
from openlithohub.workflow.process_node import ProcessNodeConfig

DEFAULT_HALO_PX = 128
"""Pre-RFC-0005 fixed halo. Used when neither node nor model is provided."""

_HALO_ROUND_PX = 8
"""Round halos up to a multiple of this so conv strides remain happy."""


def compute_halo_px(
    node: ProcessNodeConfig | None,
    model: LithographyModel | None,
    pixel_nm: float,
    tile_size: int,
) -> int:
    """Pick a tile halo big enough for the optical kernel and the model RF.

    Args:
        node: Process node carrying ``optical_radius_nm``. ``None`` means
            "no physical radius known" and contributes 0 to the max.
        model: Model carrying ``RECEPTIVE_FIELD_PX``. ``None`` contributes 0.
        pixel_nm: Pixel pitch in nanometers (used to convert OIR → pixels).
        tile_size: Tile width in pixels. The returned halo is clamped so
            that ``halo < tile_size`` (``tile_layout`` rejects otherwise).

    Returns:
        Halo size in pixels. ``DEFAULT_HALO_PX`` when both ``node`` and
        ``model`` are ``None`` (preserves pre-RFC-0005 behaviour).
    """
    if pixel_nm <= 0:
        raise ValueError(f"pixel_nm must be positive, got {pixel_nm}")
    if tile_size <= 1:
        raise ValueError(f"tile_size must be > 1, got {tile_size}")

    if node is None and model is None:
        return min(DEFAULT_HALO_PX, tile_size - 1)

    oir_px = math.ceil(node.optical_radius_nm / pixel_nm) if node is not None else 0
    rf_px = model.receptive_field_px if model is not None else 0

    raw = max(oir_px, rf_px)
    rounded = _round_up(raw, _HALO_ROUND_PX)
    return max(0, min(rounded, tile_size - 1))


def describe_halo(
    halo_px: int,
    node: ProcessNodeConfig | None,
    model: LithographyModel | None,
    pixel_nm: float,
) -> str:
    """One-line provenance string for the resolved halo, for CLI logging."""
    halo_nm = halo_px * pixel_nm
    parts: list[str] = []
    if node is not None:
        parts.append(f"{node.name} (OIR={node.optical_radius_nm:.0f} nm)")
    if model is not None:
        parts.append(f"{model.name} (RF={model.receptive_field_px} px)")
    if not parts:
        return f"{halo_px} px (≈{halo_nm:.0f} nm at {pixel_nm} nm/px) — fixed default"
    src = " + ".join(parts)
    return f"{halo_px} px (≈{halo_nm:.0f} nm at {pixel_nm} nm/px) — auto from {src}"


def _round_up(value: int, multiple: int) -> int:
    if value <= 0:
        return 0
    return ((value + multiple - 1) // multiple) * multiple
