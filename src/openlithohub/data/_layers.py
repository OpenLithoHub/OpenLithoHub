"""Per-PDK design-layer registry.

The post-route layer numbering for "metal1" (and friends) differs across
PDKs and even between source vs post-route variants of the same PDK:

- ASAP7 cell library (``asap7sc7p5t_27/GDS/...``): M1 = (10, 0)
- FreePDK45 + NanGate stdcells (``mflowgen/freepdk-45nm``): M1 = (11, 0)
- ORFS-routed ASAP7 platform (``flow/platforms/asap7``): M1 = (20, 0)

Carrying three module-level ``DEFAULT_DESIGN_LAYER`` constants invited
the (10, 0) / (20, 0) docstring drift caught in the May 2026 review.
This single registry is the source of truth; PDK adapters import from
it and re-export ``DEFAULT_DESIGN_LAYER`` so call sites stay stable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PdkLayers:
    """Layer numbers for a single PDK / variant.

    Only the fields the rest of the codebase actually reads are populated —
    add more as they're needed. ``metal1`` is the only mandatory one
    because every adapter rasterizes it as the design tensor by default.
    """

    metal1: tuple[int, int]


# Keys are the same identifiers the adapters use as module names so the
# pairing is obvious.
LAYERS: Mapping[str, PdkLayers] = {
    # ASAP7 cell-library source (BSD-3-Clause, asap7sc7p5t_27 submodule).
    "asap7": PdkLayers(metal1=(10, 0)),
    # FreePDK45 + NanGate Open Cell Library via mflowgen mirror.
    "freepdk45": PdkLayers(metal1=(11, 0)),
    # ORFS-routed ASAP7 (post-route platform stream-out map). Distinct
    # from ``asap7`` above — see module docstring of openlithohub.data.orfs
    # for why M1 jumps from 10/0 to 20/0 after routing.
    "orfs_asap7": PdkLayers(metal1=(20, 0)),
}
