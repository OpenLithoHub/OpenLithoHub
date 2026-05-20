"""Object-oriented façade over the OpenLithoHub functional API.

`Mask`, `LitheEngine`, and `Report` are thin wrappers around existing
internals (model registry, tile/halo/stitch pipeline, metric/compliance
functions). They exist for fab/EDA users who think in masks and engines
rather than tensors. The functional API is unchanged; everything here
delegates.
"""

from __future__ import annotations

from openlithohub.api.engine import LitheEngine
from openlithohub.api.mask import Mask
from openlithohub.api.report import Report

__all__ = ["LitheEngine", "Mask", "Report"]
