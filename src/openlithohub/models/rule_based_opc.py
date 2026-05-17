"""Rule-based geometric OPC baseline.

Classic non-AI optical proximity correction by morphological edge biasing.
Serves as the geometric reference point that any learning-based method must
beat: shows what plain dilation buys you and what it leaves on the table
(line-end pull-back, corner rounding, MRC violations on tight pitch).

The model is intentionally simple — no SRAFs, no segment-level fragmentation,
no rule table per layer. Real production OPC engines do far more; this exists
so that the leaderboard always carries a "no learning" geometric baseline.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry


def _binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel_size = 2 * radius + 1
    inp = mask.float().unsqueeze(0).unsqueeze(0)
    dilated = functional.max_pool2d(inp, kernel_size=kernel_size, stride=1, padding=radius)
    return dilated.squeeze(0).squeeze(0)


def _line_end_mask(mask: torch.Tensor) -> torch.Tensor:
    """Pixels that look like a horizontal/vertical line end.

    A line-end pixel is foreground with foreground neighbours on exactly one
    of the four cardinal sides — a lone "tip" sticking out. Used to apply a
    small extra bias compensating for line-end pull-back.
    """
    fg = mask.float()
    pad = functional.pad(fg.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), value=0.0)
    up = pad[:, :, :-2, 1:-1].squeeze(0).squeeze(0)
    down = pad[:, :, 2:, 1:-1].squeeze(0).squeeze(0)
    left = pad[:, :, 1:-1, :-2].squeeze(0).squeeze(0)
    right = pad[:, :, 1:-1, 2:].squeeze(0).squeeze(0)
    neighbour_count = up + down + left + right
    return ((fg > 0.5) & (neighbour_count <= 1.0)).float()


@registry.register
class RuleBasedOPCModel(LithographyModel):
    """Geometric edge-bias OPC: dilate the design by a fixed radius."""

    def __init__(
        self,
        bias_radius_px: int = 1,
        line_end_extra_px: int = 1,
    ) -> None:
        self._bias_radius_px = bias_radius_px
        self._line_end_extra_px = line_end_extra_px

    @property
    def name(self) -> str:
        return "rule-based-opc"

    @property
    def supports_curvilinear(self) -> bool:
        return False

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        bias_radius = int(kwargs.get("bias_radius_px", self._bias_radius_px))
        line_end_extra = int(kwargs.get("line_end_extra_px", self._line_end_extra_px))

        target = design.float()
        if target.ndim > 2:
            target = target.squeeze()

        mask = _binary_dilate(target, bias_radius)

        if line_end_extra > 0 and target.sum() > 0:
            tips = _line_end_mask(target)
            if tips.sum() > 0:
                tip_dilated = _binary_dilate(tips, line_end_extra)
                mask = torch.maximum(mask, tip_dilated)

        binary = (mask > 0.5).float()

        return PredictionResult(
            mask=binary,
            metadata={
                "bias_radius_px": bias_radius,
                "line_end_extra_px": line_end_extra,
            },
        )
