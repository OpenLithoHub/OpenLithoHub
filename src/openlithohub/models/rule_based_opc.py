"""Rule-based geometric OPC baseline.

Classic non-AI optical proximity correction by morphological edge biasing.
Serves as the geometric reference point that any learning-based method must
beat: shows what plain dilation buys you and what it leaves on the table
(line-end pull-back, corner rounding, MRC violations on tight pitch).

This implementation deliberately stays in the "no learning, no simulation"
regime. It augments uniform dilation with three context-aware bias modes
that are still pure geometry — directional hammerheads at line ends,
serifs at concave inner corners, and an iso/dense bias split — plus an
MRC self-check that reports (and optionally retreats from) violations
caused by the bias itself. Everything runs on torch.nn.functional pooling;
no scipy, no distance transforms.

Future work (intentionally out of scope here, to keep this a "minimal
geometric baseline"):
- SRAF placement (sub-resolution assist features) — needs distance transforms
  and per-process-node assist rules.
- Segment-based fragmentation — break edges into segments and move each
  independently, the standard production OPC primitive.
- Iterative bias loop (model-based OPC) — dilate → simulate → measure EPE
  → adjust bias. Crossing this line stops being "rule-based" by definition.
- Per-layer rule tables — different metals / poly / via layers carry
  different bias tables in real flows.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from openlithohub._utils.morphology import binary_dilation, binary_erosion
from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry


def _binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return binary_dilation(mask, radius) if radius > 0 else mask


def _line_end_mask(mask: torch.Tensor) -> torch.Tensor:
    """Pixels that look like a horizontal/vertical line end.

    A line-end pixel is foreground with foreground neighbours on exactly one
    of the four cardinal sides — a lone "tip" sticking out. Used by the
    legacy isotropic line-end bias path (kept for backwards compatibility
    when ``directional_line_end=False``).
    """
    fg = mask.float()
    pad = functional.pad(fg.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), value=0.0)
    up = pad[:, :, :-2, 1:-1].squeeze(0).squeeze(0)
    down = pad[:, :, 2:, 1:-1].squeeze(0).squeeze(0)
    left = pad[:, :, 1:-1, :-2].squeeze(0).squeeze(0)
    right = pad[:, :, 1:-1, 2:].squeeze(0).squeeze(0)
    neighbour_count = up + down + left + right
    return ((fg > 0.5) & (neighbour_count <= 1.0)).float()


def _neighbours(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return four boolean-valued float tensors: up, down, left, right neighbours."""
    fg = mask.float()
    pad = functional.pad(fg.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), value=0.0)
    up = pad[:, :, :-2, 1:-1].squeeze(0).squeeze(0)
    down = pad[:, :, 2:, 1:-1].squeeze(0).squeeze(0)
    left = pad[:, :, 1:-1, :-2].squeeze(0).squeeze(0)
    right = pad[:, :, 1:-1, 2:].squeeze(0).squeeze(0)
    return up, down, left, right


def _directional_line_end(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-direction line-end masks: (up_tips, down_tips, left_tips, right_tips).

    A "right tip" is a foreground pixel whose only foreground neighbour is
    on the left — the line extends leftward and ends here. Symmetric for
    the other three directions. A pixel with zero neighbours qualifies for
    none of the four (it's an isolated dot, not a line end).
    """
    fg = (mask.float() > 0.5).float()
    up, down, left, right = _neighbours(fg)
    neighbour_count = up + down + left + right
    exactly_one = neighbour_count == 1.0
    is_fg = fg > 0.5
    right_tips = (is_fg & exactly_one & (left > 0.5)).float()
    left_tips = (is_fg & exactly_one & (right > 0.5)).float()
    down_tips = (is_fg & exactly_one & (up > 0.5)).float()
    up_tips = (is_fg & exactly_one & (down > 0.5)).float()
    return up_tips, down_tips, left_tips, right_tips


def _directional_dilate(mask: torch.Tensor, radius: int, direction: str) -> torch.Tensor:
    """Dilate ``mask`` by ``radius`` pixels in a single cardinal direction.

    ``direction`` ∈ {"up","down","left","right"}. Implemented as a 1×k or
    k×1 max-pool with asymmetric padding so the output extends only on
    the requested side.
    """
    if radius <= 0:
        return mask
    inp = mask.float().unsqueeze(0).unsqueeze(0)
    if direction == "right":
        padded = functional.pad(inp, (radius, 0, 0, 0))
        out = functional.max_pool2d(padded, kernel_size=(1, radius + 1), stride=1)
    elif direction == "left":
        padded = functional.pad(inp, (0, radius, 0, 0))
        out = functional.max_pool2d(padded, kernel_size=(1, radius + 1), stride=1)
    elif direction == "down":
        padded = functional.pad(inp, (0, 0, radius, 0))
        out = functional.max_pool2d(padded, kernel_size=(radius + 1, 1), stride=1)
    elif direction == "up":
        padded = functional.pad(inp, (0, 0, 0, radius))
        out = functional.max_pool2d(padded, kernel_size=(radius + 1, 1), stride=1)
    else:
        raise ValueError(f"unknown direction: {direction!r}")
    return out.squeeze(0).squeeze(0)


def _inner_corner_mask(design: torch.Tensor) -> torch.Tensor:
    """Background pixels sitting in the concave notch of an L-shape.

    Detected as ``fg == 0`` with foreground neighbours on exactly two
    *adjacent* cardinal sides (e.g. up+left, not up+down). This is the
    classic concave-corner geometry where a serif compensates for resist
    rounding pull-back into the inside of the angle.
    """
    fg = (design.float() > 0.5).float()
    bg = 1.0 - fg
    up, down, left, right = _neighbours(fg)
    # adjacent corner pairs: (up,left), (up,right), (down,left), (down,right)
    ul = (up > 0.5) & (left > 0.5)
    ur = (up > 0.5) & (right > 0.5)
    dl = (down > 0.5) & (left > 0.5)
    dr = (down > 0.5) & (right > 0.5)
    any_adjacent_pair = ul | ur | dl | dr
    return ((bg > 0.5) & any_adjacent_pair).float()


def _compute_density(mask: torch.Tensor, window_px: int) -> torch.Tensor:
    """Per-pixel local foreground density via average pooling."""
    if window_px <= 1:
        return mask.float()
    inp = mask.float().unsqueeze(0).unsqueeze(0)
    pad = window_px // 2
    out = functional.avg_pool2d(inp, kernel_size=window_px, stride=1, padding=pad)
    return out.squeeze(0).squeeze(0)


def _min_run_length(mask: torch.Tensor) -> int:
    """Smallest run of consecutive 1s along any row or column.

    Returns 0 if the mask has no foreground pixels at all. Used by both
    ``_min_width_px`` (called on the mask) and ``_min_space_px`` (called
    on its complement). Pure torch — implemented by counting cumulative
    runs row-by-row, then column-by-column.
    """
    binary = (mask.float() > 0.5).to(torch.int64)
    if binary.sum().item() == 0:
        return 0
    best = _min_run_in_lines(binary)
    best_t = _min_run_in_lines(binary.t().contiguous())
    if best == 0:
        return best_t
    if best_t == 0:
        return best
    return min(best, best_t)


def _min_run_in_lines(binary: torch.Tensor) -> int:
    """Smallest non-zero run of 1s along the last dim."""
    h, w = binary.shape
    runs = torch.zeros_like(binary)
    runs[:, 0] = binary[:, 0]
    for j in range(1, w):
        runs[:, j] = (runs[:, j - 1] + 1) * binary[:, j]
    # ends of runs: position where the next pixel is 0 (or end of row)
    is_end = torch.zeros_like(binary, dtype=torch.bool)
    is_end[:, :-1] = (runs[:, :-1] > 0) & (binary[:, 1:] == 0)
    is_end[:, -1] = runs[:, -1] > 0
    end_runs = runs[is_end]
    if end_runs.numel() == 0:
        return 0
    return int(end_runs.min().item())


def _min_space_px(mask: torch.Tensor) -> int:
    """Smallest run of background pixels touching foreground on both sides.

    A space is a run of 0s flanked by 1s on the same row or column. Pure
    background runs that touch the border (i.e. open spaces around the
    pattern) don't count — only enclosed gaps matter for MRC.
    """
    binary = (mask.float() > 0.5).to(torch.int64)
    if binary.sum().item() == 0:
        return 0
    best_row = _min_enclosed_zero_run(binary)
    best_col = _min_enclosed_zero_run(binary.t().contiguous())
    candidates = [v for v in (best_row, best_col) if v > 0]
    if not candidates:
        return 0
    return min(candidates)


def _min_enclosed_zero_run(binary: torch.Tensor) -> int:
    """Smallest run of 0s along the last dim that has 1s on both sides."""
    h, w = binary.shape
    if w < 3:
        return 0
    inv = 1 - binary
    runs = torch.zeros_like(inv)
    runs[:, 0] = inv[:, 0]
    for j in range(1, w):
        runs[:, j] = (runs[:, j - 1] + 1) * inv[:, j]
    # a run "ends" at column j when binary[:, j+1] == 1 (right side closed)
    # and is enclosed when the column immediately before its start is 1
    # (left side closed). The start column is j - run_len + 1; checking
    # binary at start - 1 = j - run_len handles that.
    best = 0
    for j in range(w - 1):
        end_here = (runs[:, j] > 0) & (binary[:, j + 1] == 1)
        if not end_here.any():
            continue
        for row in torch.nonzero(end_here, as_tuple=False).flatten().tolist():
            run_len = int(runs[row, j].item())
            start = j - run_len + 1
            if start - 1 < 0:
                continue
            if binary[row, start - 1].item() != 1:
                continue
            if best == 0 or run_len < best:
                best = run_len
    return best


def _min_width_px(mask: torch.Tensor) -> int:
    """Smallest foreground run length on any row or column."""
    return _min_run_length(mask)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary erosion via the shared morphology helper, returned as {0,1} float."""
    if radius <= 0:
        return mask
    return (binary_erosion(mask, radius) > 0.5).float()


@registry.register
class RuleBasedOPCModel(LithographyModel):
    """Geometric edge-bias OPC with directional hammerheads and inner-corner serifs.

    Default behaviour is conservative: uniform 1px dilation plus directional
    line-end extension. Inner-corner serifs, iso/dense split, and MRC retreat
    are all opt-in via constructor or per-call kwargs.
    """

    def __init__(
        self,
        bias_radius_px: int = 1,
        line_end_extra_px: int = 1,
        inner_corner_extra_px: int = 0,
        directional_line_end: bool = True,
        iso_radius_px: int | None = None,
        dense_radius_px: int | None = None,
        density_window_px: int = 9,
        density_threshold: float = 0.25,
        mrc_min_space_px: int = 0,
    ) -> None:
        self._bias_radius_px = bias_radius_px
        self._line_end_extra_px = line_end_extra_px
        self._inner_corner_extra_px = inner_corner_extra_px
        self._directional_line_end = directional_line_end
        self._iso_radius_px = iso_radius_px
        self._dense_radius_px = dense_radius_px
        self._density_window_px = density_window_px
        self._density_threshold = density_threshold
        self._mrc_min_space_px = mrc_min_space_px

    @property
    def name(self) -> str:
        return "rule-based-opc"

    @property
    def supports_curvilinear(self) -> bool:
        return False

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        bias_radius = int(kwargs.get("bias_radius_px", self._bias_radius_px))
        line_end_extra = int(kwargs.get("line_end_extra_px", self._line_end_extra_px))
        inner_corner_extra = int(
            kwargs.get("inner_corner_extra_px", self._inner_corner_extra_px)
        )
        directional = bool(kwargs.get("directional_line_end", self._directional_line_end))
        iso_radius = kwargs.get("iso_radius_px", self._iso_radius_px)
        dense_radius = kwargs.get("dense_radius_px", self._dense_radius_px)
        density_window = int(kwargs.get("density_window_px", self._density_window_px))
        density_threshold = float(kwargs.get("density_threshold", self._density_threshold))
        mrc_min_space = int(kwargs.get("mrc_min_space_px", self._mrc_min_space_px))

        bias_radius_nm = kwargs.get("bias_radius_nm")
        pixel_size_nm = kwargs.get("pixel_size_nm")
        if bias_radius_nm is not None or pixel_size_nm is not None:
            if bias_radius_nm is None or pixel_size_nm is None:
                raise ValueError(
                    "bias_radius_nm and pixel_size_nm must be provided together"
                )
            if pixel_size_nm <= 0:
                raise ValueError("pixel_size_nm must be positive")
            bias_radius = int(round(float(bias_radius_nm) / float(pixel_size_nm)))

        target = design.float()
        if target.ndim > 2:
            target = target.squeeze()

        if iso_radius is not None and dense_radius is not None:
            mask = self._iso_dense_dilate(
                target,
                int(iso_radius),
                int(dense_radius),
                density_window,
                density_threshold,
            )
        else:
            mask = _binary_dilate(target, bias_radius)

        n_line_end_tips = 0
        if line_end_extra > 0 and target.sum() > 0:
            if directional:
                up_tips, down_tips, left_tips, right_tips = _directional_line_end(target)
                tip_total = up_tips + down_tips + left_tips + right_tips
                n_line_end_tips = int((tip_total > 0).sum().item())
                if n_line_end_tips > 0:
                    extension = torch.maximum(
                        torch.maximum(
                            _directional_dilate(up_tips, line_end_extra, "up"),
                            _directional_dilate(down_tips, line_end_extra, "down"),
                        ),
                        torch.maximum(
                            _directional_dilate(left_tips, line_end_extra, "left"),
                            _directional_dilate(right_tips, line_end_extra, "right"),
                        ),
                    )
                    mask = torch.maximum(mask, extension)
            else:
                tips = _line_end_mask(target)
                n_line_end_tips = int(tips.sum().item())
                if n_line_end_tips > 0:
                    mask = torch.maximum(mask, _binary_dilate(tips, line_end_extra))

        n_inner_corners = 0
        if inner_corner_extra > 0 and target.sum() > 0:
            corners = _inner_corner_mask(target)
            n_inner_corners = int(corners.sum().item())
            if n_inner_corners > 0:
                mask = torch.maximum(mask, _binary_dilate(corners, inner_corner_extra))

        binary = (mask > 0.5).float()

        min_space = _min_space_px(binary)
        min_width = _min_width_px(binary)
        mrc_violated = mrc_min_space > 0 and 0 < min_space < mrc_min_space

        if mrc_violated:
            binary = self._retreat_until_mrc_clean(binary, mrc_min_space)
            min_space = _min_space_px(binary)
            min_width = _min_width_px(binary)
            mrc_violated = 0 < min_space < mrc_min_space

        design_sum = float(target.sum().item())
        mask_area_growth = (
            float(binary.sum().item()) / design_sum if design_sum > 0 else 0.0
        )
        bias_radius_nm_meta = (
            float(bias_radius_nm) if bias_radius_nm is not None else None
        )

        return PredictionResult(
            mask=binary,
            metadata={
                "bias_radius_px": bias_radius,
                "line_end_extra_px": line_end_extra,
                "inner_corner_extra_px": inner_corner_extra,
                "directional_line_end": directional,
                "iso_radius_px": iso_radius,
                "dense_radius_px": dense_radius,
                "bias_radius_nm": bias_radius_nm_meta,
                "mask_area_growth": mask_area_growth,
                "n_line_end_tips": n_line_end_tips,
                "n_inner_corners": n_inner_corners,
                "min_space_px": min_space,
                "min_width_px": min_width,
                "mrc_min_space_px": mrc_min_space,
                "mrc_violated": mrc_violated,
                "output_geometry": "manhattan",
            },
        )

    def _iso_dense_dilate(
        self,
        target: torch.Tensor,
        iso_radius: int,
        dense_radius: int,
        window_px: int,
        threshold: float,
    ) -> torch.Tensor:
        density = _compute_density(target, window_px)
        is_iso = density < threshold
        iso_mask = _binary_dilate(target, iso_radius)
        dense_mask = _binary_dilate(target, dense_radius)
        combined = torch.where(is_iso, iso_mask, dense_mask)
        return combined

    def _retreat_until_mrc_clean(
        self, mask: torch.Tensor, min_space_px: int, max_iters: int = 8
    ) -> torch.Tensor:
        current = mask
        for _ in range(max_iters):
            space = _min_space_px(current)
            if space == 0 or space >= min_space_px:
                return current
            current = _erode(current, 1)
            if current.sum().item() == 0:
                return current
        return current
