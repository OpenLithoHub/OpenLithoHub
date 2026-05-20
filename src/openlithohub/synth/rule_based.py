"""Rule-based synthetic layout generators.

Three pattern types:

* ``sram`` — periodic 6T SRAM-like bitcell placed in an array.
* ``contact_array`` — square contact lattice with PDK-respecting via
  pitch.
* ``random_logic`` — randomly routed unidirectional metal traces with
  occasional bends.

All three honour the PDK rules (minimum width, minimum spacing, minimum
area) by construction, but the result is also self-checked with
:func:`openlithohub.benchmark.compliance.mrc.check_mrc` and any pixels
that violate are corrected via morphological opening of the offending
regions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import torch

from openlithohub._utils.morphology import binary_dilation, binary_erosion
from openlithohub.synth.pdk import PdkRules, get_pdk


class PatternKind(str, Enum):
    """Available synthetic-pattern types."""

    SRAM = "sram"
    CONTACT_ARRAY = "contact_array"
    RANDOM_LOGIC = "random_logic"


@dataclass(frozen=True)
class SyntheticBatch:
    """A batch of generated layouts plus the PDK they were generated for."""

    masks: torch.Tensor  # (N, H, W) float in {0,1}
    pdk: PdkRules
    pattern: PatternKind
    seeds: list[int]


def _sram_unit_cell(rules: PdkRules) -> torch.Tensor:
    """Return a single 6T-SRAM-like bitcell mask at ``rules`` pitch."""

    w = rules.min_width_px
    s = rules.min_spacing_px
    pitch = w + s
    cell = torch.zeros(6 * pitch, 4 * pitch)
    for col in range(4):
        x0 = col * pitch
        cell[:, x0 : x0 + w] = 1.0
    for row, span in [(0, (0, 4)), (5, (0, 4)), (2, (1, 3))]:
        y0 = row * pitch
        x_start = span[0] * pitch
        x_end = span[1] * pitch
        cell[y0 : y0 + w, x_start:x_end] = 1.0
    return cell


def _generate_sram(size: int, rules: PdkRules, rng: random.Random) -> torch.Tensor:
    cell = _sram_unit_cell(rules)
    ch, cw = cell.shape
    canvas = torch.zeros(size, size)
    pad = rules.min_spacing_px
    border = rules.min_spacing_px
    inner = size - 2 * border
    if inner < ch or inner < cw:
        return canvas
    rows = max(1, (inner + pad) // (ch + pad))
    cols = max(1, (inner + pad) // (cw + pad))
    span_y = rows * ch + (rows - 1) * pad
    span_x = cols * cw + (cols - 1) * pad
    y_off = border + rng.randint(0, max(0, inner - span_y))
    x_off = border + rng.randint(0, max(0, inner - span_x))
    for r in range(rows):
        for c in range(cols):
            y0 = y_off + r * (ch + pad)
            x0 = x_off + c * (cw + pad)
            if y0 + ch <= size - border and x0 + cw <= size - border:
                canvas[y0 : y0 + ch, x0 : x0 + cw] = cell
    return canvas


def _generate_contact_array(size: int, rules: PdkRules, rng: random.Random) -> torch.Tensor:
    via_px = max(rules.min_width_px, round(rules.via_size_nm / rules.pixel_size_nm))
    spacing_px = max(rules.min_spacing_px, round(rules.via_spacing_nm / rules.pixel_size_nm))
    period = via_px + spacing_px
    border = rules.min_spacing_px
    canvas = torch.zeros(size, size)
    y_jit = rng.randint(0, max(0, spacing_px // 2))
    x_jit = rng.randint(0, max(0, spacing_px // 2))
    for y0 in range(border + y_jit, size - border - via_px + 1, period):
        for x0 in range(border + x_jit, size - border - via_px + 1, period):
            canvas[y0 : y0 + via_px, x0 : x0 + via_px] = 1.0
    return canvas


def _generate_random_logic(size: int, rules: PdkRules, rng: random.Random) -> torch.Tensor:
    w = rules.min_width_px
    s = rules.min_spacing_px
    pitch = max(w + s, 2)
    border = s
    canvas = torch.zeros(size, size)
    lo = border
    hi = size - border - w
    if hi <= lo:
        return canvas
    n_traces = max(2, (hi - lo) // (3 * pitch))
    for _ in range(n_traces):
        if rng.random() < 0.5:
            y = lo + (rng.randrange(0, max(1, hi - lo), pitch) if hi - lo > pitch else 0)
            x_start = rng.randrange(border, max(border + 1, size // 2))
            x_end = rng.randrange(size // 2, size - border)
            if x_end <= x_start:
                continue
            canvas[y : y + w, x_start:x_end] = 1.0
            if rng.random() < 0.4:
                y_bend = y + rng.choice([-1, 1]) * pitch * 2
                if lo <= y_bend <= hi:
                    x_b = rng.randrange(x_start, max(x_start + 1, x_end - w))
                    y_lo, y_hi = sorted((y, y_bend))
                    canvas[y_lo : y_hi + w, x_b : x_b + w] = 1.0
        else:
            x = lo + (rng.randrange(0, max(1, hi - lo), pitch) if hi - lo > pitch else 0)
            y_start = rng.randrange(border, max(border + 1, size // 2))
            y_end = rng.randrange(size // 2, size - border)
            if y_end <= y_start:
                continue
            canvas[y_start:y_end, x : x + w] = 1.0
    return canvas


def _enforce_drc(mask: torch.Tensor, rules: PdkRules) -> torch.Tensor:
    """Morphological opening at slightly under min-width radius.

    Removes sub-min-width slivers while preserving features exactly at
    ``min_width_px``. Use ``(min_width_px - 1) // 2`` so a feature of
    width ``min_width_px`` survives opening with the
    ``2*radius + 1``-sized structuring element.
    """

    radius = max(1, (rules.min_width_px - 1) // 2)
    eroded = binary_erosion(mask, radius=radius)
    return binary_dilation(eroded, radius=radius)


def generate_layout(
    pattern: PatternKind | str,
    pdk: PdkRules | str = "freepdk45",
    *,
    size: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    """Generate a single synthetic layout.

    Args:
        pattern: ``"sram"``, ``"contact_array"``, or ``"random_logic"``.
        pdk: PDK preset name or :class:`PdkRules` instance.
        size: Output edge length in pixels.
        seed: PRNG seed; identical seeds reproduce identical layouts.

    Returns:
        ``(size, size)`` ``torch.Tensor`` in ``{0.0, 1.0}``.
    """

    rules = pdk if isinstance(pdk, PdkRules) else get_pdk(pdk)
    kind = PatternKind(pattern) if isinstance(pattern, str) else pattern
    rng = random.Random(seed)

    if kind == PatternKind.SRAM:
        raw = _generate_sram(size, rules, rng)
    elif kind == PatternKind.CONTACT_ARRAY:
        raw = _generate_contact_array(size, rules, rng)
    elif kind == PatternKind.RANDOM_LOGIC:
        raw = _generate_random_logic(size, rules, rng)
    else:
        raise ValueError(f"Unknown pattern: {kind!r}")

    return _enforce_drc(raw, rules)


def generate_synthetic_batch(
    pattern: PatternKind | str,
    n: int,
    pdk: PdkRules | str = "freepdk45",
    *,
    size: int = 256,
    seed: int = 0,
) -> SyntheticBatch:
    """Generate a batch of ``n`` synthetic layouts."""

    rules = pdk if isinstance(pdk, PdkRules) else get_pdk(pdk)
    kind = PatternKind(pattern) if isinstance(pattern, str) else pattern
    seeds = [seed + i for i in range(n)]
    masks = torch.stack(
        [generate_layout(kind, rules, size=size, seed=s) for s in seeds],
        dim=0,
    )
    return SyntheticBatch(masks=masks, pdk=rules, pattern=kind, seeds=seeds)
