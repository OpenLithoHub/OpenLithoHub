"""Layout-Tokens — polygon-level tokenisation for Manhattan layouts (RFC 0002 prototype).

Encodes a binary mask into a sequence of polygon vertex tokens that can
be fed to an autoregressive transformer, and decodes back to a binary
mask. Designed for exact round-trip on Manhattan masks at the chosen
canvas grid.

What this prototype is:
- A pure Python tokenizer with no torch ops in the encode/decode hot
  path (torch only used for the I/O tensor shape).
- Rectangle-decomposition strategy: each connected component is split
  into horizontal strips of constant column extent. Each strip is one
  4-vertex polygon. This is exact for Manhattan masks; for non-Manhattan
  masks it is a staircase approximation.
- Coordinate vocabulary derived from the canvas grid. The PDK is
  recorded only as a metadata tag — geometry is *not* snapped to PDK
  rules here; that belongs upstream of the tokenizer (the layout
  generator) or downstream (a DRC pass).

What it is NOT (yet):
- No transformer that consumes these tokens.
- No marching-squares / Douglas-Peucker contour path. The RFC mentions
  it for non-Manhattan; v0.2-beta scope.
- No multi-layer / hole tokens (RFC §Open questions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from openlithohub.synth.pdk import get_pdk

# Reserved special tokens.
TOK_BOS = 0
TOK_EOS = 1
TOK_POLYGON = 2
TOK_CLOSE = 3
N_RESERVED = 4


@dataclass
class TokenizedLayout:
    """Output of LayoutTokenizer.encode."""

    ids: torch.Tensor  # (T,) int64
    canvas_size: int  # pixels
    pdk_name: str  # informational tag only — geometry is not snapped to PDK rules


@dataclass
class DecodeReport:
    """Diagnostics from a decode pass.

    The fields surface model-output health to callers (autoregressive
    generation often emits malformed sequences early in training). A
    decode that yields ``polygons_drawn=0`` with non-zero
    ``polygons_skipped`` or ``unknown_tokens`` is a parser-level failure,
    not "the model emitted a blank layout".
    """

    polygons_drawn: int = 0
    polygons_skipped: int = 0  # malformed polygons (wrong vertex count, truncated)
    unknown_tokens: int = 0  # tokens outside the reserved + coord vocabulary
    out_of_range_coords: int = 0  # coord ids whose value falls outside [0, canvas_size]
    truncated: bool = False  # sequence ended before EOS

    @property
    def ok(self) -> bool:
        return (
            self.polygons_skipped == 0
            and self.unknown_tokens == 0
            and self.out_of_range_coords == 0
            and not self.truncated
        )


@dataclass
class DecodedLayout:
    """Output of LayoutTokenizer.decode — mask plus parse diagnostics."""

    mask: torch.Tensor
    report: DecodeReport = field(default_factory=DecodeReport)


class LayoutTokenizer:
    """Polygon-vertex tokenizer for binary layouts on a fixed canvas.

    The tokenizer operates purely on the canvas pixel grid; PDK rules are
    *not* enforced here. ``pdk_name`` is recorded on :class:`TokenizedLayout`
    so downstream consumers can route per-PDK postprocessing.
    """

    def __init__(self, canvas_size: int = 256, pdk_name: str = "freepdk45"):
        self.canvas_size = canvas_size
        self.pdk_name = pdk_name
        # Coordinate ids are offset by N_RESERVED so they never collide
        # with control tokens. With canvas_size pixels per axis we have
        # canvas_size + 1 distinct vertex coordinates (0..canvas_size).
        self.coord_offset = N_RESERVED
        self.vocab_size = N_RESERVED + (canvas_size + 1)

    @classmethod
    def from_pdk(cls, pdk_name: str, canvas_size: int = 256) -> LayoutTokenizer:
        # Validates that pdk_name is a known preset; raises KeyError otherwise.
        get_pdk(pdk_name)
        return cls(canvas_size=canvas_size, pdk_name=pdk_name)

    def _coord_id(self, coord: int) -> int:
        if not (0 <= coord <= self.canvas_size):
            raise ValueError(f"coordinate {coord} out of range [0, {self.canvas_size}]")
        return self.coord_offset + coord

    def _coord_value(self, token_id: int) -> int:
        return int(token_id) - self.coord_offset

    def _is_coord_token(self, token_id: int) -> bool:
        return self.coord_offset <= int(token_id) < self.vocab_size

    def encode(self, mask: torch.Tensor) -> TokenizedLayout:
        """Mask (H, W) → token sequence. H == W == canvas_size required."""
        if mask.ndim != 2:
            raise ValueError(f"expected (H, W), got shape {tuple(mask.shape)}")
        h, w = mask.shape
        if h != self.canvas_size or w != self.canvas_size:
            raise ValueError(f"expected {self.canvas_size}x{self.canvas_size} mask, got {h}x{w}")
        binary = (mask > 0.5).to(torch.uint8).numpy()

        ids: list[int] = [TOK_BOS]
        for rect in _rectangle_decomposition(binary):
            ids.append(TOK_POLYGON)
            # Vertices clockwise from top-left corner (y0, x0).
            y0, x0, y1, x1 = rect
            for vy, vx in ((y0, x0), (y0, x1), (y1, x1), (y1, x0)):
                ids.extend([self._coord_id(vy), self._coord_id(vx)])
            ids.append(TOK_CLOSE)
        ids.append(TOK_EOS)

        return TokenizedLayout(
            ids=torch.tensor(ids, dtype=torch.int64),
            canvas_size=self.canvas_size,
            pdk_name=self.pdk_name,
        )

    def decode(self, ids: torch.Tensor | TokenizedLayout) -> DecodedLayout:
        """Token sequence → :class:`DecodedLayout` (mask + parse report).

        The mask is ``(canvas_size, canvas_size)`` float32 in ``[0, 1]``.
        Parse anomalies (unknown tokens, malformed polygons, out-of-range
        coordinates) populate :class:`DecodeReport` so callers can
        distinguish "model emitted an empty layout" from "we couldn't parse it".
        """
        if isinstance(ids, TokenizedLayout):
            ids = ids.ids
        seq = [int(t) for t in ids.tolist()]
        mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        report = DecodeReport()
        i = 0
        if seq and seq[0] == TOK_BOS:
            i += 1
        saw_eos = False
        while i < len(seq):
            t = seq[i]
            if t == TOK_EOS:
                saw_eos = True
                break
            if t != TOK_POLYGON:
                # Stray pre-polygon token. Reserved control tokens here are
                # the parser's "noise"; coord tokens with no <polygon> opener
                # also count as unknown in this position.
                report.unknown_tokens += 1
                i += 1
                continue
            i += 1
            verts: list[tuple[int, int]] = []
            polygon_bad = False
            while i < len(seq) and seq[i] != TOK_CLOSE:
                if seq[i] == TOK_EOS:
                    polygon_bad = True
                    break
                if i + 1 >= len(seq):
                    polygon_bad = True
                    break
                t_y, t_x = seq[i], seq[i + 1]
                if not (self._is_coord_token(t_y) and self._is_coord_token(t_x)):
                    report.unknown_tokens += 2
                    polygon_bad = True
                    i += 2
                    continue
                vy = self._coord_value(t_y)
                vx = self._coord_value(t_x)
                if not (0 <= vy <= self.canvas_size and 0 <= vx <= self.canvas_size):
                    report.out_of_range_coords += 1
                    polygon_bad = True
                    i += 2
                    continue
                verts.append((vy, vx))
                i += 2
            if i < len(seq) and seq[i] == TOK_CLOSE:
                i += 1
            else:
                # Reached EOS or end-of-stream before a CLOSE.
                polygon_bad = True
            if polygon_bad or len(verts) != 4:
                report.polygons_skipped += 1
                continue
            # Manhattan rectangle: bbox-fill it.
            ys = [v[0] for v in verts]
            xs = [v[1] for v in verts]
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            mask[y0:y1, x0:x1] = 1
            report.polygons_drawn += 1

        report.truncated = not saw_eos
        return DecodedLayout(mask=torch.from_numpy(mask).to(torch.float32), report=report)


def _rectangle_decomposition(binary: np.ndarray[Any, Any]) -> list[tuple[int, int, int, int]]:
    """Greedy horizontal-strip decomposition of a binary mask.

    Returns a list of axis-aligned rectangles ``(y0, x0, y1, x1)`` whose
    union equals the foreground. Half-open intervals: pixel (y, x) is
    foreground iff y0 <= y < y1 and x0 <= x < x1.

    The decomposition merges vertically adjacent rows whose run pattern
    is identical, so a long vertical bar emits a single rectangle.
    """
    h, w = binary.shape
    rects: list[tuple[int, int, int, int]] = []

    # Encode each row as a tuple of (start, end) runs so we can merge
    # rows with identical run patterns.
    def _row_runs(row: np.ndarray[Any, Any]) -> tuple[tuple[int, int], ...]:
        runs: list[tuple[int, int]] = []
        in_run = False
        start = 0
        for x in range(w):
            if row[x]:
                if not in_run:
                    start = x
                    in_run = True
            elif in_run:
                runs.append((start, x))
                in_run = False
        if in_run:
            runs.append((start, w))
        return tuple(runs)

    # Compute each row's run pattern once up front; the previous version
    # recomputed `_row_runs(binary[y2])` on every iteration of the merge
    # loop (O(h*w) per outer step, O(h^2 * w) total worst case).
    row_runs_cache: list[tuple[tuple[int, int], ...]] = [_row_runs(binary[y]) for y in range(h)]

    y = 0
    while y < h:
        runs = row_runs_cache[y]
        if not runs:
            y += 1
            continue
        # Find the largest y2 such that rows y..y2-1 all share these runs.
        y2 = y + 1
        while y2 < h and row_runs_cache[y2] == runs:
            y2 += 1
        for x0, x1 in runs:
            rects.append((y, x0, y2, x1))
        y = y2
    return rects
