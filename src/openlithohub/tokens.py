"""Layout-Tokens — polygon-level tokenisation for Manhattan layouts (RFC 0002 prototype).

Encodes a binary mask into a sequence of polygon vertex tokens that can
be fed to an autoregressive transformer, and decodes back to a binary
mask. Designed for exact round-trip on Manhattan masks at the chosen
PDK grid.

What this prototype is:
- A pure Python tokenizer with no torch ops in the encode/decode hot
  path (torch only used for the I/O tensor shape).
- Rectangle-decomposition strategy: each connected component is split
  into horizontal strips of constant column extent. Each strip is one
  4-vertex polygon. This is exact for Manhattan masks; for non-Manhattan
  masks it is a staircase approximation.
- Coordinate vocabulary built from the canvas grid.

What it is NOT (yet):
- No transformer that consumes these tokens.
- No marching-squares / Douglas-Peucker contour path. The RFC mentions
  it for non-Manhattan; v0.2-beta scope.
- No multi-layer / hole tokens (RFC §Open questions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from openlithohub.synth.pdk import PdkRules, get_pdk

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
    pdk_name: str


class LayoutTokenizer:
    """Polygon-vertex tokenizer for binary layouts on a fixed canvas."""

    def __init__(self, pdk: PdkRules, canvas_size: int = 256):
        self.pdk = pdk
        self.canvas_size = canvas_size
        # Coordinate ids are offset by N_RESERVED so they never collide
        # with control tokens. With canvas_size pixels per axis we have
        # canvas_size + 1 distinct vertex coordinates (0..canvas_size).
        self.coord_offset = N_RESERVED
        self.vocab_size = N_RESERVED + (canvas_size + 1)

    @classmethod
    def from_pdk(cls, pdk_name: str, canvas_size: int = 256) -> LayoutTokenizer:
        return cls(get_pdk(pdk_name), canvas_size=canvas_size)

    def _coord_id(self, coord: int) -> int:
        if not (0 <= coord <= self.canvas_size):
            raise ValueError(f"coordinate {coord} out of range [0, {self.canvas_size}]")
        return self.coord_offset + coord

    def _coord_value(self, token_id: int) -> int:
        return int(token_id) - self.coord_offset

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
            pdk_name=self.pdk.name,
        )

    def decode(self, ids: torch.Tensor | TokenizedLayout) -> torch.Tensor:
        """Token sequence → mask (canvas_size, canvas_size) float32 in [0, 1]."""
        if isinstance(ids, TokenizedLayout):
            ids = ids.ids
        seq = [int(t) for t in ids.tolist()]
        mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        i = 0
        if seq and seq[0] == TOK_BOS:
            i += 1
        while i < len(seq):
            t = seq[i]
            if t == TOK_EOS:
                break
            if t != TOK_POLYGON:
                # Tolerate stray tokens before the next <polygon>.
                i += 1
                continue
            i += 1
            verts: list[tuple[int, int]] = []
            while i < len(seq) and seq[i] != TOK_CLOSE:
                if i + 1 >= len(seq):
                    break
                vy = self._coord_value(seq[i])
                vx = self._coord_value(seq[i + 1])
                verts.append((vy, vx))
                i += 2
            # Skip <close>.
            if i < len(seq) and seq[i] == TOK_CLOSE:
                i += 1
            if len(verts) == 4:
                # Manhattan rectangle: bbox-fill it.
                ys = [v[0] for v in verts]
                xs = [v[1] for v in verts]
                y0, y1 = min(ys), max(ys)
                x0, x1 = min(xs), max(xs)
                mask[y0:y1, x0:x1] = 1

        return torch.from_numpy(mask).to(torch.float32)


def _rectangle_decomposition(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
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
    def _row_runs(row: np.ndarray) -> tuple[tuple[int, int], ...]:
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

    y = 0
    while y < h:
        runs = _row_runs(binary[y])
        if not runs:
            y += 1
            continue
        # Find the largest y2 such that rows y..y2-1 all share these runs.
        y2 = y + 1
        while y2 < h and _row_runs(binary[y2]) == runs:
            y2 += 1
        for x0, x1 in runs:
            rects.append((y, x0, y2, x1))
        y = y2
    return rects
