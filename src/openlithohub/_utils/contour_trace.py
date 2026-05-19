"""Boundary contour tracing for binary masks (Moore neighborhood)."""

from __future__ import annotations

import numpy as np


def trace_contour(binary: np.ndarray) -> list[np.ndarray]:
    """Trace ordered boundary points from a binary mask using Moore neighborhood."""
    h, w = binary.shape
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    visited_edges: set[tuple[int, int]] = set()
    contours: list[np.ndarray] = []

    directions = [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ]

    for start_y in range(1, h + 1):
        for start_x in range(1, w + 1):
            if padded[start_y, start_x] == 0:
                continue
            if padded[start_y, start_x - 1] != 0:
                continue
            if (start_y, start_x) in visited_edges:
                continue

            points: list[tuple[int, int]] = []
            cy, cx = start_y, start_x
            entry_dir = 0

            # The boundary of an arbitrary connected region on an H*W grid is
            # O(H*W), not O(H+W) — a serpentine shape can have a perimeter that
            # threads through every pixel. Bound by 2*h*w (each pixel can be
            # visited from at most two directions in Moore tracing) so we never
            # truncate a real boundary.
            max_steps = 2 * h * w + 8
            for _ in range(max_steps):
                points.append((cy - 1, cx - 1))
                visited_edges.add((cy, cx))

                found = False
                search_start = (entry_dir + 5) % 8
                for k in range(8):
                    d = (search_start + k) % 8
                    ny = cy + directions[d][0]
                    nx = cx + directions[d][1]
                    if 0 <= ny < h + 2 and 0 <= nx < w + 2 and padded[ny, nx] > 0:
                        cy, cx = ny, nx
                        entry_dir = d
                        found = True
                        break

                if not found:
                    break
                if cy == start_y and cx == start_x:
                    break

            if len(points) >= 4:
                contours.append(np.array(points, dtype=np.float64))

    return contours
