"""Manhattan (staircase) contour extraction for traditional VSB writers."""

from __future__ import annotations

import numpy as np
import torch

from openlithohub._utils.tensor_ops import ensure_2d


def extract_manhattan_contour(
    mask: torch.Tensor,
    pixel_size_nm: float = 1.0,
) -> list[list[tuple[float, float]]]:
    """Extract Manhattan (rectilinear) polygon contours from a binary mask.

    Traces pixel-edge boundaries between foreground and background regions,
    producing axis-aligned polygons suitable for VSB mask writers.

    The algorithm:
    1. Find horizontal and vertical boundary edges between 0/1 pixels
    2. Build an adjacency graph of edges sharing vertices
    3. Trace closed loops through the edge graph

    Vertices are at pixel corners (not pixel centers), scaled by pixel_size_nm.

    Args:
        mask: Binary mask tensor (H, W).
        pixel_size_nm: Physical pixel size for coordinate scaling.

    Returns:
        List of polygons, each as a list of (x_nm, y_nm) vertices in order.
        Outer boundaries are clockwise, holes are counter-clockwise.
    """
    m = ensure_2d(mask)
    arr = (m > 0.5).detach().cpu().numpy().astype(np.int8)
    h, w = arr.shape

    # Pad with zeros to ensure boundaries are closed at image edges
    padded = np.pad(arr, 1, mode="constant", constant_values=0)

    # Find horizontal edges: between row i and row i+1
    # An edge exists where padded[i, j] != padded[i+1, j]
    h_edges: set[tuple[int, int, int, int]] = set()
    v_edges: set[tuple[int, int, int, int]] = set()

    ph, pw = padded.shape

    # Horizontal edges: edge from (j, i) to (j+1, i) in vertex space
    # A horizontal edge at grid row i exists between pixel rows i-1 and i
    for i in range(ph - 1):
        for j in range(pw - 1):
            if padded[i, j] != padded[i + 1, j]:
                h_edges.add((j, i + 1, j + 1, i + 1))  # (x1, y1, x2, y2) left to right

    # Vertical edges: edge from (j, i) to (j, i+1) in vertex space
    for i in range(ph - 1):
        for j in range(pw - 1):
            if padded[i, j] != padded[i, j + 1]:
                v_edges.add((j + 1, i, j + 1, i + 1))  # top to bottom

    all_edges = h_edges | v_edges
    if not all_edges:
        return []

    # Build adjacency: vertex -> list of connected edges
    vertex_to_edges: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for edge in all_edges:
        x1, y1, x2, y2 = edge
        vertex_to_edges.setdefault((x1, y1), []).append(edge)
        vertex_to_edges.setdefault((x2, y2), []).append(edge)

    # Trace closed loops
    remaining = set(all_edges)
    polygons: list[list[tuple[float, float]]] = []

    while remaining:
        start_edge = next(iter(remaining))
        polygon_vertices: list[tuple[int, int]] = []

        x1, y1, x2, y2 = start_edge
        polygon_vertices.append((x1, y1))
        current_vertex = (x2, y2)
        in_dir = (x2 - x1, y2 - y1)
        remaining.discard(start_edge)

        while current_vertex != (x1, y1):
            polygon_vertices.append(current_vertex)
            candidates = vertex_to_edges.get(current_vertex, [])
            next_edge = _pick_next_edge(current_vertex, in_dir, candidates, remaining)

            if next_edge is None:
                break

            remaining.discard(next_edge)
            ex1, ey1, ex2, ey2 = next_edge
            next_vertex = (ex2, ey2) if (ex1, ey1) == current_vertex else (ex1, ey1)
            in_dir = (next_vertex[0] - current_vertex[0], next_vertex[1] - current_vertex[1])
            current_vertex = next_vertex

        # Convert to physical coordinates (subtract 1 for padding offset)
        scaled: list[tuple[float, float]] = []
        for vx, vy in polygon_vertices:
            scaled.append(((vx - 1) * pixel_size_nm, (vy - 1) * pixel_size_nm))

        if len(scaled) >= 3:
            polygons.append(_simplify_collinear(scaled))

    return polygons


def _pick_next_edge(
    vertex: tuple[int, int],
    in_dir: tuple[int, int],
    candidates: list[tuple[int, int, int, int]],
    remaining: set[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Pick the next edge at a junction so the trace stays on one polygon.

    At an X- or T-junction (two foreground regions meeting at a single corner)
    multiple remaining edges are incident to ``vertex``. Picking arbitrarily
    can jump onto a different polygon and leave the original loop unclosed.

    To preserve the invariant that foreground stays on the right of the
    traversal direction, prefer turn directions in this order: right turn
    (clockwise) > straight > left turn > U-turn. Edges already consumed
    (not in ``remaining``) are ignored.
    """
    best: tuple[int, int, int, int] | None = None
    best_rank = 5
    for edge in candidates:
        if edge not in remaining:
            continue
        ex1, ey1, ex2, ey2 = edge
        out_dir = (ex2 - ex1, ey2 - ey1) if (ex1, ey1) == vertex else (ex1 - ex2, ey1 - ey2)

        cross = in_dir[0] * out_dir[1] - in_dir[1] * out_dir[0]
        dot = in_dir[0] * out_dir[0] + in_dir[1] * out_dir[1]
        if cross > 0:
            rank = 0  # right turn (image y grows downward, so cross > 0 is CW)
        elif cross == 0 and dot > 0:
            rank = 1  # straight
        elif cross < 0:
            rank = 2  # left turn
        else:
            rank = 3  # U-turn

        if rank < best_rank:
            best = edge
            best_rank = rank
    return best


def _simplify_collinear(vertices: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove vertices that lie on a straight line between their neighbors."""
    if len(vertices) <= 3:
        return vertices

    simplified: list[tuple[float, float]] = []
    n = len(vertices)

    for i in range(n):
        prev = vertices[(i - 1) % n]
        curr = vertices[i]
        nxt = vertices[(i + 1) % n]

        dx1 = curr[0] - prev[0]
        dy1 = curr[1] - prev[1]
        dx2 = nxt[0] - curr[0]
        dy2 = nxt[1] - curr[1]

        # Keep vertex if direction changes
        if (dx1, dy1) != (dx2, dy2):
            simplified.append(curr)

    return simplified if len(simplified) >= 3 else vertices
