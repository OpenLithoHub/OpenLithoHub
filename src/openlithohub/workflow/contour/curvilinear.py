"""Curvilinear contour extraction and B-spline fitting for OASIS.MBW export."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from openlithohub._utils.tensor_ops import ensure_2d


@dataclass
class BSplineCurve:
    """Representation of a fitted B-spline curve."""

    control_points: torch.Tensor
    knots: torch.Tensor
    degree: int = 3


def _trace_contour(binary: np.ndarray) -> list[np.ndarray]:
    """Trace ordered boundary points from a binary mask using Moore neighborhood."""
    h, w = binary.shape
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    visited_edges = set()
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

            max_steps = 4 * (h + w)
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


def fit_bspline(
    contour_pixels: torch.Tensor,
    tolerance_nm: float = 0.5,
    pixel_size_nm: float = 1.0,
) -> list[BSplineCurve]:
    """Fit B-spline curves to pixel-level contour data.

    If input is a 2D binary mask, extracts boundary contours first.
    If input is an (N, 2) tensor, treats it as a single ordered point loop.
    """
    try:
        from scipy.interpolate import splprep
    except ImportError:
        raise ImportError(
            "scipy is required for B-spline fitting. "
            "Install with: pip install openlithohub[workflow]"
        ) from None

    if contour_pixels.ndim == 2 and contour_pixels.shape[1] == 2:
        loops = [contour_pixels.detach().cpu().numpy()]
    else:
        m = ensure_2d(contour_pixels)
        arr = (m > 0.5).detach().cpu().numpy().astype(np.int8)
        loops = _trace_contour(arr)

    smoothing = tolerance_nm / max(pixel_size_nm, 1e-6)
    curves: list[BSplineCurve] = []

    for loop in loops:
        if len(loop) < 5:
            continue

        loop_scaled = loop * pixel_size_nm

        try:
            tck, _ = splprep(
                [loop_scaled[:, 1], loop_scaled[:, 0]],
                s=smoothing * len(loop),
                per=True,
                k=3,
            )
        except (ValueError, TypeError):
            continue

        ctrl_x = np.array(tck[1][0], dtype=np.float32)
        ctrl_y = np.array(tck[1][1], dtype=np.float32)
        control_points = torch.tensor(np.stack([ctrl_x, ctrl_y], axis=1), dtype=torch.float32)
        knots = torch.tensor(tck[0], dtype=torch.float32)

        curves.append(BSplineCurve(control_points=control_points, knots=knots, degree=3))

    return curves


def export_oasis_mbw(
    curves: list[BSplineCurve],
    output_path: str,
    *,
    format_version: str = "2.1",
    samples_per_curve: int = 64,
) -> None:
    """Serialize B-spline curves to OASIS.MBW format for multi-beam writers.

    MVP implementation: samples curves to high-resolution polygons and writes
    a simplified OASIS-compatible binary file with polygon records.
    Native curve primitives per SEMI P44 are planned for a future release.
    """
    try:
        from scipy.interpolate import splev
    except ImportError:
        raise ImportError(
            "scipy is required for OASIS.MBW export. "
            "Install with: pip install openlithohub[workflow]"
        ) from None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sampled_polygons: list[list[tuple[float, float]]] = []
    for curve in curves:
        ctrl = curve.control_points.numpy()
        knot = curve.knots.numpy()
        tck = (knot, [ctrl[:, 0], ctrl[:, 1]], curve.degree)
        u_eval = np.linspace(0.0, 1.0, samples_per_curve, endpoint=False)
        xs, ys = splev(u_eval, tck)
        polygon = [(float(x), float(y)) for x, y in zip(xs, ys, strict=False)]
        sampled_polygons.append(polygon)

    _write_oasis_binary(sampled_polygons, output, format_version)


def _write_oasis_binary(
    polygons: list[list[tuple[float, float]]],
    output_path: Path,
    version: str,
) -> None:
    """Write a simplified OASIS binary file with polygon records."""
    with open(output_path, "wb") as f:
        f.write(b"%SEMI-OASIS\r\n")
        f.write(struct.pack("<B", 1))
        f.write(f"MBW {version}\0".encode("ascii"))

        f.write(struct.pack("<B", 14))
        f.write(b"TOP\0")
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))

        for polygon in polygons:
            f.write(struct.pack("<B", 21))
            f.write(struct.pack("<H", len(polygon)))
            for x, y in polygon:
                f.write(struct.pack("<ii", int(x * 1000), int(y * 1000)))

        f.write(struct.pack("<B", 2))
