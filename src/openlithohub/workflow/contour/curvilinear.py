"""Curvilinear contour extraction and B-spline fitting for OASIS export.

Curvilinear masks (post-ILT, EUV, MBMW writers) are emitted as high-resolution
sampled polygons on a designated layer of a real OASIS file. Native SEMI P44
curve primitives are not yet emitted — `klayout.db` does not surface them in
its public Python API at the time of writing — but the file produced here is
a valid OASIS file that any vendor tool can read.
"""

from __future__ import annotations

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
    samples_per_curve: int = 64,
    pixel_size_nm: float = 1.0,
    layer: int = 1,
    datatype: int = 0,
    cell_name: str = "TOP",
) -> None:
    """Serialize B-spline curves to an OASIS file via klayout.db.

    Curves are sampled to high-resolution polygons (``samples_per_curve``
    vertices per loop) and inserted into a single top cell on the requested
    ``(layer, datatype)``. The output is a real SEMI P39 OASIS file readable
    by KLayout, Calibre, and other industry tools — not a custom binary
    blob.

    Native SEMI P44 multi-beam curve primitives are not yet emitted; the
    polygon approximation is the standard interim representation that all
    multi-beam writer flows accept.
    """
    if not curves:
        raise ValueError("Cannot export an empty curve list to OASIS.")

    try:
        from scipy.interpolate import splev
    except ImportError:
        raise ImportError(
            "scipy is required for OASIS export. Install with: pip install openlithohub[workflow]"
        ) from None

    try:
        import klayout.db as db
    except ImportError:
        raise ImportError(
            "klayout is required for OASIS export. Install with: pip install openlithohub[workflow]"
        ) from None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    layout = db.Layout()
    layout.dbu = pixel_size_nm / 1000.0
    top = layout.create_cell(cell_name)
    layer_idx = layout.layer(layer, datatype)

    for curve in curves:
        ctrl = curve.control_points.numpy()
        knot = curve.knots.numpy()
        tck = (knot, [ctrl[:, 0], ctrl[:, 1]], curve.degree)
        u_eval = np.linspace(0.0, 1.0, samples_per_curve, endpoint=False)
        xs, ys = splev(u_eval, tck)
        points = [
            db.Point(int(round(float(x) / layout.dbu)), int(round(float(y) / layout.dbu)))
            for x, y in zip(xs, ys, strict=False)
        ]
        if len(points) >= 3:
            top.shapes(layer_idx).insert(db.Polygon(points))

    layout.write(str(output))
