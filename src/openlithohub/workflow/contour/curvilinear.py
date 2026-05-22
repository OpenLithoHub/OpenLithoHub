"""Curvilinear contour extraction and B-spline fitting for OASIS export.

Curvilinear masks (post-ILT, EUV, MBMW writers) are emitted as high-resolution
sampled polygons on a designated layer of a real OASIS file. Native SEMI P44
curve primitives are not yet emitted — `klayout.db` does not surface them in
its public Python API at the time of writing — but the file produced here is
a valid OASIS file that any vendor tool can read.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from openlithohub._utils.contour_trace import trace_contour
from openlithohub._utils.tensor_ops import ensure_2d

logger = logging.getLogger(__name__)


@dataclass
class BSplineCurve:
    """Representation of a fitted B-spline curve."""

    control_points: torch.Tensor
    knots: torch.Tensor
    degree: int = 3


def fit_bspline(
    contour_pixels: torch.Tensor,
    tolerance_nm: float = 0.5,
    pixel_size_nm: float = 1.0,
    *,
    warn_on_skip: bool = True,
) -> list[BSplineCurve]:
    """Fit B-spline curves to pixel-level contour data.

    If input is a 2D binary mask, extracts boundary contours first.
    If input is an (N, 2) tensor, treats it as a single ordered point loop.

    Loops with fewer than 5 points (the minimum for a cubic periodic
    spline) and loops where ``splprep`` fails to converge are skipped.
    Both used to be silently dropped — small features (SRAFs, sharp
    points) would disappear from the OASIS export with no signal. Now a
    ``UserWarning`` is emitted per skipped loop unless
    ``warn_on_skip=False``; callers that intentionally feed mixed-size
    geometry can opt out.

    The smoothing factor passed to ``splprep`` is ``tolerance_nm**2 *
    n_points`` (matches scipy's "sum of squared residuals" semantics) so
    long contours are not over-smoothed relative to short ones.
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
        loops = trace_contour(arr)

    curves: list[BSplineCurve] = []

    for idx, loop in enumerate(loops):
        n = len(loop)
        if n < 5:
            if warn_on_skip:
                warnings.warn(
                    f"fit_bspline: loop {idx} has {n} points (< 5 needed for cubic "
                    f"periodic spline); skipping. Small features may be lost.",
                    UserWarning,
                    stacklevel=2,
                )
            continue

        # Drop a duplicated closing vertex: ``splprep(per=True)`` builds
        # the periodic boundary itself; passing the closure point twice
        # makes scipy see a zero-length segment and return a degenerate
        # spline with seam oscillation.
        if np.allclose(loop[0], loop[-1]):
            loop = loop[:-1]
            n = len(loop)
            if n < 5:
                if warn_on_skip:
                    warnings.warn(
                        f"fit_bspline: loop {idx} reduced to {n} points after "
                        f"closure dedup; skipping.",
                        UserWarning,
                        stacklevel=2,
                    )
                continue

        # Drop consecutive duplicate points anywhere in the loop. Moore
        # neighborhood tracing in ``trace_contour`` can revisit the same
        # boundary edge, producing zero-length segments that make
        # ``splprep`` raise "Invalid inputs" or singular-matrix errors.
        diffs = np.diff(loop, axis=0, append=loop[:1])
        keep = np.any(np.abs(diffs) > 1e-9, axis=1)
        if not np.all(keep):
            loop = loop[keep]
            n = len(loop)
            if n < 5:
                if warn_on_skip:
                    warnings.warn(
                        f"fit_bspline: loop {idx} reduced to {n} points after "
                        f"consecutive-duplicate dedup; skipping.",
                        UserWarning,
                        stacklevel=2,
                    )
                continue

        loop_scaled = loop * pixel_size_nm
        smoothing = (tolerance_nm**2) * n

        try:
            tck, _ = splprep(
                [loop_scaled[:, 1], loop_scaled[:, 0]],
                s=smoothing,
                per=True,
                k=3,
            )
        except (ValueError, TypeError) as e:
            if warn_on_skip:
                warnings.warn(
                    f"fit_bspline: splprep failed on loop {idx} (n={n}, "
                    f"tolerance_nm={tolerance_nm}): {e}; skipping.",
                    UserWarning,
                    stacklevel=2,
                )
            continue

        ctrl_x = np.array(tck[1][0], dtype=np.float32)
        ctrl_y = np.array(tck[1][1], dtype=np.float32)
        control_points = torch.tensor(np.stack([ctrl_x, ctrl_y], axis=1), dtype=torch.float32)
        knots = torch.tensor(tck[0], dtype=torch.float32)

        curves.append(BSplineCurve(control_points=control_points, knots=knots, degree=3))

    return curves


def _rdp_simplify(xs: np.ndarray, ys: np.ndarray, tolerance: float) -> np.ndarray:
    """Iterative Ramer-Douglas-Peucker on a closed polygon.

    Returns a boolean keep-mask the same length as ``xs``/``ys``. A point
    is kept when the perpendicular distance from it to the chord between
    its surviving neighbours exceeds ``tolerance``. ``tolerance <= 0``
    short-circuits to keep-all so callers can opt out without paying any
    geometry cost.

    Closed polygons need both endpoints anchored (the loop has no natural
    endpoints to seed the recursion); we anchor index 0 and the
    farthest-from-0 point, then DP each half. This matches the standard
    closed-curve RDP variant used by mask-write toolchains.
    """
    n = len(xs)
    if tolerance <= 0.0 or n < 4:
        return np.ones(n, dtype=bool)

    # Seed the second anchor at the farthest point from index 0 so each
    # half of the loop is a well-defined open polyline.
    dx0 = xs - xs[0]
    dy0 = ys - ys[0]
    far_idx = int(np.argmax(dx0 * dx0 + dy0 * dy0))
    if far_idx == 0:
        return np.ones(n, dtype=bool)

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[far_idx] = True
    tol_sq = tolerance * tolerance

    stack: list[tuple[int, int]] = [(0, far_idx), (far_idx, n)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        end = hi % n
        x0, y0 = xs[lo], ys[lo]
        xe, ye = xs[end], ys[end]
        ex, ey = xe - x0, ye - y0
        seg_len_sq = ex * ex + ey * ey
        max_d_sq = -1.0
        max_i = -1
        for i in range(lo + 1, hi):
            px, py = xs[i] - x0, ys[i] - y0
            if seg_len_sq == 0.0:
                d_sq = px * px + py * py
            else:
                cross = px * ey - py * ex
                d_sq = (cross * cross) / seg_len_sq
            if d_sq > max_d_sq:
                max_d_sq = d_sq
                max_i = i
        if max_d_sq > tol_sq and max_i > 0:
            keep[max_i] = True
            stack.append((lo, max_i))
            stack.append((max_i, hi))

    return keep


def export_oasis_mbw(
    curves: list[BSplineCurve],
    output_path: str,
    *,
    samples_per_curve: int = 64,
    pixel_size_nm: float = 1.0,
    layer: int = 1,
    datatype: int = 0,
    cell_name: str = "TOP",
    min_area_nm2: float = 0.0,
    vertex_tolerance_nm: float = 0.0,
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

    ``min_area_nm2`` is an opt-in filter for sub-resolution islands: any
    sampled polygon with absolute area below this threshold is dropped
    before insertion. Default ``0.0`` keeps every shape so academic /
    Hackathon evaluation stays bit-exact. A positive value is intended for
    fab-ready exports where MRC would otherwise reject the smallest SRAFs
    a curvilinear ILT can produce; the count of dropped shapes is logged
    at INFO level so the filter is auditable.

    ``vertex_tolerance_nm`` is an opt-in Ramer-Douglas-Peucker simplification
    on each sampled polygon: any vertex within this perpendicular distance
    of the chord between its surviving neighbours is dropped. Default ``0.0``
    keeps every sampled vertex (bit-exact academic behaviour). Positive
    values cut OASIS file size dramatically on smooth ILT contours — a
    multi-beam mask writer (MBMW) consumes shots and bytes, so 0.5 nm
    typically halves vertex count without measurable wafer-image change.
    """
    if not curves:
        raise ValueError("Cannot export an empty curve list to OASIS.")
    if min_area_nm2 < 0.0:
        raise ValueError(f"min_area_nm2 must be >= 0, got {min_area_nm2}")
    if vertex_tolerance_nm < 0.0:
        raise ValueError(f"vertex_tolerance_nm must be >= 0, got {vertex_tolerance_nm}")

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

    # KLayout DB units: layout.dbu is in microns. A DB integer coord i represents
    # i * dbu microns = i * dbu * 1000 nm. xs/ys here are already in nm
    # (fit_bspline scaled control points by pixel_size_nm), so divide by
    # (dbu * 1000) — equivalently by pixel_size_nm.
    nm_per_dbu = layout.dbu * 1000.0

    n_filtered = 0
    n_vertices_before = 0
    n_vertices_after = 0
    for curve in curves:
        ctrl = curve.control_points.numpy()
        knot = curve.knots.numpy()
        tck = (knot, [ctrl[:, 0], ctrl[:, 1]], curve.degree)
        u_eval = np.linspace(0.0, 1.0, samples_per_curve, endpoint=False)
        xs, ys = splev(u_eval, tck)
        xs_arr = np.asarray(xs, dtype=np.float64)
        ys_arr = np.asarray(ys, dtype=np.float64)
        if min_area_nm2 > 0.0 and len(xs_arr) >= 3:
            # Shoelace on the sampled polygon (xs/ys are in nm because
            # ``fit_bspline`` scales control points by pixel_size_nm).
            area_nm2 = 0.5 * abs(
                float(np.dot(xs_arr, np.roll(ys_arr, -1)) - np.dot(ys_arr, np.roll(xs_arr, -1)))
            )
            if area_nm2 < min_area_nm2:
                n_filtered += 1
                continue
        if vertex_tolerance_nm > 0.0 and len(xs_arr) >= 4:
            n_vertices_before += len(xs_arr)
            keep_mask = _rdp_simplify(xs_arr, ys_arr, vertex_tolerance_nm)
            xs_arr = xs_arr[keep_mask]
            ys_arr = ys_arr[keep_mask]
            n_vertices_after += len(xs_arr)
        points = [
            db.Point(int(round(float(x) / nm_per_dbu)), int(round(float(y) / nm_per_dbu)))
            for x, y in zip(xs_arr, ys_arr, strict=False)
        ]
        if len(points) >= 3:
            top.shapes(layer_idx).insert(db.Polygon(points))

    if n_filtered > 0:
        logger.info(
            "export_oasis_mbw: filtered %d shape(s) below min_area_nm2=%g nm^2",
            n_filtered,
            min_area_nm2,
        )
    if vertex_tolerance_nm > 0.0 and n_vertices_before > 0:
        logger.info(
            "export_oasis_mbw: RDP simplified %d → %d vertices (%.1f%% reduction) "
            "at tolerance_nm=%g",
            n_vertices_before,
            n_vertices_after,
            100.0 * (1.0 - n_vertices_after / n_vertices_before),
            vertex_tolerance_nm,
        )

    layout.write(str(output))
