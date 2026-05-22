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
    """
    if not curves:
        raise ValueError("Cannot export an empty curve list to OASIS.")
    if min_area_nm2 < 0.0:
        raise ValueError(f"min_area_nm2 must be >= 0, got {min_area_nm2}")

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

    n_filtered = 0
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
        points = [
            db.Point(int(round(float(x) / layout.dbu)), int(round(float(y) / layout.dbu)))
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

    layout.write(str(output))
