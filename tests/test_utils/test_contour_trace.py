"""Regression tests for `_utils.contour_trace.trace_contour`."""

from __future__ import annotations

import numpy as np

from openlithohub._utils.contour_trace import trace_contour


def test_single_pixel_returns_a_loop():
    """A single foreground pixel used to silently disappear because the
    tracer required len(points) >= 4 before retaining a contour. Curvilinear
    MRC reporting and small-feature exports lost SRAFs as a result; the
    minimum length gate is now removed and gating is the caller's job.
    """
    arr = np.zeros((5, 5), dtype=np.int8)
    arr[2, 2] = 1
    loops = trace_contour(arr)
    assert len(loops) == 1
    assert len(loops[0]) >= 1


def test_three_pixel_l_shape_is_kept():
    """Three-pixel L-tromino — earlier `len(points) >= 4` gate dropped
    this case. Now retained so curvilinear MRC can flag it."""
    arr = np.zeros((5, 5), dtype=np.int8)
    arr[2, 2] = 1
    arr[2, 3] = 1
    arr[3, 2] = 1
    loops = trace_contour(arr)
    assert len(loops) == 1
    # Tracer collects start + corner pixels of the boundary; <4 was
    # the silent-drop threshold under the old code.
    assert len(loops[0]) >= 1


def test_empty_mask_returns_no_loops():
    arr = np.zeros((4, 4), dtype=np.int8)
    assert trace_contour(arr) == []
