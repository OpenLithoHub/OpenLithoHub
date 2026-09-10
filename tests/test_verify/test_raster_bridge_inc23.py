import math

import numpy as np

from openlithohub.verify.raster_bridge import compare_binary_rasters


def test_equal_masks_have_zero_bridge_radius():
    a = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    b = compare_binary_rasters(a, a, max_radius=2)
    assert b.exact_equal
    assert b.mutual_chebyshev_radius_px == 0
    assert b.euclidean_center_hausdorff_upper_nm(pixel_nm=8.0) == 0.0


def test_one_pixel_dense_expansion_has_radius_one():
    ref = np.zeros((5, 5), dtype=np.uint8)
    ref[2, 2] = 1
    cand = np.zeros((5, 5), dtype=np.uint8)
    cand[1:4, 1:4] = 1
    b = compare_binary_rasters(ref, cand, max_radius=2)
    assert b.false_positive_pixels == 8
    assert b.false_negative_pixels == 0
    assert b.candidate_in_reference_dilation_px == 1
    assert b.reference_in_candidate_dilation_px == 0
    assert b.mutual_chebyshev_radius_px == 1
    assert abs(b.euclidean_center_hausdorff_upper_nm(pixel_nm=8.0) - 8.0 * math.sqrt(2.0)) < 1e-12


def test_disconnected_far_candidate_reports_no_small_bridge():
    ref = np.zeros((9, 9), dtype=np.uint8)
    ref[1, 1] = 1
    cand = np.zeros((9, 9), dtype=np.uint8)
    cand[7, 7] = 1
    b = compare_binary_rasters(ref, cand, max_radius=2)
    assert b.candidate_in_reference_dilation_px is None
    assert b.reference_in_candidate_dilation_px is None
    assert b.mutual_chebyshev_radius_px is None
