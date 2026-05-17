"""Tests for workflow layer."""

import pytest

from openlithohub.workflow.contour.curvilinear import export_oasis_mbw, fit_bspline
from openlithohub.workflow.contour.manhattan import extract_manhattan_contour
from openlithohub.workflow.export import export_oasis
from openlithohub.workflow.parsing import parse_layout
from openlithohub.workflow.tiling import tile_layout


def test_parse_layout_not_implemented():
    with pytest.raises(NotImplementedError, match="parsing"):
        parse_layout("/fake/path.oas")


def test_tile_layout_not_implemented(sample_design):
    with pytest.raises(NotImplementedError, match="tiling"):
        tile_layout(sample_design)


def test_manhattan_contour_not_implemented(sample_mask):
    with pytest.raises(NotImplementedError, match="Manhattan"):
        extract_manhattan_contour(sample_mask)


def test_bspline_not_implemented(sample_mask):
    with pytest.raises(NotImplementedError, match="B-spline"):
        fit_bspline(sample_mask)


def test_oasis_mbw_export_not_implemented():
    with pytest.raises(NotImplementedError, match="OASIS.MBW"):
        export_oasis_mbw([], "/fake/output.oas")


def test_export_oasis_not_implemented(sample_mask):
    with pytest.raises(NotImplementedError, match="OASIS export"):
        export_oasis(sample_mask, "/fake/output.oas")
