"""Tests for compliance checks."""

import pytest

from openlithohub.benchmark.compliance.drc import check_drc
from openlithohub.benchmark.compliance.mrc import check_mrc


def test_mrc_not_implemented(sample_mask):
    with pytest.raises(NotImplementedError, match="MRC"):
        check_mrc(sample_mask)


def test_drc_not_implemented(sample_mask):
    with pytest.raises(NotImplementedError, match="DRC"):
        check_drc(sample_mask)
