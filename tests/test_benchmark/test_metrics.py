"""Tests for benchmark metrics."""

import pytest

from openlithohub.benchmark.metrics.epe import compute_epe
from openlithohub.benchmark.metrics.pvband import compute_pvband
from openlithohub.benchmark.metrics.shot_count import estimate_shot_count
from openlithohub.benchmark.metrics.stochastic import compute_stochastic_robustness


def test_epe_not_implemented(sample_design, sample_mask):
    with pytest.raises(NotImplementedError, match="EPE"):
        compute_epe(sample_design, sample_mask)


def test_pvband_not_implemented(sample_mask):
    with pytest.raises(NotImplementedError, match="PV Band"):
        compute_pvband(sample_mask)


def test_shot_count_not_implemented(sample_mask):
    with pytest.raises(NotImplementedError, match="Shot count"):
        estimate_shot_count(sample_mask)


def test_stochastic_not_implemented(sample_mask):
    with pytest.raises(NotImplementedError, match="Stochastic"):
        compute_stochastic_robustness(sample_mask)
