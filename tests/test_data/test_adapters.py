"""Tests for data layer interfaces."""

import pytest

from openlithohub.data import LithoBenchDataset, LithoSample, LithoSimDataset


def test_litho_sample_creation(sample_design, sample_mask):
    sample = LithoSample(design=sample_design, mask=sample_mask, metadata={"node": "45nm"})
    assert sample.design.shape == (64, 64)
    assert sample.mask.shape == (64, 64)
    assert sample.metadata["node"] == "45nm"


def test_lithobench_not_implemented():
    ds = LithoBenchDataset(root="/tmp/fake")
    with pytest.raises(NotImplementedError):
        len(ds)
    with pytest.raises(NotImplementedError):
        ds[0]


def test_lithosim_not_implemented():
    ds = LithoSimDataset(split="test")
    with pytest.raises(NotImplementedError):
        len(ds)
    with pytest.raises(NotImplementedError):
        ds[0]
