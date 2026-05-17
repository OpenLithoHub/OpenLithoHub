"""Shared test fixtures for OpenLithoHub."""

import pytest
import torch


@pytest.fixture
def sample_design() -> torch.Tensor:
    """A simple 64x64 binary design pattern for testing."""
    t = torch.zeros(64, 64)
    t[16:48, 16:48] = 1.0  # Square feature
    return t


@pytest.fixture
def sample_mask() -> torch.Tensor:
    """A simple 64x64 binary mask (slightly larger than design for OPC bias)."""
    t = torch.zeros(64, 64)
    t[14:50, 14:50] = 1.0  # Biased square
    return t
