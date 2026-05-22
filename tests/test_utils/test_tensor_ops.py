"""Tests for openlithohub._utils.tensor_ops."""

import pytest
import torch

from openlithohub._utils.tensor_ops import ensure_2d


class TestEnsure2D:
    def test_2d_noop(self) -> None:
        t = torch.rand(16, 16)
        result = ensure_2d(t)
        assert result.shape == (16, 16)

    def test_3d_squeeze(self) -> None:
        t = torch.rand(1, 16, 16)
        result = ensure_2d(t)
        assert result.shape == (16, 16)

    def test_4d_squeeze(self) -> None:
        t = torch.rand(1, 1, 32, 32)
        result = ensure_2d(t)
        assert result.shape == (32, 32)

    def test_preserves_values(self) -> None:
        t = torch.rand(1, 1, 8, 8)
        result = ensure_2d(t)
        assert torch.equal(result, t.squeeze())

    def test_rejects_batch_4d(self) -> None:
        with pytest.raises(ValueError, match="Iterate over the batch"):
            ensure_2d(torch.rand(2, 1, 8, 8))

    def test_rejects_multichannel_4d(self) -> None:
        with pytest.raises(ValueError, match="Iterate over the batch"):
            ensure_2d(torch.rand(1, 3, 8, 8))

    def test_rejects_batch_3d(self) -> None:
        with pytest.raises(ValueError, match="Iterate over the batch"):
            ensure_2d(torch.rand(2, 8, 8))
