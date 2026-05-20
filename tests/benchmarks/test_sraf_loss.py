"""Tests for openlithohub.benchmark.metrics.sraf."""

from __future__ import annotations

import pytest
import torch

from openlithohub.benchmark.metrics import sraf_print_penalty


class TestSrafPrintPenalty:
    def test_zero_loss_when_below_budget(self) -> None:
        aerial = torch.full((16, 16), 0.10)
        sraf = torch.zeros((16, 16))
        sraf[4:8, 4:8] = 1.0
        loss = sraf_print_penalty(aerial, sraf, print_threshold=0.30, margin=0.05)
        assert loss.item() == pytest.approx(0.0)

    def test_positive_loss_when_above_budget(self) -> None:
        aerial = torch.full((16, 16), 0.40)
        sraf = torch.zeros((16, 16))
        sraf[4:8, 4:8] = 1.0
        loss = sraf_print_penalty(aerial, sraf, print_threshold=0.30, margin=0.05)
        assert loss.item() > 0.0

    def test_loss_grows_monotonically_with_violation(self) -> None:
        sraf = torch.zeros((8, 8))
        sraf[2:6, 2:6] = 1.0
        loss_low = sraf_print_penalty(torch.full((8, 8), 0.30), sraf)
        loss_high = sraf_print_penalty(torch.full((8, 8), 0.50), sraf)
        assert loss_high.item() > loss_low.item()

    def test_only_sraf_pixels_contribute(self) -> None:
        aerial = torch.zeros((8, 8))
        aerial[0:2, 0:2] = 0.9  # bright but outside SRAF region
        sraf = torch.zeros((8, 8))
        sraf[6:8, 6:8] = 1.0  # SRAF region is dark
        loss = sraf_print_penalty(aerial, sraf, print_threshold=0.30, margin=0.05)
        assert loss.item() == pytest.approx(0.0)

    def test_zero_when_no_sraf_pixels(self) -> None:
        aerial = torch.full((8, 8), 0.99)
        sraf = torch.zeros((8, 8))
        loss = sraf_print_penalty(aerial, sraf)
        assert loss.item() == pytest.approx(0.0)

    def test_gradient_flows_to_aerial(self) -> None:
        aerial = torch.full((8, 8), 0.40, requires_grad=True)
        sraf = torch.zeros((8, 8))
        sraf[2:6, 2:6] = 1.0
        loss = sraf_print_penalty(aerial, sraf, print_threshold=0.30, margin=0.05)
        loss.backward()
        assert aerial.grad is not None
        assert aerial.grad[2:6, 2:6].abs().sum().item() > 0.0
        # Pixels outside the SRAF mask should have zero gradient.
        outside = aerial.grad.clone()
        outside[2:6, 2:6] = 0.0
        assert outside.abs().sum().item() == pytest.approx(0.0)

    def test_batched_shape_supported(self) -> None:
        aerial = torch.full((2, 1, 8, 8), 0.40)
        sraf = torch.zeros((2, 1, 8, 8))
        sraf[:, :, 2:6, 2:6] = 1.0
        loss = sraf_print_penalty(aerial, sraf, print_threshold=0.30, margin=0.05)
        assert loss.dim() == 0
        assert loss.item() > 0.0

    def test_shape_mismatch_raises(self) -> None:
        aerial = torch.zeros((8, 8))
        sraf = torch.zeros((4, 4))
        with pytest.raises(ValueError, match="must share shape"):
            sraf_print_penalty(aerial, sraf)
