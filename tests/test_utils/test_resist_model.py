"""Tests for openlithohub._utils.resist_model."""

import torch

from openlithohub._utils.resist_model import simulate_resist, simulate_resist_soft


class TestSimulateResist:
    def test_binary_output(self) -> None:
        aerial = torch.rand(32, 32)
        resist = simulate_resist(aerial)
        unique = resist.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique)

    def test_high_intensity_passes(self) -> None:
        aerial = torch.ones(16, 16)
        resist = simulate_resist(aerial, quencher_concentration=0.0, acid_diffusion_length_nm=0.0)
        assert resist.sum() == resist.numel()

    def test_low_intensity_blocked(self) -> None:
        aerial = torch.zeros(16, 16)
        resist = simulate_resist(aerial)
        assert resist.sum() == 0.0

    def test_diffusion_blurs_edges(self) -> None:
        aerial = torch.zeros(32, 32)
        aerial[10:22, 10:22] = 1.0
        resist_no_diff = simulate_resist(aerial, acid_diffusion_length_nm=0.0)
        resist_with_diff = simulate_resist(aerial, acid_diffusion_length_nm=3.0)
        # Diffusion should change the result (either widen or narrow features)
        assert not torch.equal(resist_no_diff, resist_with_diff)

    def test_quencher_narrows_features(self) -> None:
        aerial = torch.zeros(32, 32)
        aerial[8:24, 8:24] = 0.8
        resist_low_q = simulate_resist(aerial, quencher_concentration=0.0, threshold=0.5)
        resist_high_q = simulate_resist(aerial, quencher_concentration=0.3, threshold=0.5)
        assert resist_high_q.sum() <= resist_low_q.sum()


class TestSimulateResistSoft:
    def test_output_in_0_1_range(self) -> None:
        aerial = torch.rand(16, 16)
        resist = simulate_resist_soft(aerial)
        assert resist.min() >= 0.0
        assert resist.max() <= 1.0

    def test_differentiable(self) -> None:
        aerial = torch.rand(16, 16, requires_grad=True)
        resist = simulate_resist_soft(aerial)
        loss = resist.sum()
        loss.backward()
        assert aerial.grad is not None

    def test_high_steepness_approaches_binary(self) -> None:
        aerial = torch.zeros(16, 16)
        aerial[4:12, 4:12] = 1.0
        resist = simulate_resist_soft(
            aerial, steepness=200.0, acid_diffusion_length_nm=0.0, quencher_concentration=0.0
        )
        nearly_binary = ((resist > 0.99) | (resist < 0.01)).float()
        assert nearly_binary.mean() > 0.9
