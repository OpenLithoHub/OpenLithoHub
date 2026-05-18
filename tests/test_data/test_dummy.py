"""Tests for the procedural dummy layout generator."""

from __future__ import annotations

import pytest
import torch

from openlithohub.data import DummyLayoutSpec, generate_dummy_layout, generate_dummy_pair


class TestGenerateDummyLayout:
    def test_default_returns_binary_tensor(self) -> None:
        mask = generate_dummy_layout(size=128, seed=0)
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (128, 128)
        assert mask.dtype == torch.float32
        unique = torch.unique(mask)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_seed_is_deterministic(self) -> None:
        a = generate_dummy_layout(size=64, seed=42)
        b = generate_dummy_layout(size=64, seed=42)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self) -> None:
        # Use a spec where the morphological kernel is gentle enough that
        # different seeds produce different post-cleanup layouts.
        spec_a = DummyLayoutSpec(size=256, seed=1, min_width_nm=8.0, min_spacing_nm=8.0)
        spec_b = DummyLayoutSpec(size=256, seed=2, min_width_nm=8.0, min_spacing_nm=8.0)
        a = generate_dummy_layout(spec_a)
        b = generate_dummy_layout(spec_b)
        assert not torch.equal(a, b)

    def test_fill_ratio_in_expected_range(self) -> None:
        mask = generate_dummy_layout(size=256, seed=0)
        fill = float(mask.mean())
        # After morphological cleanup the actual fill is below the target ratio
        # but should still be a meaningful fraction of the canvas.
        assert 0.05 < fill < 0.6

    def test_size_below_minimum_rejected(self) -> None:
        with pytest.raises(ValueError, match="size must be"):
            generate_dummy_layout(size=16, seed=0)

    def test_invalid_fill_ratio_rejected(self) -> None:
        with pytest.raises(ValueError, match="fill_ratio"):
            generate_dummy_layout(spec=DummyLayoutSpec(fill_ratio=0.0))

    def test_pair_design_shape_matches_mask(self) -> None:
        design, mask = generate_dummy_pair(size=64, seed=0)
        assert design.shape == mask.shape == (64, 64)
        # Mask is dilated, so it covers at least as much as the design
        assert float(mask.sum()) >= float(design.sum())
