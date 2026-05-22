"""Tests for batch active sampling for hotspot pattern selection."""

from __future__ import annotations

import pytest
import torch

from openlithohub.baselines.hotspot_batchal import batch_active_select, extract_clip_features


class TestBatchActiveSelect:
    def test_returns_k_unique_indices(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(50, 8)
        probs = torch.rand(50)
        out = batch_active_select(features, probs, k=10)
        assert out.shape == (10,)
        assert out.dtype == torch.long
        assert len(set(out.tolist())) == 10

    def test_first_index_is_highest_uncertainty(self) -> None:
        torch.manual_seed(1)
        features = torch.randn(20, 4)
        probs = torch.linspace(0.0, 1.0, 20)
        out = batch_active_select(features, probs, k=5)
        assert int(out[0].item()) == 19

    def test_diversity_avoids_duplicate_features(self) -> None:
        # Build a pool where the top-3 highest-uncertainty entries are
        # near-duplicates and a 4th entry is highly uncertain but
        # orthogonal. Diversity-aware selection should prefer the
        # orthogonal one over the near-duplicate.
        features = torch.zeros(10, 4)
        features[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        features[1] = torch.tensor([1.0, 0.001, 0.0, 0.0])
        features[2] = torch.tensor([1.0, 0.0, 0.001, 0.0])
        features[3] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # orthogonal seed
        features[4:] = torch.randn(6, 4) * 0.01  # low-feature noise
        probs = torch.tensor([0.99, 0.98, 0.97, 0.96, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        out = batch_active_select(features, probs, k=2, n=4)
        selected = set(out.tolist())
        assert 0 in selected, "highest-uncertainty seed must be picked first"
        assert 3 in selected, "diversity must beat the near-duplicate at index 1 or 2"

    def test_deterministic(self) -> None:
        torch.manual_seed(42)
        features = torch.randn(30, 6)
        probs = torch.rand(30)
        a = batch_active_select(features, probs, k=7)
        b = batch_active_select(features, probs, k=7)
        assert torch.equal(a, b)

    def test_top_n_prefilter_constrains_search(self) -> None:
        # When n == k, no diversity room — output must equal top-k by probability.
        features = torch.randn(20, 4)
        probs = torch.linspace(0.0, 1.0, 20)
        out = batch_active_select(features, probs, k=5, n=5)
        assert sorted(out.tolist()) == [15, 16, 17, 18, 19]

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"k": 0}, "1 <= k"),
            ({"k": 100}, "1 <= k"),
            ({"k": 5, "n": 3}, "k=5 <= n"),
            ({"k": 5, "n": 100}, "n <= P"),
        ],
    )
    def test_invalid_args_raise(self, kwargs: dict, match: str) -> None:
        features = torch.randn(20, 4)
        probs = torch.rand(20)
        with pytest.raises(ValueError, match=match):
            batch_active_select(features, probs, **kwargs)

    def test_probabilities_out_of_range_raises(self) -> None:
        features = torch.randn(10, 4)
        probs = torch.tensor([0.5] * 9 + [1.5])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            batch_active_select(features, probs, k=3)


class TestExtractClipFeatures:
    def test_extract_pools_each_clip(self) -> None:
        design = torch.zeros(100, 100)
        design[20:30, 20:30] = 1.0
        clip_sites = [
            {"x0_nm": 0.0, "y0_nm": 0.0, "x1_nm": 50.0, "y1_nm": 50.0},
            {"x0_nm": 50.0, "y0_nm": 50.0, "x1_nm": 100.0, "y1_nm": 100.0},
        ]
        feats = extract_clip_features(design, clip_sites, pixel_nm=1.0, feature_dim=4)
        assert feats.shape == (2, 16)
        # First clip overlaps the bright square; mean must be > 0.
        assert float(feats[0].mean()) > 0.0
        # Second clip is in empty space.
        assert float(feats[1].abs().sum()) == pytest.approx(0.0)

    def test_empty_clip_list_returns_zero_rows(self) -> None:
        design = torch.zeros(10, 10)
        feats = extract_clip_features(design, [], pixel_nm=1.0, feature_dim=4)
        assert feats.shape == (0, 16)

    def test_clip_clipped_to_design_bounds(self) -> None:
        design = torch.ones(10, 10)
        clip_sites = [{"x0_nm": -50.0, "y0_nm": -50.0, "x1_nm": 5.0, "y1_nm": 5.0}]
        feats = extract_clip_features(design, clip_sites, pixel_nm=1.0, feature_dim=2)
        assert feats.shape == (1, 4)
        assert torch.all(feats == 1.0)

    def test_origin_offset_applied(self) -> None:
        design = torch.zeros(10, 10)
        design[2:5, 2:5] = 1.0
        # Clip at world coords (1002, 1002) → (1005, 1005); origin shifts it to pixel (2,2).
        clip_sites = [{"x0_nm": 1002.0, "y0_nm": 1002.0, "x1_nm": 1005.0, "y1_nm": 1005.0}]
        feats = extract_clip_features(
            design, clip_sites, pixel_nm=1.0, origin_nm=(1000.0, 1000.0), feature_dim=1
        )
        assert float(feats[0]) == pytest.approx(1.0)
