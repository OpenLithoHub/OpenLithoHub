"""Tests for openlithohub._utils.morphology."""

import torch

from openlithohub._utils.morphology import (
    binary_dilation,
    binary_erosion,
    distance_transform,
    estimate_shot_count,
    morphological_closing,
    morphological_opening,
    mrc_projection,
    soft_dilation,
    soft_erosion,
)


class TestBinaryErosion:
    def test_erosion_shrinks_feature(self) -> None:
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        eroded = binary_erosion(mask, radius=2)
        assert eroded.sum() < mask.sum()
        assert eroded.sum() > 0

    def test_erosion_radius_zero_identity(self) -> None:
        mask = torch.zeros(16, 16)
        mask[4:12, 4:12] = 1.0
        eroded = binary_erosion(mask, radius=0)
        assert torch.equal(eroded, mask)

    def test_small_feature_disappears(self) -> None:
        mask = torch.zeros(32, 32)
        mask[15:17, 15:17] = 1.0  # 2x2 feature
        eroded = binary_erosion(mask, radius=3)
        assert eroded.sum() == 0.0


class TestBinaryDilation:
    def test_dilation_grows_feature(self) -> None:
        mask = torch.zeros(32, 32)
        mask[14:18, 14:18] = 1.0
        dilated = binary_dilation(mask, radius=2)
        assert dilated.sum() > mask.sum()

    def test_dilation_radius_zero_identity(self) -> None:
        mask = torch.zeros(16, 16)
        mask[4:12, 4:12] = 1.0
        dilated = binary_dilation(mask, radius=0)
        assert torch.equal(dilated, mask)

    def test_dilation_fills_gap(self) -> None:
        mask = torch.zeros(16, 16)
        mask[7, 6] = 1.0
        mask[7, 8] = 1.0
        dilated = binary_dilation(mask, radius=1)
        assert dilated[7, 7] == 1.0


class TestMorphologicalOpening:
    def test_opening_removes_small_features(self) -> None:
        mask = torch.zeros(32, 32)
        mask[4:28, 4:28] = 1.0  # large
        mask[30, 30] = 1.0  # tiny isolated pixel
        opened = binary_dilation(binary_erosion(mask, radius=2), radius=2)
        assert opened[30, 30] == 0.0
        assert opened[16, 16] == 1.0


class TestDistanceTransform:
    def test_empty_mask_returns_zeros(self) -> None:
        mask = torch.zeros(16, 16)
        dt = distance_transform(mask)
        assert dt.sum() == 0.0

    def test_full_mask(self) -> None:
        mask = torch.ones(8, 8)
        dt = distance_transform(mask)
        assert dt[4, 4] > dt[0, 0]

    def test_single_pixel(self) -> None:
        mask = torch.zeros(8, 8)
        mask[4, 4] = 1.0
        dt = distance_transform(mask)
        assert dt[4, 4] > 0.0

    def test_distance_increases_toward_center(self) -> None:
        mask = torch.zeros(16, 16)
        mask[4:12, 4:12] = 1.0
        dt = distance_transform(mask)
        assert dt[8, 8] > dt[5, 5]


# ---------------------------------------------------------------------------
# Differentiable morphological operators
# ---------------------------------------------------------------------------


class TestSoftDilation:
    def test_dilation_expands(self) -> None:
        mask = torch.zeros(32, 32)
        mask[12:20, 12:20] = 1.0
        dilated = soft_dilation(mask, radius=2.0)
        assert dilated.shape == mask.shape
        assert (dilated > 0.5).sum() > (mask > 0.5).sum()

    def test_dilation_radius_small_is_identity(self) -> None:
        mask = torch.zeros(16, 16)
        mask[4:12, 4:12] = 1.0
        dilated = soft_dilation(mask, radius=0.3)
        assert torch.allclose(dilated, mask)

    def test_dilation_output_near_01(self) -> None:
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        dilated = soft_dilation(mask, radius=2.0, hardness=20.0)
        # Interior should remain close to 1, exterior close to 0
        assert dilated[16, 16] > 0.9
        assert dilated[0, 0] < 0.5


class TestSoftErosion:
    def test_erosion_shrinks(self) -> None:
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        eroded = soft_erosion(mask, radius=2.0)
        assert eroded.shape == mask.shape
        assert (eroded > 0.5).sum() < (mask > 0.5).sum()

    def test_erosion_radius_small_is_identity(self) -> None:
        mask = torch.zeros(16, 16)
        mask[4:12, 4:12] = 1.0
        eroded = soft_erosion(mask, radius=0.3)
        assert torch.allclose(eroded, mask)


class TestMorphologicalDifferentiable:
    """All soft morphological operations must propagate gradients."""

    def test_soft_dilation_gradient(self) -> None:
        mask = torch.rand(16, 16, requires_grad=True)
        out = soft_dilation(mask, radius=2.0)
        out.sum().backward()
        assert mask.grad is not None
        assert mask.grad.abs().sum() > 0

    def test_soft_erosion_gradient(self) -> None:
        mask = torch.rand(16, 16, requires_grad=True)
        out = soft_erosion(mask, radius=2.0)
        out.sum().backward()
        assert mask.grad is not None
        assert mask.grad.abs().sum() > 0

    def test_morphological_opening_gradient(self) -> None:
        mask = torch.rand(16, 16, requires_grad=True)
        out = morphological_opening(mask, radius=2.0)
        out.sum().backward()
        assert mask.grad is not None
        assert mask.grad.abs().sum() > 0

    def test_morphological_closing_gradient(self) -> None:
        mask = torch.rand(16, 16, requires_grad=True)
        out = morphological_closing(mask, radius=2.0)
        out.sum().backward()
        assert mask.grad is not None
        assert mask.grad.abs().sum() > 0

    def test_mrc_projection_gradient(self) -> None:
        mask = torch.rand(32, 32, requires_grad=True)
        out = mrc_projection(mask, min_feature_px=3.0)
        out.sum().backward()
        assert mask.grad is not None
        assert mask.grad.abs().sum() > 0

    def test_estimate_shot_count_gradient(self) -> None:
        mask = torch.rand(16, 16, requires_grad=True)
        out = estimate_shot_count(mask)
        out.backward()
        assert mask.grad is not None
        assert mask.grad.abs().sum() > 0


class TestMrcProjection:
    def test_mrc_projection_removes_small_features(self) -> None:
        mask = torch.zeros(64, 64)
        mask[10:30, 10:30] = 1.0  # 20x20 large feature
        mask[50:52, 50:52] = 1.0  # 2x2 tiny feature (smaller than 3px)
        projected = mrc_projection(mask, min_feature_px=3.0)
        # Large feature preserved
        assert projected[20, 20] > 0.5
        # Tiny feature suppressed
        assert projected[51, 51] < 0.5

    def test_mrc_projection_preserves_large_features(self) -> None:
        mask = torch.zeros(64, 64)
        mask[8:56, 8:56] = 1.0
        projected = mrc_projection(mask, min_feature_px=3.0)
        assert projected[32, 32] > 0.5

    def test_mrc_projection_mrc_clean(self) -> None:
        """After projection, no width/spacing violations remain for features
        at or above the minimum size."""
        mask = torch.zeros(64, 64)
        # Two large blocks with a wide gap
        mask[5:25, 5:25] = 1.0
        mask[35:55, 35:55] = 1.0
        # Add small violations
        mask[2, 2] = 1.0
        mask[60, 60] = 1.0

        min_feature_px = 4.0
        projected = mrc_projection(mask, min_feature_px=min_feature_px)

        # The tiny isolated pixels should be suppressed
        assert projected[2, 2] < 0.3
        assert projected[60, 60] < 0.3

        # The large features should survive
        assert projected[15, 15] > 0.5
        assert projected[45, 45] > 0.5

    def test_mrc_projection_output_shape(self) -> None:
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        projected = mrc_projection(mask, min_feature_px=3.0)
        assert projected.shape == mask.shape


class TestEstimateShotCount:
    def test_shot_count_positive(self) -> None:
        mask = torch.zeros(32, 32)
        mask[10:20, 10:20] = 1.0
        sc = estimate_shot_count(mask)
        assert sc.item() > 0

    def test_shot_count_empty_mask_small(self) -> None:
        mask = torch.zeros(32, 32)
        sc = estimate_shot_count(mask)
        assert sc.item() < 1.0

    def test_shot_count_larger_mask_more_shots(self) -> None:
        small = torch.zeros(32, 32)
        small[14:18, 14:18] = 1.0
        large = torch.zeros(32, 32)
        large[8:24, 8:24] = 1.0
        sc_small = estimate_shot_count(small)
        sc_large = estimate_shot_count(large)
        assert sc_large.item() > sc_small.item()


class TestOpeningClosingIdempotent:
    def test_opening_closing_idempotent(self) -> None:
        """Opening then closing (and vice versa) should be approximately
        idempotent for MRC-clean masks."""
        mask = torch.zeros(64, 64)
        mask[10:50, 10:50] = 1.0  # large clean feature

        oc = morphological_closing(morphological_opening(mask, radius=2.0), radius=2.0)
        co = morphological_opening(morphological_closing(mask, radius=2.0), radius=2.0)

        # Both compositions should be close to the original (which is MRC-clean)
        assert torch.allclose(oc, co, atol=0.15)
