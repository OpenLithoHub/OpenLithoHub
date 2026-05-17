"""Tests for openlithohub._utils.morphology."""

import torch

from openlithohub._utils.morphology import binary_dilation, binary_erosion, distance_transform


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
