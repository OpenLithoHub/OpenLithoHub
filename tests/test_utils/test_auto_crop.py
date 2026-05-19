"""Tests for openlithohub._utils.auto_crop."""

from __future__ import annotations

import torch

from openlithohub._utils.auto_crop import auto_crop, find_most_complex_window, score_complexity


class TestScoreComplexity:
    def test_empty_mask_scores_zero(self) -> None:
        mask = torch.zeros(64, 64)
        score = score_complexity(mask, window_px=8)
        assert score.shape == mask.shape
        assert score.max().item() == 0.0

    def test_dense_region_scores_higher_than_blank(self) -> None:
        mask = torch.zeros(128, 128)
        # Sparse: one block top-left.
        mask[8:24, 8:24] = 1.0
        # Dense: striped lines bottom-right.
        for y in range(70, 120, 4):
            mask[y : y + 2, 70:120] = 1.0
        score = score_complexity(mask, window_px=16)
        assert score[90, 90].item() > score[16, 16].item()


class TestFindMostComplexWindow:
    def test_small_mask_returns_full_extent(self) -> None:
        mask = torch.zeros(32, 32)
        mask[4:8, 4:8] = 1.0
        bbox = find_most_complex_window(mask, window_size=64)
        assert bbox == (0, 0, 32, 32)

    def test_window_centred_on_dense_region(self) -> None:
        mask = torch.zeros(512, 512)
        for y in range(380, 480, 4):
            mask[y : y + 2, 380:480] = 1.0
        y0, x0, y1, x1 = find_most_complex_window(mask, window_size=128)
        assert y1 - y0 == 128
        assert x1 - x0 == 128
        # The dense region centres around (430, 430); the chosen window must
        # include it.
        assert y0 <= 430 <= y1
        assert x0 <= 430 <= x1

    def test_window_stays_in_bounds(self) -> None:
        mask = torch.zeros(300, 300)
        # Put the dense region near the corner so the window has to clamp.
        for y in range(0, 80, 4):
            mask[y : y + 2, 0:80] = 1.0
        y0, x0, y1, x1 = find_most_complex_window(mask, window_size=128)
        assert 0 <= y0 < y1 <= 300
        assert 0 <= x0 < x1 <= 300
        assert y1 - y0 == 128
        assert x1 - x0 == 128


class TestAutoCrop:
    def test_passthrough_when_within_budget(self) -> None:
        mask = torch.zeros(256, 256)
        cropped, bbox = auto_crop(mask, target_size=1024)
        assert cropped.shape == mask.shape
        assert bbox == (0, 0, 256, 256)

    def test_returns_target_sized_crop_for_oversize_input(self) -> None:
        mask = torch.zeros(2048, 2048)
        for y in range(1500, 1700, 4):
            mask[y : y + 2, 1500:1700] = 1.0
        cropped, bbox = auto_crop(mask, target_size=512)
        assert cropped.shape == (512, 512)
        y0, x0, y1, x1 = bbox
        assert y1 - y0 == 512 and x1 - x0 == 512
        assert cropped.sum() > 0  # picked the busy region, not blank corners
