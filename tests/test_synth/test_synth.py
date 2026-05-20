"""Tests for the synthetic layout generator."""

from __future__ import annotations

import pytest
import torch

from openlithohub.benchmark import check_mrc
from openlithohub.synth import (
    PDK_PRESETS,
    DiffusionLayoutGenerator,
    PatternKind,
    generate_layout,
    generate_synthetic_batch,
    get_pdk,
)


class TestPdk:
    def test_known_presets(self) -> None:
        assert "freepdk45" in PDK_PRESETS
        assert "asap7" in PDK_PRESETS

    def test_get_pdk_case_insensitive(self) -> None:
        assert get_pdk("FreePDK45").name == "freepdk45"

    def test_unknown_pdk_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown PDK"):
            get_pdk("nonexistent")

    def test_derived_pixel_quantities(self) -> None:
        rules = get_pdk("freepdk45")
        assert rules.min_width_px >= 1
        assert rules.pitch_nm == rules.min_width_nm + rules.min_spacing_nm


class TestGenerate:
    @pytest.mark.parametrize("pattern", list(PatternKind))
    def test_each_pattern_returns_binary_mask(self, pattern: PatternKind) -> None:
        mask = generate_layout(pattern, "freepdk45", size=128, seed=0)
        assert mask.shape == (128, 128)
        assert torch.all((mask == 0) | (mask == 1))

    def test_seed_determinism(self) -> None:
        a = generate_layout(PatternKind.RANDOM_LOGIC, "freepdk45", size=128, seed=42)
        b = generate_layout(PatternKind.RANDOM_LOGIC, "freepdk45", size=128, seed=42)
        assert torch.equal(a, b)

    def test_different_seeds_produce_different_layouts(self) -> None:
        a = generate_layout(PatternKind.RANDOM_LOGIC, "freepdk45", size=128, seed=1)
        b = generate_layout(PatternKind.RANDOM_LOGIC, "freepdk45", size=128, seed=2)
        assert not torch.equal(a, b)

    def test_batch(self) -> None:
        batch = generate_synthetic_batch(PatternKind.SRAM, 3, "asap7", size=128, seed=0)
        assert batch.masks.shape == (3, 128, 128)
        assert batch.pdk.name == "asap7"
        assert batch.pattern == PatternKind.SRAM
        assert batch.seeds == [0, 1, 2]

    @pytest.mark.parametrize("pattern", list(PatternKind))
    def test_layouts_pass_mrc(self, pattern: PatternKind) -> None:
        rules = get_pdk("freepdk45")
        mask = generate_layout(pattern, rules, size=128, seed=0)
        report = check_mrc(
            mask,
            min_width_nm=rules.min_width_nm,
            min_spacing_nm=rules.min_spacing_nm,
            pixel_size_nm=rules.pixel_size_nm,
        )
        assert report.violation_rate == 0.0, (
            f"{pattern.value} violates MRC: {report.violation_rate}"
        )


class TestEnforceDRC:
    """Verify the DRC pass corrects constructed violations.

    The round-trip-on-clean tests above only confirm generators emit valid
    output; they would still pass if `_enforce_drc` were a no-op. These
    tests construct known-bad inputs and assert correction.
    """

    def test_sub_min_width_sliver_is_removed(self) -> None:
        from openlithohub.synth.rule_based import _enforce_drc

        rules = get_pdk("freepdk45")
        # Single-pixel-wide vertical sliver — narrower than min_width_px.
        mask = torch.zeros(64, 64)
        mask[10:50, 30] = 1.0
        assert mask.sum() > 0  # sanity
        cleaned = _enforce_drc(mask, rules)
        # Opening at radius >= 1 must remove a 1-px-wide line.
        assert cleaned.sum() == 0

    def test_at_threshold_feature_survives(self) -> None:
        from openlithohub.synth.rule_based import _enforce_drc

        rules = get_pdk("freepdk45")
        w = rules.min_width_px
        mask = torch.zeros(64, 64)
        mask[10:54, 20 : 20 + w] = 1.0
        cleaned = _enforce_drc(mask, rules)
        # Feature exactly at min_width_px must be preserved by opening
        # with radius (min_width_px - 1) // 2.
        assert cleaned.sum() > 0


class TestDiffusionStub:
    def test_construct_then_sample_raises(self) -> None:
        gen = DiffusionLayoutGenerator(pdk="freepdk45")
        with pytest.raises(NotImplementedError, match="diffusion path"):
            gen.sample(n=1, size=64, seed=0)
