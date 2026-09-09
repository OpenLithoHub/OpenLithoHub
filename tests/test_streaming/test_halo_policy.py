"""Halo policy tests (RFC 0008, prompt §4-6)."""

import pytest
import torch

from openlithohub.streaming.halo_policy import (
    HaloContext,
    HaloRequirement,
    HaloStatus,
    KernelTailHaloPolicy,
    LegacyFixedHaloPolicy,
    PhysicalInteractionHaloPolicy,
    combine_requirements,
    estimate_minimum_halo,
    kernel_tail_mass,
)


class TestLegacyFixed:
    def test_fixed_halo(self):
        req = LegacyFixedHaloPolicy(128).required_halo(HaloContext(pixel_nm=1.0, tile_core_px=512))
        assert req.halo_px == 128
        assert req.status is HaloStatus.FIXED
        assert not req.certified

    def test_clamped_to_core(self):
        req = LegacyFixedHaloPolicy(128).required_halo(HaloContext(pixel_nm=1.0, tile_core_px=64))
        assert req.halo_px == 63


class TestPhysicalInteraction:
    def test_max_of_sources(self):
        class Model:
            RECEPTIVE_FIELD_PX = 24

        node = type("Node", (), {"optical_radius_nm": 120.0})()
        ctx = HaloContext(pixel_nm=4.0, tile_core_px=512, node=node, model=Model())
        req = PhysicalInteractionHaloPolicy().required_halo(ctx)
        # OIR = ceil(120/4) = 30 -> round up to 32; RF 24 < 32
        assert req.halo_px == 32
        assert req.status is HaloStatus.PHYSICS_ESTIMATED
        assert "optical" in req.reason

    def test_no_sources_gives_zero(self):
        req = PhysicalInteractionHaloPolicy().required_halo(
            HaloContext(pixel_nm=1.0, tile_core_px=256)
        )
        assert req.halo_px == 0


def _gaussian_kernel(size: int = 33, sigma: float = 2.0) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


class TestKernelTail:
    def test_tighter_tolerance_needs_bigger_halo(self):
        kernel = _gaussian_kernel()
        loose = KernelTailHaloPolicy(tolerance=1e-2).required_halo(
            HaloContext(pixel_nm=1.0, tile_core_px=512, kernel=kernel)
        )
        tight = KernelTailHaloPolicy(tolerance=1e-4).required_halo(
            HaloContext(pixel_nm=1.0, tile_core_px=512, kernel=kernel)
        )
        assert tight.halo_px >= loose.halo_px
        assert kernel_tail_mass(kernel, tight.halo_px) <= 1e-4

    def test_certified_status_when_sample_trusted(self):
        req = KernelTailHaloPolicy(tolerance=1e-3, trusted_sample=True).required_halo(
            HaloContext(pixel_nm=1.0, tile_core_px=512, kernel=_gaussian_kernel())
        )
        assert req.status is HaloStatus.CERTIFIED_SUFFICIENT
        assert req.certified
        assert req.error_bound == pytest.approx(1e-3)

    def test_untrusted_sample_never_certified(self):
        req = KernelTailHaloPolicy(tolerance=1e-3, trusted_sample=False).required_halo(
            HaloContext(pixel_nm=1.0, tile_core_px=512, kernel=_gaussian_kernel())
        )
        assert req.status is HaloStatus.PHYSICS_ESTIMATED
        assert not req.certified

    def test_no_kernel_is_inconclusive(self):
        req = KernelTailHaloPolicy().required_halo(
            HaloContext(pixel_nm=1.0, tile_core_px=256, kernel=None)
        )
        assert req.status is HaloStatus.INCONCLUSIVE

    def test_degenerate_kernel_inconclusive(self):
        req = KernelTailHaloPolicy().required_halo(
            HaloContext(pixel_nm=1.0, tile_core_px=256, kernel=torch.zeros(9, 9))
        )
        assert req.status is HaloStatus.INCONCLUSIVE

    def test_invalid_tolerance(self):
        with pytest.raises(ValueError, match="tolerance"):
            KernelTailHaloPolicy(tolerance=0.0)


class TestCombine:
    def test_max_halo_and_demotion(self):
        fixed = HaloRequirement(128, "legacy", None, False, "fixed", HaloStatus.FIXED)
        certified = HaloRequirement(
            32, "kernel_tail", 1e-3, True, "tail", HaloStatus.CERTIFIED_SUFFICIENT
        )
        merged = combine_requirements(fixed, certified)
        assert merged.halo_px == 128
        assert merged.status is HaloStatus.PHYSICS_ESTIMATED  # fixed demotes
        assert not merged.certified

    def test_all_certified_stays_certified(self):
        a = HaloRequirement(16, "a", 1e-3, True, "a", HaloStatus.CERTIFIED_SUFFICIENT)
        b = HaloRequirement(24, "b", 1e-2, True, "b", HaloStatus.CERTIFIED_SUFFICIENT)
        merged = combine_requirements(a, b)
        assert merged.halo_px == 24
        assert merged.certified
        assert merged.error_bound == pytest.approx(1e-2)

    def test_inconclusive_poisons(self):
        good = HaloRequirement(16, "a", 1e-3, True, "a", HaloStatus.CERTIFIED_SUFFICIENT)
        bad = HaloRequirement(16, "b", None, False, "b", HaloStatus.INCONCLUSIVE)
        assert combine_requirements(good, bad).status is HaloStatus.INCONCLUSIVE

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            combine_requirements()

    def test_certified_requires_bound(self):
        with pytest.raises(ValueError, match="CERTIFIED_SUFFICIENT"):
            HaloRequirement(8, "x", None, True, "x", HaloStatus.CERTIFIED_SUFFICIENT)


class TestEstimateMinimumHalo:
    @staticmethod
    def _deterministic_layout(h: int = 64, w: int = 64) -> torch.Tensor:
        # deterministic pseudo-layout (no RNG needed for a data check)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        return ((xx * 7 + yy * 13) % 16).float() / 16.0

    def test_identity_forward_stable_immediately(self):
        full = self._deterministic_layout()
        req = estimate_minimum_halo(
            lambda t: t,
            core_bbox_px=(16, 16, 48, 48),
            full_context=full,
            candidate_halos=(0, 4, 8, 16),
            tolerance=1e-6,
        )
        assert req.status is HaloStatus.EMPIRICALLY_STABLE
        assert req.halo_px == 0

    def test_shifted_forward_needs_larger_halo(self):
        h, w = 64, 64
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        full = torch.sin(0.3 * xx.float()) * torch.cos(0.2 * yy.float())

        def blur(tile: torch.Tensor) -> torch.Tensor:
            kernel = torch.ones(1, 1, 5, 5) / 25.0
            padded = torch.nn.functional.pad(
                tile.float().unsqueeze(0).unsqueeze(0), (2, 2, 2, 2), mode="replicate"
            )
            return torch.nn.functional.conv2d(padded, kernel).squeeze()

        req = estimate_minimum_halo(
            blur,
            core_bbox_px=(16, 16, 48, 48),
            full_context=full,
            candidate_halos=(0, 2, 4, 8, 16),
            tolerance=1e-3,
        )
        # a 5x5 blur needs ~2px of context; whatever the exact answer, the
        # stable halo must have a bounded discrepancy
        assert req.status is HaloStatus.EMPIRICALLY_STABLE
        assert req.error_bound == pytest.approx(1e-3)

    def test_never_stabilises_is_inconclusive(self):
        full = self._deterministic_layout()
        calls = {"n": 0}

        def drifting(tile: torch.Tensor) -> torch.Tensor:
            # deterministic but different on every call -> discrepancy never
            # settles between candidate halos
            calls["n"] += 1
            return tile + 0.25 * calls["n"]

        req = estimate_minimum_halo(
            drifting,
            core_bbox_px=(16, 16, 48, 48),
            full_context=full,
            candidate_halos=(0, 4),
            tolerance=1e-12,
        )
        assert req.status is HaloStatus.INCONCLUSIVE

    def test_empty_candidates_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            estimate_minimum_halo(
                lambda t: t,
                core_bbox_px=(0, 0, 8, 8),
                full_context=self._deterministic_layout(16, 16),
                candidate_halos=[],
            )
