"""Tests for the simulator backend interface."""

from __future__ import annotations

import pytest
import torch

from openlithohub.simulators import (
    BaseSimulator,
    CalibreSimulator,
    HopkinsSimulator,
    SimulatorConfig,
    SimulatorResult,
    TachyonSimulator,
    get_simulator,
    list_simulators,
    register_simulator,
)


def _make_mask() -> torch.Tensor:
    mask = torch.zeros(64, 64)
    mask[16:48, 16:48] = 1.0
    return mask


class TestHopkinsSimulator:
    def test_simulate_returns_aerial_and_resist(self) -> None:
        sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
        result = sim.simulate(_make_mask())
        assert isinstance(result, SimulatorResult)
        assert result.backend == "hopkins"
        assert result.aerial.shape == (64, 64)
        assert result.resist is not None
        assert result.resist.shape == (64, 64)
        assert torch.all((result.resist == 0) | (result.resist == 1))

    def test_simulate_propagates_gradients(self) -> None:
        mask = _make_mask().requires_grad_(True)
        sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
        result = sim.simulate(mask)
        result.aerial.sum().backward()
        assert mask.grad is not None
        assert torch.isfinite(mask.grad).all()

    def test_batched_input(self) -> None:
        mask = _make_mask().unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)
        sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
        result = sim.simulate(mask)
        assert result.aerial.shape == (2, 1, 64, 64)
        assert result.metadata["differentiable"] is True

    def test_canonical_aerial_image_regression(self) -> None:
        """Pin the aerial intensity for the LithoBench-canonical optical config.

        Per Yang2023_LithoBench (NeurIPS'23) Table II /
        §3.2, the reference forward-simulation pipeline uses ArF 193nm,
        NA=1.35, sigma=0.7, 24 SOCS kernels, threshold=0.225. If any of
        those defaults — or the underlying SOCS decomposition — drifts,
        every published L2/PVB number computed against this backend
        becomes wrong by a constant. This test catches that drift.

        Reference values were rebaselined when issue #29 (polar-grid
        Jacobian for source samples) was fixed: pre-fix mean was 0.173868
        because the dc-side of the disk was over-weighted; post-fix mean
        of 0.161144 reflects the area-correct integration. Tolerances
        remain tight (~1e-4) because the kernel construction is
        deterministic.
        """
        sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
        result = sim.simulate(_make_mask())

        assert result.metadata["num_kernels"] == 24
        assert result.metadata["illumination"] == "circular"
        # Reference stats for a 32x32 centered pad on a 64x64 grid.
        assert result.aerial.mean().item() == pytest.approx(0.161144, abs=2e-4)
        assert result.aerial.max().item() == pytest.approx(0.942922, abs=2e-4)
        # Resist duty cycle at threshold=0.225, dose=1.0.
        assert result.resist.sum().item() == pytest.approx(1004.0, abs=4.0)

    def test_dose_actually_moves_resist_contour(self) -> None:
        """Issue #52: dose used to be cancelled in HopkinsSimulator.simulate
        because the threshold was scaled by dose AND the aerial was scaled
        by dose, so the comparison `aerial >= threshold * dose` reduced to
        `aerial_unit >= threshold_unit`. PW dose corners, MC dose jitter,
        and PVB dose-axis sweeps all reported zero dose sensitivity. After
        the fix, threshold is held constant and dose linearly scales the
        aerial — so a dose nudge clearly moves the resist contour."""
        mask = _make_mask()
        cfg_low = SimulatorConfig(pixel_size_nm=4.0, dose=0.95)
        cfg_high = SimulatorConfig(pixel_size_nm=4.0, dose=1.05)
        r_low = HopkinsSimulator(cfg_low).simulate(mask).resist
        r_high = HopkinsSimulator(cfg_high).simulate(mask).resist
        # Higher dose clears resist over a strictly larger area than lower
        # dose (more of the aerial exceeds the fixed threshold).
        assert r_high.sum().item() > r_low.sum().item()


class TestHopkinsWithConfig:
    """Direct coverage for HopkinsSimulator.with_config kernel-reuse path.

    A wrong _hparams_match predicate would silently produce stale aerial
    images: the cloned simulator would simulate with new dose/threshold
    using the *previous* config's optical kernels. These tests pin
    identity of the cached HopkinsParams object so a regression to "rebuild
    on every clone" or "reuse when we shouldn't" both fail loudly.
    """

    def _base_config(self) -> SimulatorConfig:
        return SimulatorConfig(
            wavelength_nm=193.0,
            na=1.35,
            sigma=0.7,
            pixel_size_nm=4.0,
            dose=1.0,
            threshold=0.225,
            extra={"num_kernels": 8, "illumination": "circular"},
        )

    def test_reuses_kernels_when_only_dose_changes(self) -> None:
        sim = HopkinsSimulator(self._base_config())
        new_cfg = SimulatorConfig(
            wavelength_nm=sim.config.wavelength_nm,
            na=sim.config.na,
            sigma=sim.config.sigma,
            pixel_size_nm=sim.config.pixel_size_nm,
            dose=2.5,
            threshold=sim.config.threshold,
            extra=dict(sim.config.extra),
        )
        sibling = sim.with_config(new_cfg)
        assert sibling is not sim
        assert sibling.config.dose == 2.5
        assert sibling._hparams is sim._hparams

    def test_reuses_kernels_when_only_threshold_changes(self) -> None:
        sim = HopkinsSimulator(self._base_config())
        new_cfg = SimulatorConfig(
            wavelength_nm=sim.config.wavelength_nm,
            na=sim.config.na,
            sigma=sim.config.sigma,
            pixel_size_nm=sim.config.pixel_size_nm,
            dose=sim.config.dose,
            threshold=0.4,
            extra=dict(sim.config.extra),
        )
        sibling = sim.with_config(new_cfg)
        assert sibling._hparams is sim._hparams

    @pytest.mark.parametrize(
        "field,value",
        [
            ("wavelength_nm", 13.5),
            ("na", 1.20),
            ("sigma", 0.9),
            ("sigma_inner", 0.4),
            ("pixel_size_nm", 2.0),
            ("defocus_nm", 30.0),
        ],
    )
    def test_rebuilds_kernels_when_optical_field_changes(self, field: str, value: float) -> None:
        sim = HopkinsSimulator(self._base_config())
        kwargs = {
            "wavelength_nm": sim.config.wavelength_nm,
            "na": sim.config.na,
            "sigma": sim.config.sigma,
            "sigma_inner": sim.config.sigma_inner,
            "pixel_size_nm": sim.config.pixel_size_nm,
            "defocus_nm": sim.config.defocus_nm,
            "dose": sim.config.dose,
            "threshold": sim.config.threshold,
            "extra": dict(sim.config.extra),
        }
        kwargs[field] = value
        sibling = sim.with_config(SimulatorConfig(**kwargs))
        assert sibling._hparams is not sim._hparams

    @pytest.mark.parametrize(
        "key,value",
        [
            ("num_kernels", 16),
            ("illumination", "annular"),
            ("dipole_angle_deg", 45.0),
            ("pole_opening_deg", 60.0),
        ],
    )
    def test_rebuilds_kernels_when_extra_changes(self, key: str, value: object) -> None:
        sim = HopkinsSimulator(self._base_config())
        new_extra = dict(sim.config.extra)
        new_extra[key] = value
        new_cfg = SimulatorConfig(
            wavelength_nm=sim.config.wavelength_nm,
            na=sim.config.na,
            sigma=sim.config.sigma,
            pixel_size_nm=sim.config.pixel_size_nm,
            dose=sim.config.dose,
            threshold=sim.config.threshold,
            extra=new_extra,
        )
        sibling = sim.with_config(new_cfg)
        assert sibling._hparams is not sim._hparams

    def test_reused_kernels_produce_dose_scaled_aerial(self) -> None:
        """End-to-end: kernel reuse must not strand the dose multiplier.

        If with_config copied the parent's _hparams but failed to thread
        the new dose through simulate(), the cloned simulator would emit
        the parent's aerial image. Numerical check that a 3x dose really
        produces a 3x aerial after kernel reuse.
        """
        sim = HopkinsSimulator(self._base_config())
        new_cfg = SimulatorConfig(
            wavelength_nm=sim.config.wavelength_nm,
            na=sim.config.na,
            sigma=sim.config.sigma,
            pixel_size_nm=sim.config.pixel_size_nm,
            dose=3.0,
            threshold=sim.config.threshold,
            extra=dict(sim.config.extra),
        )
        sibling = sim.with_config(new_cfg)
        assert sibling._hparams is sim._hparams  # reuse path was taken

        mask = _make_mask()
        base_aerial = sim.simulate(mask).aerial
        scaled_aerial = sibling.simulate(mask).aerial
        assert torch.allclose(scaled_aerial, base_aerial * 3.0, atol=1e-5)


class TestStubAdapters:
    def test_calibre_validates_required_extras(self) -> None:
        with pytest.raises(ValueError, match="calibre_home"):
            CalibreSimulator(SimulatorConfig())

    def test_calibre_non_mock_raises_toolchain_error(self) -> None:
        sim = CalibreSimulator(
            SimulatorConfig(extra={"calibre_home": "/opt/calibre", "runset": "x.svrf"})
        )
        with pytest.raises(Exception, match="preflight failed"):
            sim.simulate(_make_mask())

    def test_tachyon_validates_required_extras(self) -> None:
        with pytest.raises(ValueError, match="tachyon_home"):
            TachyonSimulator(SimulatorConfig())

    def test_tachyon_non_mock_raises_toolchain_error(self) -> None:
        sim = TachyonSimulator(
            SimulatorConfig(extra={"tachyon_home": "/opt/tachyon", "recipe": "x.tcl"})
        )
        with pytest.raises(Exception, match="preflight failed"):
            sim.simulate(_make_mask())


class TestMockMode:
    def test_tachyon_mock_mode_simulate(self) -> None:
        sim = TachyonSimulator(
            SimulatorConfig(
                pixel_size_nm=4.0,
                extra={"mock_mode": True},
            )
        )
        assert sim.mock_mode is True
        result = sim.simulate(_make_mask())
        assert isinstance(result, SimulatorResult)
        assert result.backend == "tachyon"
        assert result.metadata["mock"] is True
        assert result.aerial.shape == (64, 64)
        assert result.resist is not None

    def test_calibre_mock_mode_simulate(self) -> None:
        sim = CalibreSimulator(
            SimulatorConfig(
                pixel_size_nm=4.0,
                extra={"mock_mode": True},
            )
        )
        assert sim.mock_mode is True
        result = sim.simulate(_make_mask())
        assert isinstance(result, SimulatorResult)
        assert result.backend == "calibre"
        assert result.metadata["mock"] is True
        assert result.aerial.shape == (64, 64)
        assert result.resist is not None

    def test_mock_aerial_values_in_range(self) -> None:
        sim = TachyonSimulator(SimulatorConfig(extra={"mock_mode": True}))
        result = sim.simulate(_make_mask())
        assert result.aerial.min() >= 0.0
        assert result.aerial.max() <= 1.0

    def test_mock_resist_is_binary(self) -> None:
        sim = CalibreSimulator(SimulatorConfig(extra={"mock_mode": True}))
        result = sim.simulate(_make_mask())
        assert torch.all((result.resist == 0) | (result.resist == 1))


class TestPreflight:
    def test_tachyon_mock_preflight_passes(self) -> None:
        sim = TachyonSimulator(SimulatorConfig(extra={"mock_mode": True}))
        status = sim.preflight()
        assert status.ok is True
        assert status.tool_found is True
        assert status.license_ok is True

    def test_calibre_mock_preflight_passes(self) -> None:
        sim = CalibreSimulator(SimulatorConfig(extra={"mock_mode": True}))
        status = sim.preflight()
        assert status.ok is True

    def test_tachyon_real_preflight_fails_without_tool(self) -> None:
        sim = TachyonSimulator(
            SimulatorConfig(
                extra={
                    "tachyon_home": "/nonexistent/tachyon",
                    "recipe": "x.tcl",
                }
            )
        )
        status = sim.preflight()
        assert status.ok is False
        assert len(status.messages) > 0

    def test_calibre_real_preflight_fails_without_tool(self) -> None:
        sim = CalibreSimulator(
            SimulatorConfig(
                extra={
                    "calibre_home": "/nonexistent/calibre",
                    "runset": "x.svrf",
                }
            )
        )
        status = sim.preflight()
        assert status.ok is False
        assert len(status.messages) > 0


class TestProtocolConformance:
    def test_tachyon_satisfies_protocol(self) -> None:
        from openlithohub.simulators.commercial import CommercialSimulatorAdapter

        sim = TachyonSimulator(SimulatorConfig(extra={"mock_mode": True}))
        assert isinstance(sim, CommercialSimulatorAdapter)

    def test_calibre_satisfies_protocol(self) -> None:
        from openlithohub.simulators.commercial import CommercialSimulatorAdapter

        sim = CalibreSimulator(SimulatorConfig(extra={"mock_mode": True}))
        assert isinstance(sim, CommercialSimulatorAdapter)


class TestRegistry:
    def test_get_simulator_known(self) -> None:
        sim = get_simulator("hopkins")
        assert isinstance(sim, HopkinsSimulator)

    def test_get_simulator_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown simulator"):
            get_simulator("nonexistent")

    def test_list_simulators_includes_defaults(self) -> None:
        names = list_simulators()
        assert {"hopkins", "calibre", "tachyon"}.issubset(names)

    def test_register_custom_simulator(self) -> None:
        class FakeSim(BaseSimulator):
            name = "fake"

            def simulate(self, mask: torch.Tensor) -> SimulatorResult:
                return SimulatorResult(aerial=torch.zeros_like(mask), backend="fake")

        register_simulator("fake", FakeSim)
        try:
            sim = get_simulator("fake")
            assert isinstance(sim, FakeSim)
        finally:
            from openlithohub.simulators import registry as _registry

            _registry._REGISTRY.pop("fake", None)


class TestParity:
    """Schema-only parity between the open Hopkins backend and commercial stubs.

    The vendor toolchains (Calibre / Tachyon) are not installable in CI,
    so a numerical Hopkins ↔ Calibre/Tachyon comparison is impossible
    here. What we *can* lock down is that the three backends agree on
    the public surface — same ``SimulatorConfig`` keys, same registry
    handle, same ``SimulatorResult`` envelope on success — so a refactor
    that drifts the stubs away from Hopkins fails CI immediately.
    """

    @pytest.mark.parametrize(
        "name,extras",
        [
            ("hopkins", {}),
            ("calibre", {"calibre_home": "/opt/calibre", "runset": "x.svrf"}),
            ("tachyon", {"tachyon_home": "/opt/tachyon", "recipe": "x.tcl"}),
        ],
    )
    def test_construction_accepts_shared_config(self, name: str, extras: dict) -> None:
        # Optical/process knobs from SimulatorConfig must round-trip
        # through every backend constructor — drift here would mean the
        # leaderboard cannot evaluate a model on more than one backend
        # without rewriting the config.
        cfg = SimulatorConfig(
            wavelength_nm=193.0,
            na=1.35,
            sigma=0.7,
            pixel_size_nm=4.0,
            threshold=0.225,
            dose=1.0,
            extra=extras,
        )
        sim = get_simulator(name, cfg)
        assert sim.config.wavelength_nm == 193.0
        assert sim.config.na == 1.35
        assert sim.config.threshold == 0.225

    def test_hopkins_returns_complete_result(self) -> None:
        # Anchor: Hopkins is the open reference. SimulatorResult on a
        # successful run carries an aerial tensor of the input shape,
        # a binary resist contour, and the backend tag.
        sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
        result = sim.simulate(_make_mask())
        assert isinstance(result, SimulatorResult)
        assert result.aerial.shape == (64, 64)
        assert result.resist is not None and result.resist.shape == (64, 64)
        assert result.backend == "hopkins"

    def test_stubs_share_simulate_signature(self) -> None:
        # The stub ``simulate`` raises NotImplementedError but must
        # accept exactly the same call shape as Hopkins — refactors that
        # break this contract would silently divert one backend's API.
        import inspect

        hopkins_sig = inspect.signature(HopkinsSimulator.simulate)
        for cls in (CalibreSimulator, TachyonSimulator):
            assert inspect.signature(cls.simulate) == hopkins_sig
