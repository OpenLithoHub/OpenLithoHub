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

    def test_calibre_simulate_raises_not_implemented(self) -> None:
        sim = CalibreSimulator(
            SimulatorConfig(extra={"calibre_home": "/opt/calibre", "runset": "x.svrf"})
        )
        with pytest.raises(NotImplementedError, match="vendor toolchain"):
            sim.simulate(_make_mask())

    def test_tachyon_validates_required_extras(self) -> None:
        with pytest.raises(ValueError, match="tachyon_home"):
            TachyonSimulator(SimulatorConfig())

    def test_tachyon_simulate_raises_not_implemented(self) -> None:
        sim = TachyonSimulator(
            SimulatorConfig(extra={"tachyon_home": "/opt/tachyon", "recipe": "x.tcl"})
        )
        with pytest.raises(NotImplementedError):
            sim.simulate(_make_mask())


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
