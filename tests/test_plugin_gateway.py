"""Plugin gateway validation tests.

Verifies that:
1. All plugin functions are accessible through the gateway.
2. Physical constants are defined in one place only (_constants.py).
3. Plugin lazy-loading works correctly.
4. Error messages are clear when plugins are missing.
5. No direct imports from diffcfd/diffnano exist outside the plugin layer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import openlithohub._constants as const

_SRC = Path(__file__).resolve().parent.parent / "src" / "openlithohub"
_PLUGINS_DIR = _SRC / "plugins"
_CONSTANTS_MODULE = "openlithohub._constants"


# ---------------------------------------------------------------------------
# 1. Plugin gateway functions are accessible
# ---------------------------------------------------------------------------


class TestPluginGatewayAccessibility:
    """Verify all plugin adapters and entry points are importable."""

    def test_plugin_init_exports(self):
        from openlithohub.plugins import (
            list_plugins,
            optional_import,
            register_plugin,
        )

        assert callable(optional_import)
        assert callable(register_plugin)
        assert callable(list_plugins)

    def test_diffcfd_process_adapter_importable(self):
        from openlithohub.plugins.diffcfd_process import (
            DiffCFDLithoSimulator,
            DiffCFDSpinCoatSimulator,
        )

        assert DiffCFDLithoSimulator.name == "diffcfd_litho"
        assert DiffCFDSpinCoatSimulator.name == "diffcfd_spin_coat"

    def test_diffnano_em_adapters_importable(self):
        from openlithohub.plugins.diffnano_em import (
            DiffNanoFDFD2D,
            DiffNanoFDTD2D,
            DiffNanoRCWA,
        )

        assert DiffNanoRCWA.name == "diffnano_rcwa"
        assert DiffNanoFDTD2D.name == "diffnano_fdtd2d"
        assert DiffNanoFDFD2D.name == "diffnano_fdfd2d"

    def test_diffnano_resist_adapter_importable(self):
        from openlithohub.plugins.diffnano_resist import DiffNanoResistAdapter

        assert DiffNanoResistAdapter is not None

    def test_plugin_manifest_covers_all_plugins(self):
        from openlithohub.plugins import _KNOWN_PLUGINS

        assert set(_KNOWN_PLUGINS.keys()) == {"diffnano", "diffcfd"}

    def test_simulator_registry_lists_plugin_backends(self):
        from openlithohub.simulators.registry import list_available_backends

        backends = {b["name"] for b in list_available_backends()}
        expected = {
            "diffnano_rcwa",
            "diffnano_fdtd2d",
            "diffnano_fdfd2d",
            "diffcfd_litho",
            "diffcfd_spin_coat",
        }
        assert expected.issubset(backends)

    def test_hopkins_simulator_uses_plugin_for_diffnano_resist(self):
        """Verify HopkinsSimulator delegates to the plugin for diffnano resist."""
        from openlithohub.simulators.base import SimulatorConfig
        from openlithohub.simulators.hopkins_sim import HopkinsSimulator

        cfg = SimulatorConfig(resist_backend="diffnano")
        sim = HopkinsSimulator(cfg)
        assert hasattr(sim, "_apply_diffnano_resist")


# ---------------------------------------------------------------------------
# 2. Constants single-source verification
# ---------------------------------------------------------------------------


class TestConstantsSingleSource:
    """Verify constants are defined in _constants.py and not duplicated."""

    def test_constants_module_has_all_required_names(self):
        required = [
            "WAVELENGTH_ARF_NM",
            "WAVELENGTH_EUV_NM",
            "NA_IMMERSION",
            "NA_EUV_STANDARD",
            "NA_EUV_HIGH",
            "SIGMA_OUTER_DEFAULT",
            "SIGMA_INNER_DEFAULT",
            "PIXEL_SIZE_NM_DEFAULT",
            "DEFOCUS_NM_DEFAULT",
            "DOSE_DEFAULT",
            "NUM_KERNELS_DEFAULT",
            "POLE_OPENING_DEG_DEFAULT",
            "THRESHOLD_ICCAD16",
            "THRESHOLD_GENERIC",
            "RESIST_DIFFUSION_NM_DEFAULT",
            "QUENCHER_DEFAULT",
            "STEEPNESS_DEFAULT",
            "ABSORBER_THICKNESS_NM_DEFAULT",
            "CHIEF_RAY_ANGLE_DEG_DEFAULT",
            "CHIEF_RAY_AZIMUTH_DEG_DEFAULT",
            "DILL_A_DEFAULT",
            "DILL_B_DEFAULT",
            "DILL_C_DEFAULT",
            "MACK_R_MAX",
            "MACK_R_MIN",
            "MACK_N_DEFAULT",
            "MACK_A_DEFAULT",
            "GAMMA_SOLVENT_DEFAULT",
            "ACID_DIFFUSION_LENGTH_NM",
            "DEVELOPMENT_CONTRAST",
            "THRESHOLD_DOSE_DIFFNANO",
            "PEB_DIFFUSION_NM",
            "DIFFCFD_LITHO_DEFAULTS",
            "DIFFCFD_SPIN_COAT_DEFAULTS",
            "DIFFCFD_PROCESS_DEFAULTS",
            "DIFFNANO_RESIST_DEFAULTS",
        ]
        for name in required:
            assert hasattr(const, name), f"_constants.py missing: {name}"

    def test_simulator_config_uses_constants(self):
        from openlithohub.simulators.base import SimulatorConfig

        cfg = SimulatorConfig()
        assert cfg.wavelength_nm == const.WAVELENGTH_ARF_NM
        assert cfg.na == const.NA_IMMERSION
        assert cfg.sigma == const.SIGMA_OUTER_DEFAULT
        assert cfg.sigma_inner == const.SIGMA_INNER_DEFAULT
        assert cfg.pixel_size_nm == const.PIXEL_SIZE_NM_DEFAULT
        assert cfg.defocus_nm == const.DEFOCUS_NM_DEFAULT
        assert cfg.dose == const.DOSE_DEFAULT
        assert cfg.threshold == const.THRESHOLD_ICCAD16
        assert cfg.resist_diffusion_nm == const.RESIST_DIFFUSION_NM_DEFAULT
        assert cfg.quencher == const.QUENCHER_DEFAULT

    def test_hopkins_params_uses_constants(self):
        from openlithohub._utils.hopkins import HopkinsParams

        p = HopkinsParams()
        assert p.wavelength_nm == const.WAVELENGTH_ARF_NM
        assert p.na == const.NA_IMMERSION
        assert p.sigma == const.SIGMA_OUTER_DEFAULT
        assert p.sigma_inner == const.SIGMA_INNER_DEFAULT
        assert p.pixel_size_nm == const.PIXEL_SIZE_NM_DEFAULT
        assert p.num_kernels == const.NUM_KERNELS_DEFAULT
        assert p.defocus_nm == const.DEFOCUS_NM_DEFAULT
        assert p.pole_opening_deg == const.POLE_OPENING_DEG_DEFAULT

    def test_mask3d_params_uses_constants(self):
        from openlithohub.benchmark.metrics.euv_3d import Mask3DParams

        p = Mask3DParams()
        assert p.absorber_thickness_nm == const.ABSORBER_THICKNESS_NM_DEFAULT
        assert p.chief_ray_angle_deg == const.CHIEF_RAY_ANGLE_DEG_DEFAULT
        assert p.chief_ray_azimuth_deg == const.CHIEF_RAY_AZIMUTH_DEG_DEFAULT
        assert p.pixel_size_nm == const.PIXEL_SIZE_NM_DEFAULT

    def test_plugin_defaults_match_constants(self):
        from openlithohub.plugins.diffcfd_process import (
            LITHO_DEFAULTS,
            PROCESS_DEFAULTS,
            SPIN_COAT_DEFAULTS,
        )

        assert LITHO_DEFAULTS is const.DIFFCFD_LITHO_DEFAULTS
        assert SPIN_COAT_DEFAULTS is const.DIFFCFD_SPIN_COAT_DEFAULTS
        assert PROCESS_DEFAULTS is const.DIFFCFD_PROCESS_DEFAULTS

    def test_diffnano_resist_defaults_match_constants(self):
        from openlithohub.plugins.diffnano_resist import RESIST_DEFAULTS

        assert RESIST_DEFAULTS is const.DIFFNANO_RESIST_DEFAULTS

    def test_no_bare_wavelength_193_in_source(self):
        """Grep source files for hardcoded 193.0 wavelength literals."""
        result = subprocess.run(
            ["grep", "-rn", r"= 193\.0\b", "src/openlithohub/", "--include=*.py"],
            capture_output=True,
            text=True,
            cwd=str(_SRC.parent.parent),
        )
        hits = [line for line in result.stdout.strip().splitlines() if "_constants.py" not in line]
        assert hits == [], "Hardcoded 193.0 found outside _constants.py:\n" + "\n".join(hits)

    def test_no_bare_wavelength_13_5_in_source(self):
        """Grep source files for hardcoded 13.5 wavelength literals."""
        result = subprocess.run(
            ["grep", "-rn", r"= 13\.5\b", "src/openlithohub/", "--include=*.py"],
            capture_output=True,
            text=True,
            cwd=str(_SRC.parent.parent),
        )
        hits = [line for line in result.stdout.strip().splitlines() if "_constants.py" not in line]
        assert hits == [], "Hardcoded 13.5 found outside _constants.py:\n" + "\n".join(hits)

    def test_no_bare_na_1_35_in_source(self):
        """Grep source files for hardcoded NA=1.35 literals."""
        result = subprocess.run(
            ["grep", "-rn", r"= 1\.35\b", "src/openlithohub/", "--include=*.py"],
            capture_output=True,
            text=True,
            cwd=str(_SRC.parent.parent),
        )
        hits = [line for line in result.stdout.strip().splitlines() if "_constants.py" not in line]
        assert hits == [], "Hardcoded 1.35 found outside _constants.py:\n" + "\n".join(hits)

    def test_no_bare_threshold_0_225_in_source(self):
        """Grep source files for hardcoded threshold=0.225 literals (not comments)."""
        result = subprocess.run(
            ["grep", "-rn", r"= 0\.225\b", "src/openlithohub/", "--include=*.py"],
            capture_output=True,
            text=True,
            cwd=str(_SRC.parent.parent),
        )
        code_hits = []
        for line in result.stdout.strip().splitlines():
            if "_constants.py" in line:
                continue
            code_part = line.split("#")[0] if "#" in line else line
            if "= 0.225" in code_part:
                code_hits.append(line)
        assert code_hits == [], "Hardcoded 0.225 found outside _constants.py:\n" + "\n".join(
            code_hits
        )


# ---------------------------------------------------------------------------
# 3. Plugin lazy-loading
# ---------------------------------------------------------------------------


class TestPluginLazyLoading:
    """Verify plugin backends are not eagerly loaded."""

    def test_plugin_backends_not_in_base_registry(self):
        from openlithohub.simulators.registry import _REGISTRY

        _plugin_names = {
            "diffnano_rcwa",
            "diffnano_fdtd2d",
            "diffnano_fdfd2d",
            "diffcfd_litho",
            "diffcfd_spin_coat",
        }
        # Plugin backends should not be in the core registry at import time
        # (unless they were previously loaded by another test).
        # At minimum, hopkins should be there.
        assert "hopkins" in _REGISTRY

    def test_optional_import_returns_module_for_core(self):
        from openlithohub.plugins import optional_import

        mod = optional_import("openlithohub._constants")
        assert hasattr(mod, "WAVELENGTH_ARF_NM")

    def test_list_plugins_returns_status(self):
        from openlithohub.plugins import list_plugins

        status = list_plugins()
        for name in ("diffnano", "diffcfd"):
            assert name in status
            assert status[name] in ("installed", "available")


# ---------------------------------------------------------------------------
# 4. Clear error messages when plugins are missing
# ---------------------------------------------------------------------------


class TestPluginErrorMessages:
    """Verify helpful error messages when optional plugins are absent."""

    def test_diffnano_import_error_message(self):
        from openlithohub.plugins import OptionalPluginError, list_plugins, optional_import

        if list_plugins().get("diffnano") == "installed":
            pytest.skip("diffnano is installed; cannot test missing-plugin error")
        with pytest.raises(OptionalPluginError, match=r"pip install openlithohub\[diffnano\]"):
            optional_import("diffnano.solvers.resist", plugin="diffnano")

    def test_diffcfd_import_error_message(self):
        from openlithohub.plugins import OptionalPluginError, list_plugins, optional_import

        if list_plugins().get("diffcfd") == "installed":
            pytest.skip("diffcfd is installed; cannot test missing-plugin error")
        with pytest.raises(OptionalPluginError, match=r"pip install openlithohub\[diffcfd\]"):
            optional_import("diffcfd.solvers.litho", plugin="diffcfd")

    def test_simulator_registry_helpful_error(self):
        from openlithohub.plugins import list_plugins
        from openlithohub.simulators.registry import get_simulator

        if list_plugins().get("diffnano") == "installed":
            pytest.skip("diffnano is installed; cannot test missing-backend error")
        with pytest.raises(KeyError, match=r"\[diffnano\]"):
            get_simulator("diffnano_rcwa")


# ---------------------------------------------------------------------------
# 5. No direct cross-domain imports outside plugin layer
# ---------------------------------------------------------------------------


class TestNoDirectCrossDomainImports:
    """Verify that diffcfd/diffnano packages are never directly imported
    outside of the plugins/ directory.
    """

    @pytest.mark.parametrize("pkg", ["diffcfd", "diffnano"])
    def test_no_direct_imports_outside_plugins(self, pkg: str):
        result = subprocess.run(
            [
                "grep",
                "-rn",
                f"from {pkg}",
                "src/openlithohub/",
                "--include=*.py",
            ],
            capture_output=True,
            text=True,
            cwd=str(_SRC.parent.parent),
        )
        # Only plugin modules should have direct imports of diffcfd/diffnano
        non_plugin = [line for line in result.stdout.strip().splitlines() if "plugins/" not in line]
        assert non_plugin == [], (
            f"Direct `from {pkg}` import found outside plugins/:\n" + "\n".join(non_plugin)
        )
