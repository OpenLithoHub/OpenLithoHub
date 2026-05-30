"""Tests for the optional plugin infrastructure (P0)."""

from __future__ import annotations

import pytest

from openlithohub.plugins import (
    LithoPlugin,
    OptionalPluginError,
    list_plugins,
    optional_import,
)
from openlithohub.simulators.base import SimulatorConfig
from openlithohub.simulators.registry import (
    get_simulator,
    list_available_backends,
    list_simulators,
)

# ---------------------------------------------------------------------------
# optional_import
# ---------------------------------------------------------------------------

class TestOptionalImport:
    def test_missing_plugin_raises_with_plugin_hint(self):
        with pytest.raises(OptionalPluginError, match=r"\[diffnano\]"):
            optional_import("diffnano.solvers.resist", plugin="diffnano")

    def test_missing_module_raises_raw_import_error_without_plugin(self):
        with pytest.raises(ImportError, match="nonexistent_module"):
            optional_import("nonexistent_module_xyz")

    def test_existing_module_succeeds(self):
        mod = optional_import("openlithohub.plugins")
        assert hasattr(mod, "LithoPlugin")


# ---------------------------------------------------------------------------
# OptionalPluginError message
# ---------------------------------------------------------------------------

class TestOptionalPluginError:
    def test_message_includes_pip_install_command(self):
        err = OptionalPluginError("diffnano")
        assert "pip install openlithohub[diffnano]" in str(err)

    def test_unknown_plugin_generic_message(self):
        err = OptionalPluginError("totally_unknown")
        assert "not installed" in str(err)
        assert "pip install" not in str(err)


# ---------------------------------------------------------------------------
# list_plugins
# ---------------------------------------------------------------------------

class TestListPlugins:
    def test_returns_known_plugins(self):
        plugins = list_plugins()
        assert "diffnano" in plugins
        assert "diffcfd" in plugins

    def test_plugins_available_when_not_installed(self):
        plugins = list_plugins()
        # DiffNano/DiffCFD are not installed in the core-only test env
        assert plugins["diffnano"] in ("available", "installed")
        assert plugins["diffcfd"] in ("available", "installed")


# ---------------------------------------------------------------------------
# SimulatorConfig resist_backend field
# ---------------------------------------------------------------------------

class TestSimulatorConfigResistBackend:
    def test_default_is_ctr(self):
        cfg = SimulatorConfig()
        assert cfg.resist_backend == "ctr"

    def test_can_set_diffnano(self):
        cfg = SimulatorConfig(resist_backend="diffnano")
        assert cfg.resist_backend == "diffnano"


# ---------------------------------------------------------------------------
# Registry plugin backend discovery
# ---------------------------------------------------------------------------

class TestRegistryPluginDiscovery:
    def test_core_backends_always_registered(self):
        names = list_simulators()
        assert "hopkins" in names
        assert "calibre" in names
        assert "tachyon" in names

    def test_plugin_backends_listed_as_available(self):
        backends = list_available_backends()
        names = {b["name"] for b in backends}
        assert "diffnano_rcwa" in names
        assert "diffcfd_litho" in names

    def test_get_simulator_gives_helpful_error_for_plugin(self):
        with pytest.raises(KeyError, match=r"pip install openlithohub\[diffnano\]"):
            get_simulator("diffnano_rcwa")

    def test_get_simulator_diffcfd_loads_when_installed(self):
        # DiffCFD is installed in this environment — should load successfully
        sim = get_simulator("diffcfd_litho")
        assert sim.name == "diffcfd_litho"

    def test_unknown_backend_error(self):
        with pytest.raises(KeyError, match="Unknown simulator"):
            get_simulator("nonexistent_backend")


# ---------------------------------------------------------------------------
# LithoPlugin protocol
# ---------------------------------------------------------------------------

class TestLithoPluginProtocol:
    def test_protocol_is_runtime_checkable(self):
        class MyPlugin:
            name = "test"

            def register(self):
                pass

        assert isinstance(MyPlugin(), LithoPlugin)

    def test_non_conforming_not_instance(self):
        class NotAPlugin:
            pass

        assert not isinstance(NotAPlugin(), LithoPlugin)
