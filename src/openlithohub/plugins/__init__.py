"""Optional physics plugin infrastructure for OpenLithoHub.

Provides:
- :func:`optional_import` — lazy-import with actionable error messages
- :class:`LithoPlugin` — protocol all physics plugins must satisfy
- Plugin auto-registration helpers

Plugins are **opt-in**: ``pip install openlithohub[diffnano]`` or
``pip install openlithohub[diffcfd]``.  Core tests must pass with **zero**
optional plugins installed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "LithoPlugin",
    "OptionalPluginError",
    "PluginManifest",
    "optional_import",
    "register_plugin",
    "list_plugins",
]

# ---------------------------------------------------------------------------
# Plugin manifest — describes an available-but-maybe-not-installed plugin
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PluginManifest:
    """Static metadata about a physics plugin.

    Attributes:
        name: Short identifier (e.g. ``"diffnano"``).
        extra: ``pip install`` extra that provides this plugin.
        description: One-line human-readable summary.
        modules: Top-level Python modules the plugin ships.
        simulators: Simulator backend names this plugin can register.
    """

    name: str
    extra: str
    description: str
    modules: tuple[str, ...]
    simulators: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Known plugins — static registry of what *can* be installed
# ---------------------------------------------------------------------------

_KNOWN_PLUGINS: dict[str, PluginManifest] = {
    "diffnano": PluginManifest(
        name="diffnano",
        extra="diffnano",
        description="Differentiable nanophotonics: high-precision resist, RCWA/FDTD EM solvers",
        modules=("diffnano",),
        simulators=("diffnano_rcwa", "diffnano_fdtd2d", "diffnano_fdfd2d"),
    ),
    "diffcfd": PluginManifest(
        name="diffcfd",
        extra="diffcfd",
        description="Differentiable CFD: spin coating, Dill/Mack lithography, joint optimization",
        modules=("diffcfd",),
        simulators=("diffcfd_litho", "diffcfd_spin_coat"),
    ),
}


# ---------------------------------------------------------------------------
# Error type — actionable message for missing optional deps
# ---------------------------------------------------------------------------

class OptionalPluginError(ImportError):
    """Raised when an optional plugin is requested but not installed.

    The string representation includes the ``pip install`` command needed
    to resolve the issue.
    """

    def __init__(self, plugin_name: str) -> None:
        manifest = _KNOWN_PLUGINS.get(plugin_name)
        if manifest is not None:
            cmd = f"pip install openlithohub[{manifest.extra}]"
            msg = (
                f"Optional plugin {plugin_name!r} is not installed.  "
                f"Install it with:  {cmd}"
            )
        else:
            msg = f"Optional plugin {plugin_name!r} is not installed."
        super().__init__(msg)
        self.plugin_name = plugin_name


# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------

def optional_import(
    module_name: str,
    *,
    plugin: str | None = None,
) -> Any:
    """Import *module_name*, raising :class:`OptionalPluginError` on failure.

    Parameters
    ----------
    module_name:
        Dotted Python module path (e.g. ``"diffnano.solvers.resist"``).
    plugin:
        Plugin identifier to produce a helpful error message.  When *None*,
        the raw ``ImportError`` is re-raised unchanged.

    Returns
    -------
    module
        The imported module object.

    Raises
    ------
    OptionalPluginError
        When the import fails and *plugin* maps to a known manifest.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        if plugin is not None:
            raise OptionalPluginError(plugin) from exc
        raise


# ---------------------------------------------------------------------------
# Plugin protocol — what every plugin must implement
# ---------------------------------------------------------------------------

@runtime_checkable
class LithoPlugin(Protocol):
    """Protocol that physics plugins satisfy.

    A plugin is any object with a ``register()`` method that, when called,
    registers its simulator backends with the core registry.  This keeps
    the plugin system decoupled — the core never imports plugin code
    directly.
    """

    name: str

    def register(self) -> None:
        """Register simulator backends with :mod:`openlithohub.simulators.registry`."""
        ...


# ---------------------------------------------------------------------------
# Runtime plugin tracking
# ---------------------------------------------------------------------------

_LOADED_PLUGINS: dict[str, LithoPlugin] = {}


def register_plugin(name: str) -> None:
    """Attempt to discover and register a plugin by name.

    If the plugin's module is not installed the call is a silent no-op
    (the plugin is simply unavailable).  This is safe to call from
    top-level module code — it will never raise.
    """
    manifest = _KNOWN_PLUGINS.get(name)
    if manifest is None:
        return

    for mod_name in manifest.modules:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            return

    # Module loaded — look for a ``_openlithohub_plugin`` entry point object
    # that satisfies the LithoPlugin protocol.
    for mod_name in manifest.modules:
        mod = importlib.import_module(mod_name)
        plugin_obj = getattr(mod, "_openlithohub_plugin", None)
        if plugin_obj is not None and isinstance(plugin_obj, LithoPlugin):
            plugin_obj.register()
            _LOADED_PLUGINS[name] = plugin_obj
            return


def list_plugins() -> dict[str, str]:
    """Return ``{name: status}`` for all known plugins.

    Status is one of: ``"installed"``, ``"available"``, ``"unknown"``.
    """
    result: dict[str, str] = {}
    for name, manifest in _KNOWN_PLUGINS.items():
        try:
            for mod_name in manifest.modules:
                importlib.import_module(mod_name)
            result[name] = "installed"
        except ImportError:
            result[name] = "available"
    return result
