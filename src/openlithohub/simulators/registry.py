"""Open registry of simulator backends, keyed by string name.

Resist / Lithography Backend Selection Guide
=============================================

This registry dispatches to several fidelity tiers of lithography simulation:

**Lightweight built-in (default)**
    :func:`~openlithohub._utils.resist_model.apply_differentiable_resist`
    provides a fast sigmoid-threshold resist with optional Gaussian acid diffusion.
    This is the default path when ``config.resist_backend == "ctr"``.

**High-fidelity Dill/Mack exposure + development**
    Available via the ``diffcfd_litho`` backend, which uses
    :class:`~openlithohub.plugins.diffcfd_process.DiffCFDLithoSimulator`.
    Requires the ``[diffcfd]`` extra.  Models Beer-Lambert exposure with PAC
    bleaching (Dill) and solvent-dependent dissolution (Mack).

**High-fidelity CAR/PEB resist**
    Available via the ``diffnano`` resist backend, which uses
    :class:`~openlithohub.plugins.diffnano_resist.DiffNanoResistAdapter`.
    Requires the ``[diffnano]`` extra.  Models acid diffusion, PEB diffusion,
    and sigmoid development with calibratable contrast.

**Hopkins forward model**
    The bundled :class:`~openlithohub.simulators.hopkins_sim.HopkinsSimulator`
    provides the core Hopkins/SOCS aerial-image simulation.  DiffNano also ships
    ``diffnano.solvers.litho.HopkinsLithoModel`` (a simplified PSF model that
    is functionally equivalent to the Hopkins path here).  For new work, prefer
    the core ``HopkinsSimulator`` which supports multiple illumination types and
    SOCS kernel caching.

See ``docs/resist_capability_matrix.md`` for the full capability comparison.
"""

from __future__ import annotations

from openlithohub.simulators.base import BaseSimulator, SimulatorConfig
from openlithohub.simulators.calibre import CalibreSimulator
from openlithohub.simulators.hopkins_sim import HopkinsSimulator
from openlithohub.simulators.tachyon import TachyonSimulator

_REGISTRY: dict[str, type[BaseSimulator]] = {
    "hopkins": HopkinsSimulator,
    "calibre": CalibreSimulator,
    "tachyon": TachyonSimulator,
}

# Plugin-provided backends that are available via extras but not yet loaded.
# Maps backend name → (extra, package_to_check, module_path, class_name).
_PLUGIN_BACKENDS: dict[str, tuple[str, str, str, str]] = {
    "diffnano_rcwa": (
        "diffnano",
        "diffnano",
        "openlithohub.plugins.diffnano_em",
        "DiffNanoRCWA",
    ),
    "diffnano_fdtd2d": (
        "diffnano",
        "diffnano",
        "openlithohub.plugins.diffnano_em",
        "DiffNanoFDTD2D",
    ),
    "diffnano_fdfd2d": (
        "diffnano",
        "diffnano",
        "openlithohub.plugins.diffnano_em",
        "DiffNanoFDFD2D",
    ),
    "diffcfd_litho": (
        "diffcfd",
        "diffcfd",
        "openlithohub.plugins.diffcfd_process",
        "DiffCFDLithoSimulator",
    ),
    "diffcfd_spin_coat": (
        "diffcfd",
        "diffcfd",
        "openlithohub.plugins.diffcfd_process",
        "DiffCFDSpinCoatSimulator",
    ),
}


def register_simulator(name: str, cls: type[BaseSimulator]) -> None:
    """Register a simulator class under ``name``.

    Overwrites any previous registration to keep the API simple — users
    that want defensiveness can guard with ``name in list_simulators()``.
    """

    _REGISTRY[name] = cls


def _try_load_plugin_backend(name: str) -> bool:
    """Attempt to lazy-load a plugin backend into the registry.

    Checks that the actual plugin package is importable before loading
    the adapter. Returns ``True`` if the backend was loaded and registered.
    """
    entry = _PLUGIN_BACKENDS.get(name)
    if entry is None:
        return False

    extra, pkg, module_path, class_name = entry
    try:
        import importlib

        importlib.import_module(pkg)
    except ImportError:
        return False

    try:
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        _REGISTRY[name] = cls
        return True
    except Exception:
        return False


def list_simulators() -> list[str]:
    """Return the names of all registered simulator backends, sorted."""

    return sorted(_REGISTRY)


def list_available_backends() -> list[dict[str, str]]:
    """Return info dicts for plugin backends not yet loaded.

    Each dict has keys ``name``, ``extra``, ``status`` (``"loaded"`` or
    ``"available"``).
    """
    from openlithohub.plugins import list_plugins

    list_plugins()  # ensure plugin status is computed
    result: list[dict[str, str]] = []
    for name, (extra, _pkg, _mod, _cls) in sorted(_PLUGIN_BACKENDS.items()):
        status = "loaded" if name in _REGISTRY else "available"
        result.append({"name": name, "extra": extra, "status": status})
    return result


def describe_simulators() -> list[tuple[str, type[BaseSimulator]]]:
    """Return ``(name, class)`` pairs for every registered simulator, sorted by name.

    Used by the CLI to print human-readable backend listings without
    reaching into ``_REGISTRY`` from the outside.
    """

    return sorted(_REGISTRY.items())


def get_simulator(
    name: str,
    config: SimulatorConfig | None = None,
) -> BaseSimulator:
    """Construct a simulator by name.

    Args:
        name: Registered backend name (e.g. ``"hopkins"``).
        config: Optional :class:`SimulatorConfig`.

    Raises:
        KeyError: If ``name`` is not registered and not a known plugin backend.
    """

    # Try plugin lazy-load first
    if name not in _REGISTRY and name in _PLUGIN_BACKENDS:
        extra = _PLUGIN_BACKENDS[name][0]
        if _try_load_plugin_backend(name):
            cls = _REGISTRY[name]
            return cls(config)
        raise KeyError(
            f"Backend {name!r} requires the [{extra}] extra.  "
            f"Install with:  pip install openlithohub[{extra}]"
        )

    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown simulator {name!r}; registered: {list_simulators()}") from exc
    return cls(config)
