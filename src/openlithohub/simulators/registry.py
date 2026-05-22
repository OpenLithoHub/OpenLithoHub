"""Open registry of simulator backends, keyed by string name."""

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


def register_simulator(name: str, cls: type[BaseSimulator]) -> None:
    """Register a simulator class under ``name``.

    Overwrites any previous registration to keep the API simple — users
    that want defensiveness can guard with ``name in list_simulators()``.
    """

    _REGISTRY[name] = cls


def list_simulators() -> list[str]:
    """Return the names of all registered simulator backends, sorted."""

    return sorted(_REGISTRY)


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
        KeyError: If ``name`` is not registered.
    """

    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown simulator {name!r}; registered: {list_simulators()}") from exc
    return cls(config)
