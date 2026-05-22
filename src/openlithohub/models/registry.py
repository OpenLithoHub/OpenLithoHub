"""Model registry — discover and instantiate lithography models."""

from __future__ import annotations

import inspect
from typing import Any

from openlithohub.models.base import LithographyModel


class ModelRegistry:
    """Registry for discovering and instantiating lithography models."""

    def __init__(self) -> None:
        self._models: dict[str, type[LithographyModel]] = {}

    def register(self, model_cls: type[LithographyModel]) -> type[LithographyModel]:
        """Register a model class. Can be used as a decorator.

        The model class must define ``NAME`` directly on itself (not inherit
        it). The registry reads ``vars(model_cls)`` so that a default ``NAME``
        on a future base class cannot cause every concrete subclass that
        forgets to override it to silently collide on the same key.
        """
        name = vars(model_cls).get("NAME")
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"Model {model_cls.__name__} must define a class-level "
                f"`NAME: ClassVar[str]` attribute on itself (not inherited) "
                f"to be registered."
            )
        self._models[name] = model_cls
        return model_cls

    def get(self, name: str, **kwargs: Any) -> LithographyModel:
        """Instantiate a registered model by name.

        Kwargs that the target model's ``__init__`` does not accept are
        silently dropped, so optional CLI flags like ``--pretrained`` work
        across the whole registry without each call site needing to know
        which models support which options. Real bugs in the model's
        ``__init__`` (mistyped args, missing required positionals) still
        propagate as ``TypeError``.
        """
        if name not in self._models:
            available = ", ".join(sorted(self._models.keys()))
            raise KeyError(f"Model '{name}' not found. Available: [{available}]")
        cls = self._models[name]
        return cls(**_filter_supported_kwargs(cls, kwargs))

    def supports_kwargs(self, name: str, kwargs: dict[str, Any]) -> dict[str, bool]:
        """Return a per-key flag indicating whether the named model accepts each kwarg."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found.")
        cls = self._models[name]
        accepted = _accepted_kwargs(cls)
        if accepted is None:
            return {k: True for k in kwargs}
        return {k: k in accepted for k in kwargs}

    def list_models(self) -> list[str]:
        """Return names of all registered models."""
        return sorted(self._models.keys())


def _accepted_kwargs(cls: type[LithographyModel]) -> set[str] | None:
    """Names accepted by ``cls.__init__``. Returns None if it accepts ``**kwargs``."""
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return None
    names: set[str] = set()
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.add(param.name)
    return names


def _filter_supported_kwargs(cls: type[LithographyModel], kwargs: dict[str, Any]) -> dict[str, Any]:
    accepted = _accepted_kwargs(cls)
    if accepted is None:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in accepted}


registry = ModelRegistry()


def register_builtin_models() -> None:
    """Side-effect import the in-tree models so the registry is populated.

    Idempotent — Python caches modules in ``sys.modules`` so repeated calls
    are cheap. Both the optimize CLI and the multiprocessing workers call
    this so workers populate their registry the same way the parent does.
    """
    import openlithohub.models.examples.dummy_model  # noqa: F401
    import openlithohub.models.gan_opc  # noqa: F401
    import openlithohub.models.levelset_ilt  # noqa: F401
    import openlithohub.models.neural_ilt  # noqa: F401
    import openlithohub.models.openilt  # noqa: F401
    import openlithohub.models.rule_based_opc  # noqa: F401
