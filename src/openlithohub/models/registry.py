"""Model registry — discover and instantiate lithography models."""

from __future__ import annotations

from typing import Any

from openlithohub.models.base import LithographyModel


class ModelRegistry:
    """Registry for discovering and instantiating lithography models."""

    def __init__(self) -> None:
        self._models: dict[str, type[LithographyModel]] = {}

    def register(self, model_cls: type[LithographyModel]) -> type[LithographyModel]:
        """Register a model class. Can be used as a decorator.

        The model class must define ``NAME`` as a class-level attribute. The
        registry reads it without instantiating, so anything set in
        ``__init__`` is invisible here.
        """
        name = getattr(model_cls, "NAME", None)
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"Model {model_cls.__name__} must define a class-level "
                f"`NAME: ClassVar[str]` attribute to be registered."
            )
        self._models[name] = model_cls
        return model_cls

    def get(self, name: str, **kwargs: Any) -> LithographyModel:
        """Instantiate a registered model by name."""
        if name not in self._models:
            available = ", ".join(sorted(self._models.keys()))
            raise KeyError(f"Model '{name}' not found. Available: [{available}]")
        return self._models[name](**kwargs)

    def list_models(self) -> list[str]:
        """Return names of all registered models."""
        return sorted(self._models.keys())


registry = ModelRegistry()
