"""Version management for OpenLithoHub."""

from __future__ import annotations

try:
    from importlib.metadata import version

    __version__ = version("openlithohub")
except Exception:
    __version__ = "0.0.0.dev0"
