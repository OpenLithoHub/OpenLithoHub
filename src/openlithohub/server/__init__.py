"""HTTP micro-service interface for OpenLithoHub.

Exposes the optimization engine over a small FastAPI surface so that
fab-side schedulers (Slurm, LSF) and legacy C++/Perl pipelines can
invoke the Python engine via plain `curl` instead of embedding the
Python interpreter.

The engine stays resident: each model is loaded on first use and
cached per-process, so repeat requests skip the weight-load cost.

Import hygiene (PR-A tail B): importing ``openlithohub.server`` (or its
FastAPI-free submodules such as ``openlithohub.server.config``) must
NOT require the optional ``[server]`` extra — the core CLI depends on
this package path. ``create_app`` is therefore exported lazily and
imports FastAPI only on first access.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from openlithohub.server.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
