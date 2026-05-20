"""HTTP micro-service interface for OpenLithoHub.

Exposes the optimization engine over a small FastAPI surface so that
fab-side schedulers (Slurm, LSF) and legacy C++/Perl pipelines can
invoke the Python engine via plain `curl` instead of embedding the
Python interpreter.

The engine stays resident: each model is loaded on first use and
cached per-process, so repeat requests skip the weight-load cost.
"""

from __future__ import annotations

from openlithohub.server.app import create_app

__all__ = ["create_app"]
