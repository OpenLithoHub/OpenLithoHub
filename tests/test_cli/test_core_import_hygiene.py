"""PR-A tail B: the core CLI must be usable without the ``[server]`` extra.

Import-hygiene contract established after the remote package-smoke/Docker
CLI smoke regressions:

    core CLI import must not require [server] extras
    FastAPI is required only when server functionality is actually invoked

The subprocess below installs a meta-path finder that blocks
``fastapi``/``uvicorn``/``starlette`` so the guarantee is verified
deterministically even in environments where those packages exist.
"""

from __future__ import annotations

import subprocess
import sys

_HYGIENE_SCRIPT = """
import importlib.abc
import sys


class _NoServerExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("fastapi", "uvicorn", "starlette"):
            raise ImportError(
                f"server extra {fullname!r} is not installed (test blocker)"
            )
        return None


sys.meta_path.insert(0, _NoServerExtras())

# 1. core package import PASS
import openlithohub  # noqa: E402

# 2. the server config module is FastAPI-free
import openlithohub.server.config  # noqa: E402, F401

# 3. the whole CLI surface imports without server extras
from openlithohub.cli.app import app  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

runner = CliRunner()
version = runner.invoke(app, ["--version"])
assert version.exit_code == 0, version.output
help_text = runner.invoke(app, ["--help"])
assert help_text.exit_code == 0, help_text.output

# 4. `serve` without the extras fails closed with a clear message
serve = runner.invoke(app, ["serve", "--port", "0"])
assert serve.exit_code == 1, serve.output
assert "server extras" in serve.output, serve.output

# 5. the FastAPI app surface itself requires the extra
try:
    from openlithohub.server import create_app  # noqa: F401
except ImportError as exc:
    assert "fastapi" in str(exc), exc
else:
    raise AssertionError("create_app imported without fastapi")

print("HYGIENE-OK")
"""


def test_core_cli_imports_without_server_extras() -> None:
    proc = subprocess.run(  # noqa: S603 - fixed argv, test-only blocker script
        [sys.executable, "-c", _HYGIENE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "HYGIENE-OK" in proc.stdout
