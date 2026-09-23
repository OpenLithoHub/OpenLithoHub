"""PR-F import-hygiene gate: the client must never bind the ``[server]``
extra.

``import openlithohub.client`` (and its re-exports of the shared contract
from ``openlithohub.server.schemas``) executes inside a subprocess whose
meta-path blocks ``fastapi``/``uvicorn``/``starlette`` — the exact hygiene
contract PR-A tail B established for the core CLI, now extended to the
client surface. If anyone makes the client (or ``server.schemas``)
accidentally import the server stack, this fails deterministically even
in environments where those packages are installed.
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

# 1. the client surface imports without server extras
from openlithohub.client import (  # noqa: E402
    API_SCHEMA_VERSION,
    ErrorCode,
    JobStatus,
    JobStatusResponse,
    OptimizeResult,
    OpenLithoHubApiError,
    OpenLithoHubClient,
)

# 2. the shared contract vocabulary is the server's own (one schema source)
from openlithohub.server.schemas import ErrorCode as ServerErrorCode  # noqa: E402
from openlithohub.server.schemas import JobStatus as ServerJobStatus  # noqa: E402

assert ErrorCode is ServerErrorCode
assert JobStatus is ServerJobStatus
assert API_SCHEMA_VERSION == "1"

# 3. typed parsing works client-side without the server stack
snapshot = JobStatusResponse.model_validate(
    {
        "api_schema_version": "1",
        "job_id": "job-1-abc",
        "status": "succeeded",
        "created_utc": "now",
        "started_utc": "now+1",
        "completed_utc": "now+2",
        "summary": {
            "shape": [4, 4],
            "tiles": 1,
            "halo_px": 0,
            "writer": "vsb",
            "export_format": "oasis",
            "execution_mode": "dense",
            "execution_reason": "DENSE_REQUESTED_EXPLICITLY",
            "input_backend": "dense-raster",
            "output_backend": "dense-oasis",
            "threshold": 0.5,
        },
        "error": None,
    }
)
assert snapshot.summary is not None
assert snapshot.summary.execution_reason == "DENSE_REQUESTED_EXPLICITLY"

error = OpenLithoHubApiError(
    status_code=429,
    code="ADMISSION_FULL",
    message="server at maximum concurrent optimizations",
    request_id="req-1",
)
assert error.code == "ADMISSION_FULL" and error.status_code == 429

assert "fastapi" not in sys.modules
assert "uvicorn" not in sys.modules
assert "starlette" not in sys.modules
print("client import hygiene OK")
"""


def test_client_imports_without_server_extras() -> None:
    result = subprocess.run(  # noqa: S603 — fixed argv, repo-local interpreter
        [sys.executable, "-c", _HYGIENE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "client import hygiene OK" in result.stdout
