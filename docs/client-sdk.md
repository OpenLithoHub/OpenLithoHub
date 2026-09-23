# Python Client SDK (PR-F)

`openlithohub.client` is a **thin, typed, synchronous** client for the
OpenLithoHub HTTP API. It owns exactly five things: HTTP transport, typed
error mapping, job polling, atomic artifact download, and streamed
transfer — it never re-implements server logic and never generates a
second API schema.

```bash
pip install "openlithohub[client]"
```

## One contract source

The client re-exports the **same typed models** the server validates with
(`openlithohub.server.schemas`): `JobStatus`, `ErrorCode`,
`JobStatusResponse`, `OptimizeMetadata`, `CapabilitiesResponse`, … Since
PR-D froze those models (`extra="forbid"`) and the OpenAPI snapshot gate
blocks drift, the SDK cannot drift from the server — there is only one
contract authority. This import is hygiene-safe: it needs only pydantic
(a core dependency), never the `[server]` extra, and a hostile subprocess
test blocks `fastapi`/`uvicorn`/`starlette` while importing the client.

## Quickstart

```python
from openlithohub.client import OpenLithoHubClient

with OpenLithoHubClient("http://localhost:8000", api_key=None) as client:
    # synchronous optimize — streamed upload + streamed download
    result = client.optimize(
        "chip.npy",
        model="dummy-identity",
        node="3nm-euv",
        writer="vsb",
        execution_mode="auto",
        destination="out.oas",
    )
    print(result.execution_mode, result.execution_reason, result.tiles)

    # long-running job: submit, poll, download
    job_id = client.create_job("chip.npy", model="dummy-identity", writer="vsb")
    snapshot = client.wait_for_job(job_id, timeout=3600)
    if snapshot.status == "succeeded":
        client.download_artifact(job_id, "out.oas")
    else:
        print(snapshot.status, snapshot.error)
```

## Errors are programmable

Every non-2xx response raises `OpenLithoHubApiError` parsed from the
PR-D envelope. **Key on `error.code`** (the frozen vocabulary in
[`server-api-contract.md`](server-api-contract.md)) — never on message
text:

```python
from openlithohub.client import OpenLithoHubApiError

try:
    client.optimize("chip.npy", model="neural-ilt", destination="out.oas")
except OpenLithoHubApiError as e:
    if e.code == "ADMISSION_FULL":
        ...  # retry later — your choice, the client never retries
    elif e.code == "STREAMING_UNSUPPORTED":
        ...  # large job on an unsupported combination: fail your pipeline
    raise
```

`e.status_code`, `e.request_id` and `e.detail` carry the rest of the
contract. Connection failures (DNS, timeout, reset — failures where the
server never answered) raise `OpenLithoHubConnectionError` so callers can
distinguish "the server said no" from "we could not ask".

## Industrial semantics

* **Large files never fully transit RAM.** Uploads stream from disk via
  multipart; downloads stream in chunks to `<destination>.part` and are
  moved into place with an atomic rename — a crash never leaves a partial
  file at the final path, and `optimize`/`download_artifact` accept paths
  (not bytes) by design.
* **No automatic retries.** One request, one attempt — for the industrial
  default. If you want backoff around idempotent GETs, wrap
  `wait_for_job`/`get_job` yourself; POSTs must not be blindly retried.
* **Polling survives server restarts.** With the server's SQLite job
  backend, `job_id`s stay valid across process restarts and recovery
  (queued jobs run; a job that was mid-execution during the crash
  surfaces as `failed` with the server's restart-interruption error).
  `wait_for_job` against the restarted server simply keeps working.
* **Terminal states are data, not exceptions.** `wait_for_job` returns
  the snapshot for `succeeded`, `failed` *and* `cancelled` — check
  `snapshot.status` / `snapshot.error`. Only client-side problems raise:
  `OpenLithoHubJobTimeoutError` (the job keeps running server-side),
  `OpenLithoHubApiError`, `OpenLithoHubConnectionError`,
  `FileExistsError` (download destination exists without
  `overwrite=True`).

## Scope

Sync only by design this round. The server's long-task surface is already
the async job API; the sync client performs short requests plus local
polling and never asks the server to hold long connections. An async
variant (`AsyncOpenLithoHubClient`) can be added later purely additively —
the contract models, error vocabulary and endpoint surface it needs are
already frozen.
