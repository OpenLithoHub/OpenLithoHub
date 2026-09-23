# Server API Contract (v1)

This document is the compatibility authority for the OpenLithoHub HTTP
surface (`/v1/*`). It froze at **PR-D**; the machine-level authorities are:

* **OpenAPI document** — `docs/api/openapi-v1.json`, checked in and drift-gated
  by the `api-contract` CI job (`scripts/check_openapi_snapshot.py`).
* **Typed models** — `src/openlithohub/server/schemas.py`
  (`extra="forbid"`: unknown fields are a contract error).
* **Schema version** — the single constant `API_SCHEMA_VERSION = "1"`.
* **Error codes** — the finite `ErrorCode` registry in the same module.
* **Job states** — the `JobStatus` enum and `JOB_TRANSITIONS` table.

## Scope and honesty

The v1 contract describes a **process-local, non-durable, single-worker**
service:

* default backend `in-memory`: jobs live in memory; **restarting the
  process loses them**;
* optional backend `sqlite` (`OPENLITHOHUB_JOB_BACKEND=sqlite` +
  `OPENLITHOHUB_STATE_DIR`): committed jobs, durable inputs/artifacts and
  the queued workload survive restarts; a job that was RUNNING during a
  crash fails closed (outcome unknown, never retried automatically);
* job state is not shared across worker processes — run **one** uvicorn
  worker process (the SQLite store additionally takes an exclusive
  per-directory ownership lock, so a second live process fails fast);
* `/v1/metrics` is process-local and non-durable by definition;
* artifacts live under retention policy (TTL + history cap); actively
  downloading artifacts are protected by a process-local lease.

Multi-worker shared state remains future work and will arrive as additive
changes, not silent reinterpretations. The `jobs` block of
`/v1/capabilities` reports the backend truthfully
(`durable` / `restart_loses_jobs`).

## Response envelope

Every JSON response is validated against a typed model. Error responses
carry the transitional envelope (the legacy top-level `detail` is kept
verbatim for existing clients; new clients should key on `error.code`):

```json
{
  "api_schema_version": "1",
  "detail": "job queue is full",
  "error": {
    "code": "QUEUE_FULL",
    "message": "job queue is full",
    "request_id": "0f1e2d3c4b5a"
  }
}
```

## Error codes

One code per failure class — never key on message text:

| Code | HTTP | Meaning |
| --- | --- | --- |
| `INVALID_REQUEST` | 400 | Bad user input (bad form field, unknown node, oversized/malformed request). |
| `UNKNOWN_MODEL` | 404 | Model name not in the registry. |
| `UNKNOWN_JOB` | 404 | Job id not in the (process-local) store. |
| `UPLOAD_TOO_LARGE` | 413 | Upload exceeded `OPENLITHOHUB_MAX_UPLOAD_BYTES`. |
| `QUEUE_FULL` | 429 | Bounded job queue has no free slot. |
| `ADMISSION_FULL` | 429 | Max concurrent optimize capacity exhausted. |
| `SERVER_NOT_ACCEPTING` | 503 | Runtime draining/stopped or not started. |
| `JOB_ARTIFACT_UNAVAILABLE` | 409 | Job has no downloadable artifact (not succeeded). |
| `JOB_RUNNING` | 409 | Deletion targets a running job. |
| `STREAMING_UNSUPPORTED` | 400 | Streaming requested on an unsupported combination (PR-C fail-closed). |
| `MISSING_OPTIONAL_DEPENDENCY` | 503 | Optional extra not installed (e.g. klayout). |
| `OPTIMIZATION_FAILED` | — | Reserved for explicit runner failure classification. |
| `INTERNAL_ERROR` | 500 | Unexpected failure; no internals are leaked. |

Note: malformed form bodies answer **400 INVALID_REQUEST** (the FastAPI
default 422 is mapped at PR-D — an intentional, reviewed status change).

## Job lifecycle

```
QUEUED -> RUNNING -> SUCCEEDED
                 \-> FAILED
QUEUED -> CANCELLED
```

* `RUNNING` means **admitted execution started** (`started_utc`); a job
  parked waiting for admission is still `QUEUED`.
* `RUNNING -> CANCELLED` is deliberately absent: the runtime drains
  already-admitted work and cannot cancel an executing model.
* Public timestamps: `created_utc` (accepted/enqueued), `started_utc`
  (worker begins admitted execution), `completed_utc` (terminal
  transition). Monotonic clocks remain internal (TTL/eviction only).
* Public job snapshots never contain scratch paths, private output paths,
  monotonic timestamps or queue internals. The optimize summary is a typed
  projection (`OptimizeMetadata`) of the PR-C planner vocabulary
  (`execution_mode`, `execution_reason`, `input_backend`,
  `output_backend`, threshold, bounded work counters).

## Request correlation

Every response carries `X-Request-ID`. A client-supplied ID is preserved
verbatim if it matches `^[A-Za-z0-9._-]{1,64}$`; anything else is replaced
by a generated bounded ID. The same value appears in `error.request_id`
and in the structured access log.

## Sync optimize metadata headers

`POST /v1/optimize` returns binary content, so run metadata is contractual
headers: `X-Request-ID`, `X-OLH-Tiles`, `X-OLH-Halo-Px`,
`X-OLH-Export-Format`, `X-OLH-Shape`, `X-OLH-Execution-Mode`,
`X-OLH-Execution-Reason`, `X-OLH-Input-Backend`, `X-OLH-Output-Backend`.

## Metrics

`GET /v1/metrics` renders the process-local registry plus
`ServerRuntime.snapshot()` (the only source of queue/admission state).
Metric series are keyed **only** on bounded dimensions — HTTP status
class, `execution_mode`, `execution_reason`, input/output backend — with a
hard cardinality cap and an `_other` bucket. Request/job IDs, filenames
and paths appear in structured events (logs), never as metric labels.
Durations are count/sum only — no percentile claims.

## Compatibility policy

PATCH-compatible (no review, snapshot regeneration required):

* additive **optional** response fields;
* additive endpoints.

REVIEW REQUIRED (update this doc + snapshot in the same PR, and call it
out in the PR description):

* removing/renaming a response field;
* changing a field type or requiredness;
* changing an HTTP status or the error-code mapping;
* changing `JobStatus` values or transition semantics;
* changing an `error.code` value;
* changing header names/semantics.

There is no semver tooling yet — the OpenAPI snapshot diff **is** the
review gate.
