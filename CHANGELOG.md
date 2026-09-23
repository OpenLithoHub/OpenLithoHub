# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **PR-G (phase 1) — bottleneck-driven throughput engineering** (exact-vector spatial index, execution-device authority, bounded GPU tile batching, engineering stage timing) — implemented in the roadmap's revised priority order: fix the measured CPU geometry bottleneck FIRST, then device placement, then batching; formal Industrial Benchmark v2 measurement is explicitly deferred until the engineering surface freezes (§26) and runs on its own authority afterwards. **G1** (perf/vector): `ExactVectorRunSource` gains an immutable 256-row **band index** over vector-geometry bboxes (`candidates_for_row` / `query_bbox` / `index_stats`) — first-touch row lookup no longer scans every flattened polygon per unseen row (the recorded dominant bottleneck); the index is candidate filtering ONLY (exact rational scanlines, hole subtraction, union, contributor identity, run ids and tile clipping are untouched), its memory is bounded by geometry (a 16k² layout with one polygon holds ≤2 index entries — it can never become a raster in disguise), it is immutable after construction (safe concurrent reads), and 9 hostile tests (G1-A…I) prove indexed/reference EXACT identity of owned runs, run ids, contributor ids, window rasters and verification ownership metadata across holes, overlaps, hierarchy/instances, band-straddling geometry, empty windows and overlapping tile reads. **G2** (feat/device): one server-owned execution-device policy — `OPENLITHOHUB_DEVICE=auto|cpu|cuda|cuda:N` (auto = CUDA when available else CPU; explicit CUDA on an unusable host FAILS CLOSED, never silent CPU fallback); `LithographyModel` gains `SUPPORTED_DEVICES` (default cpu-only; importing Torch is not GPU safety) and the server refuses a device/model mismatch; the selected device enters resident model-cache identity (distinct cache entries per device — never one cached model moved CPU↔GPU per request); `/v1/capabilities` now reports `gpu.selected_device` and `gpu.product_execution_enabled` truthfully — CUDA availability alone no longer implies product GPU execution. **G3** (feat/streaming): `SUPPORTS_BATCHED_PREDICT` capability (default False) plus bounded same-shape tile micro-batching in `run_streaming` (`batch_size`/`batched_forward_fn`) — up to N same-shape windows per forward, flushed on shape change (boundary tiles never padded), each tile still individually sliced/committed with its own terminal ownership disposition; batching is REFUSED with verification plugins or screening attached (verifier re-entrancy would reorder execution), `batch_size=1` keeps the pre-PR-G path bit-identical, forward invocations are surfaced as `StreamingRunReport.forward_batches` (strictly fewer than tiles when batched), and CUDA OOM propagates — no silent CPU fallback. **G0** (docs/obs): engineering-only stage timing (`stage_timing` on the internal result: index build, forward wall, write wall, end-to-end — diagnostics, not benchmark authority), precise multi-GPU fallback semantics in docs (workers pin `cuda:i` only when ≥ N devices are visible, otherwise all fall back to CPU dispatch), and explicit host-RAM-not-VRAM wording for `SharedStateDictServer`. OpenAPI unchanged; Industrial Benchmark v1.1 and P-054/B04 frozen semantics untouched; multi-GPU (G4) and pinned-transfer/prefetch remain gated on post-G1 profiling per the roadmap. Industrial Benchmark v2 authority (schema/identity/Tier A–C harness) lands as the follow-up once this engineering surface freezes.
- **PR-F — Python Client SDK** (`src/openlithohub/client/`, `[client]` extra) — a thin, typed, **synchronous** client for the frozen v1 API, designed against the PR-D contract freeze and PR-E durability semantics rather than server internals. **One contract source**: the client re-exports and validates against the very models behind the server's `response_model` declarations and the OpenAPI snapshot (`openlithohub.server.schemas` — pydantic is a core dependency and the module is import-hygiene-safe, so no second schema, no generated client code, zero drift; a hostile subprocess test blocks `fastapi`/`uvicorn`/`starlette` while importing the client, extending the PR-A hygiene contract to the SDK surface). Responsibilities are exactly five: HTTP transport (`httpx` via the new `openlithohub[client]` extra, transport injection for tests), typed error mapping (every non-2xx → `OpenLithoHubApiError` keyed on the frozen `error.code` with status/request_id/detail; connection failures separated as `OpenLithoHubConnectionError`), job polling (`create_job`/`get_job`/`wait_for_job` with per-poll callback and explicit `OpenLithoHubJobTimeoutError` that leaves the job running server-side), atomic artifact download (chunked stream → `.part` → rename, `overwrite` guard), and streamed multipart upload — large layouts never fully transit RAM on either side. Industrial semantics: **no automatic retries** (one request, one attempt; callers choose backoff), terminal job states return as data (`succeeded`/`failed`/`cancelled` snapshots — job failure is a result, not a client exception), and polling simply keeps working against a restarted SQLite-backend server (submit → restart → poll → download covered end-to-end). Per-request `X-Request-ID` generation correlates client logs with the server envelope. Async (`AsyncOpenLithoHubClient`) intentionally deferred — purely additive later. Docs: `docs/client-sdk.md`.
- **PR-E — Durable Job Store / Restart Recovery + bounded positioning tail** (`server/job_store.py`, `server/runtime.py`, `server/config.py`, `server/app.py`) — durability substitution under the frozen PR-D contract: **a service restart now changes storage ownership, not the public API**. New single `JobStore` authority (`server/job_store.py`) with two backends behind one protocol: `InMemoryJobStore` preserves the lightweight mode verbatim, and `SQLiteJobStore` (stdlib sqlite3, WAL, busy_timeout, parameterized SQL, schema-version table that fails startup closed on unknown newer databases) makes committed QUEUED rows themselves the durable queue — `queue.Queue` no longer owns anything durable, and the in-memory wake signal is a pure optimization (a lost wake can never lose a committed job; the worker polls). `OPENLITHOHUB_JOB_BACKEND=sqlite` + `OPENLITHOHUB_STATE_DIR` persist job metadata, finite versioned-JSON execution params (never pickle; malformed persisted params fail the job closed, never misexecute), durable inputs (`tmp write -> fsync -> atomic rename -> QUEUED commit -> 202` — no state where the client saw 202 without a durable job) and artifacts (`artifact.tmp.<ext> -> fsync -> atomic rename -> SUCCEEDED commit` with SHA-256 + byte size recorded — a partial output can never be served as success). Hostile crash/restart matrix E1–E12 plus dual-backend API parity and OpenAPI-invariance tests: queued survives restart; stale RUNNING recovers fail-closed to FAILED with an explicit restart-interruption error and NO automatic retry; SUCCEEDED artifacts remain byte-identical and downloadable after restart; persisted rows count toward queue depth so uploads cannot overbook; upload crashes leave no ghost job and no consumed capacity (orphan staging/jobs collected at startup); deletion stays deleted; TTL/history retention moves into the store, never touches RUNNING rows, and skips actively downloaded artifacts (process-local download lease registered BEFORE the sweep protects the FileResponse stream). `ServerRuntime.snapshot()` remains the observability authority (job counters derived from the store; admission/reservations stay runtime-owned); `/v1/metrics` unchanged. Capabilities report truthfully: sqlite → `durable=true, restart_loses_jobs=false`; both backends stay `single_process_only=true` — and a second live process pointing at the same state directory fails fast on an exclusive ownership lock instead of racing rows. No OpenAPI change (snapshot gate green). Positioning tail (bounded): README/README_zh tighten over-broad memory slogans to the scoped, artifact-backed claim (256× layout growth at ~0.35–0.42 GiB streaming RSS; explicit not-a-universal-guarantee boundary), add a self-evidence "Why OpenLithoHub" section and the explicit P-054 academic lineage with separated software/P-054 citation responsibilities; `CITATION.cff` retitled to the canonical vendor-neutral platform positioning with P-054 as a related DOI (separate citable object); no competitor ranking, no new benchmark numbers, Industrial Benchmark v1.1 and P-054 frozen authorities untouched.
- **PR-D — Stable API / Job Contract + Observability** (`server/schemas.py`, `server/errors.py`, `server/observability.py`, `server/app.py`, `server/runtime.py`) — contract-freeze PR over the PR-C execution spine. `GET /v1/health|ready|version|capabilities|metrics` and the job endpoints now carry real `response_model` declarations backed by Pydantic v2 models with `extra="forbid"` (field drift fails tests, never leaks into OpenAPI); one `API_SCHEMA_VERSION` constant owns the schema version. Every error is rendered as a machine-readable envelope (`error.code` + `error.request_id`, legacy `detail` preserved) from ONE central exception mapping (`runtime_error_code`): unknown model → 404 `UNKNOWN_MODEL`, unknown job → 404 `UNKNOWN_JOB`, queue/admission full → 429, draining → 503 `SERVER_NOT_ACCEPTING`, artifact/running → 409, streaming-unsupported → 400 `STREAMING_UNSUPPORTED`, malformed input → 400 `INVALID_REQUEST` (intentional review-gated change from FastAPI's default 422), unexpected failures → fixed-text 500 `INTERNAL_ERROR` with the traceback kept server-side. `JobStatus` is a public str-enum with a frozen transition table (`QUEUED→RUNNING/CANCELLED`, `RUNNING→SUCCEEDED/FAILED`; `RUNNING→CANCELLED` deliberately absent because the runtime does not own cancellation of executing models) enforced by one checked `transition_job_status` helper — the worker now flips QUEUED→RUNNING only after admission is won (a dequeued-but-unadmitted job stays QUEUED and cancels legally), public job snapshots gain `started_utc`/`completed_utc` with stable semantics and are projected through `OptimizeMetadata` so scratch paths, private output paths and the internal work ledger can never leak (a contract-violating runner summary degrades to `summary=null` with a loud log, not a 500). Request correlation: client `X-Request-ID` preserved iff `^[A-Za-z0-9._-]{1,64}$`, else replaced by a bounded generated ID; the same value spans response header, `error.request_id` and the access log. Sync optimize metadata headers are contractual and tested, adding `X-OLH-Input-Backend`/`X-OLH-Output-Backend`. Observability is a provider-neutral, process-local registry (`/v1/metrics` = registry + `ServerRuntime.snapshot()`, no second source of truth) with hard cardinality caps and an `_other` bucket — identifiers/filenames/paths are never metric dimensions — plus a frozen structured-event vocabulary (`http_request_completed`, `optimize_started/completed/failed`, `job_*`, `runtime_*`) that makes the PR-C planner externally inspectable (mode/reason/backends/tile counts on every run). A deterministic checked-in OpenAPI v1 snapshot (`docs/api/openapi-v1.json`, normalized sorted-keys JSON) is drift-gated by a dedicated fast `api-contract` CI job that also pins the error-code registry, JobStatus vocabulary and transition table; `docs/server-api-contract.md` states the PATCH vs REVIEW-REQUIRED compatibility policy and the non-durable/process-local limitations. No SQLite/Redis/Prometheus/OTel — those stay with later PRs. Industrial Benchmark v1.1 and P-054/B04 frozen authorities untouched.
- **PR-C — Production Streaming Execution Gate** (`src/openlithohub/workflow/execution.py`, `streaming/sinks.py`, all four product surfaces) — the product optimize path now has a single execution spine: one planner (`plan_execution`) decides **dense vs streaming** with typed reasons (`DENSE_SMALL_LAYOUT`, `STREAMING_REQUIRED_BY_MEMORY_POLICY`, `STREAMING_UNSUPPORTED_{INPUT,OUTPUT,MODEL}`, …) and every surface — CLI `optimize run`, `LitheEngine.optimize`, sync `POST /v1/optimize` and the async job API — dispatches through `optimize_layout` instead of four private pipelines. `auto` is fail-honest: under the new dense memory policy (`OPENLITHOHUB_MAX_DENSE_BYTES`, default 8 GiB per fp32 raster, `0` = unlimited) small jobs stay byte-compatible dense (tile/blend/stitch); above the policy supported combinations stream via the existing `run_streaming` authority — memmap `.npy` input (header-only probing), exact vector GDS/OASIS input (`KLayoutAlignedRunSource`, parser-only, requires integer DBU-per-pixel pixel alignment) — and **unsupported combinations fail closed (HTTP 400) instead of silently materializing a full-chip raster**. Gate C1: streaming `.npy` output is a self-describing memmap-backed raster artifact (`MemmapTileSink(npy=True)`) with no full input/output tensor, no `tile_results` list and no `stitch_tiles` call on the branch (enforced by structural hostile tests: poisoned dense loader, poisoned stitch, weakref retention firewall, weakref live-window memory gate — an engineering regression check, not a benchmark; Industrial Benchmark v1.1 and P-054 untouched). Gate C2 (bounded): streaming Manhattan mask-writer output via `StreamingManhattanTileSink` — each trusted core is run-length encoded and inserted into a KLayout cell exactly once, O(core + rectangles) memory, occupancy round-trip verified against the raster artifact; streaming + curvilinear (mbmw) is explicitly unsupported for large jobs rather than faked. Dense/streaming equivalence is pinned bitwise for deterministic pointwise models (service- and engine-level, `.npy` inputs; the GDS PIL-vs-exact-scanline boundary-convention difference is documented, not hidden). `GET /v1/capabilities` now reports a truthful `streaming` matrix (engine/product_execution, per-input/output support, memory policy) populated from code-owned tables and cross-checked against real probe behaviour; the response and job summaries carry the execution decision (`X-OLH-Execution-Mode` / `X-OLH-Execution-Reason` headers, `execution_reason` / `input_backend` / `output_backend` fields). `LithographyModel` gains `SUPPORTS_STREAMING` (default True) so a model that cannot honour windowed prediction opts out honestly. Docs: new `docs/streaming-execution.md`.
- **Fourth-pass fixes** (audit 2026-09-21 fourth pass) — closes the remaining pre-rerun blockers: `publish_family` no longer JSON-parses the distribution-freeze text blob (JSON members and blob members are handled separately; the freeze is byte-hashed against `environment_lock.distribution_freeze_sha256` — the earlier "complete family" drill had silently exercised the pre-freeze publication path); PEP 610 provenance uses the REAL keys (`vcs_info.commit_id`, `dir_info.editable`) via a tested pure `format_direct_reference` helper; `verify_source_closure` takes a role-keyed mapping so failure messages never crash on `Path` attributes; SHA256SUMS is parsed/validated in ONE pass with duplicate-filename, absolute-path and `..`-traversal rejection; the authority root rejects unknown JSON files and unknown schemas fail-closed; `--models` is now a real configuration knob (canonicalized, registry-validated, deduplicated and bound into the run identity AND stage execution — unknown/duplicate/empty models hard-fail); verifier and claims paths are anchored to the repository root, not the caller's cwd; the production drill test runs REAL `publish_family` → PRODUCTION verifier → PRODUCTION claims `--check` with zero problem filtering plus freeze-byte/run-config-arg/artifact-byte/unknown-root mutation failures; `write_build_info.py` replaces the invalid here-doc quoting in the publish workflow with a py_compile-checked generator and the release wheel smoke now asserts the baked `BUILD_COMMIT == GITHUB_SHA`; `/v1/version` imports `openlithohub._build` explicitly; Docker passes `VERSION`/`BUILD_COMMIT`/`BUILD_RUN_ID` build-args and smokes package version + baked identity; server hardening: exception-safe `reserve_job_slot` context manager, artifact GET snapshot under lock with TTL refresh, eviction scratch cleanup moved outside the job lock, worker lifespan ownership with graceful shutdown drain; `all` extra split from dev tooling (`dev-all` added). Deferred: P3.3 legal wording (counsel), P4.2 image tiering, P5 service maturity, diff-surrogate PyPI publication.
- **Third-pass authority-composition fixes** (audit 2026-09-21 third pass, P0.1–P0.10 + P1.1–P1.6 + P2/P3/P4) — the verifier parses family roles by SCHEMA (`OpenLithoHub.industrial-run-config.v1` gets its own `validate_run_config`; benchmark artifacts get `validate_artifact`), uses the industrial module's single `validate_artifact_family`, RECOMPUTES the run identity from the published run-config (content-addressed identity — a modified config can no longer keep the old identity label), independently recomputes `environment_lock.lock_sha256` over the lock body, ties every artifact's `git_commit` to its `measurement_source.commit`, closes manifest membership/SHA/byte-size exactly over the canonical family, verifies the published distribution-freeze bytes against the lock, and dropped the duplicated parse loop. The freeze now carries PEP 610 direct-reference provenance (diff-surrogate frozen at its exact git URL@commit; editables carry their source path) and is PUBLISHED as `industrial-distribution-freeze.txt` inside the family. Publication is **manifest-last commit-marker** semantics (per-file replacement is explicitly NOT cross-file atomic; readers must validate the manifest/SHA family), and the run workspace enforces the EXACT `industrial-*` member set at promotion. Three family-closure tests that were accidentally nested after a `return` (never collected by pytest) are restored as a real class with a collection guard; new end-to-end tests run the PRODUCTION verifier and claims `--check` against a synthetic real-schema family. Server: worker startup is idempotent, queue enqueue uses `put_nowait` with a pre-upload slot reservation (a full queue answers 429 before ingesting the body), the job worker holds admission capacity (queued jobs wait instead of failing on transient contention), job reads return snapshots under the lock, GETs drive TTL eviction, readiness exposes queue metrics. Release: runtime SBOM generated against a freeze of the runtime venv with the generator outside it (P2.1); build identity (commit/timestamp/run id) baked into the wheel via `_build.py` and surfaced in `/v1/version` (P2.2); Docker CLI image no longer carries build wheels (P4.1); capabilities split runtime proof availability from repository governance provenance (P4.3); `requirements-minimum.txt` deleted (pyproject is the single dependency source, P3.1). Deferred: P3.3 legal wording review (counsel), P4.2 image tiering and P5 service maturity (post-rerun).
- **Second-pass pre-rerun blockers closed** (audit 2026-09-21 second pass, B0.1–B0.10 + C1–C7 + E1–E3) — closes the hostile-review findings on the authority layer: artifact publication is now **family-atomic** (stage artifacts write ONLY into `runs/<identity>/`; `publish_family` requires the exact {runtime, quality, fulldie, run-config} set, validates every member plus cross-artifact closure (same run identity / commit / fixture hash / environment lock), then atomically promotes the complete family with `manifest.json` + `SHA256SUMS.txt` to the public root — a mixed or partial family can never appear there); the distribution freeze is computed first and its hash is INSIDE `lock_sha256` (persisted as `runs/<identity>/distribution-freeze.txt`, no repo-root temp file); the preflight script uses THE HARNESS'S OWN parser and `compute_run_identity_from_args` and prints the exact `RUN_IDENTITY` the formal run will use; `run_identity` + `environment_lock.lock_sha256` are mandatory schema fields with cross-artifact family closure in the verifier; `SHA256SUMS.txt` is set-complete (every artifact indexed, no stale/foreign entries); `claims --check` is a TRUE drift gate (regenerates the claims JSON/MD deterministically in memory and byte-compares with the checked-in files); `requirements.lock` honestly renamed to `requirements-minimum.txt` and synced to the commit pin; the dataset provenance file renamed to `DATASET_MANIFEST.SHA256` so the pre-existing full-data `MANIFEST.SHA256` integrity contract is no longer shadowed; the public family includes `industrial-run-config.json`; 42 hostile authority tests (every-arg identity mutation, source/fixture/env mutation, wrong checkpoint identity/stage/NaN hard-fails, stale fixture refusal, mixed-run family rejection, SHA-set incompleteness, preflight≡harness identity). Correctness sweep: NOTICE contradiction removed + diff-surrogate moved to core deps; Docker `OLH_SCRATCH_DIR` actually used via `scratch_root()` for requests/jobs/readiness; the job API is a real bounded queue (fixed worker, locked transitions, `429` on full queue, scratch cleanup on failure/eviction, TTL); die-grid derivation is per-axis for rectangular dies; duplicated verifier loop removed; SECURITY.md orphan sentence and two Chinese-README regressions fixed. Release governance: publish workflow verifies the tag commit is reachable from protected `main`, runs wheel smoke on the release bytes, generates a RUNTIME CycloneDX SBOM from a clean venv of the wheel, and grants OIDC `id-token: write` to the publish job only; `.github/CODEOWNERS` now also protects `.github/CODEOWNERS` itself and the `ci.yml`/`publish.yml`/`docker.yml` trust boundaries (server-side ruleset additions — making `package-smoke`/`perf-sentinel` required and one-approval enforcement — are maintainer settings actions).
- **Pre-rerun hardening — run identity, environment lock, fail-closed fixture/layout validation, preflight gate** (audit 2026-09-21, Phase A) — before any formal long measurement: `compute_run_identity` binds ONE SHA-256 to the measurement source commit + harness/core/generator/support file hashes, the environment lock (package versions, torch build/threads, thread env vars, CPU/RAM), parent + ICCAD fixture hashes and EVERY semantic CLI argument; all mutable run state lives under `benchmarks/results/industrial/runs/<run_identity>/` (`run-config.json`, `fixtures/`, `checkpoints/`, `progress.json`, `RUN.log`) so a different identity can never touch another run's state; checkpoints are strict JSON with per-row `{_schema, _run_identity, _stage, _key}` identity — mismatched rows hard-fail instead of silently reusing stale results; derived fixtures carry full identity records (parent/derived SHA-256, crop bbox, generator commit/harness hash) and are re-verified before ANY reuse; parent-layout validation is fail-closed (expected top cell — never `top_cells()[0]` — exact 1 nm/DBU arithmetic, layer present AND non-empty); die-grid `tiles_per_side` is derived from the die bbox with any override recorded and coverage-validated; `scripts/preflight_industrial_benchmark.py` prints `PRE-RERUN PREFLIGHT: PASS` only when every gate closes.
- **Verifier semantic closure + CI platform gates** (Phase A) — `scripts/verify_industrial_artifacts.py` now re-hashes the recorded source files AS COMMITTED (`git show <commit>:<path>`) and cross-checks every claim's artifact hash against both the file bytes and `SHA256SUMS.txt`; `ci.yml`'s lint job checks out full history and runs the verifier plus the claims drift gate on every pull request; `docs.yml` builds docs strict and runs the link lint on pull requests too (previously main-push only). All GitHub Actions across every workflow are pinned to full commit SHAs with version comments.
- **Server operations surface** (Phase D, lean implementation) — global admission semaphore (`OPENLITHOHUB_MAX_CONCURRENT_OPTIMIZE`, excess → `429` + `Retry-After`); long-task job API (`POST /v1/jobs/optimize`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/artifact`, `DELETE /v1/jobs/{id}`) so minute-scale OPC/ILT never holds an HTTP connection; `/v1/ready` readiness (models registered, scratch writable); optional `OPENLITHOHUB_API_KEY` boundary (all endpoints except `/v1/health`); every response carries `X-Request-ID` + duration with structured access logs; `/v1/capabilities` reports `api_schema_version`/`capability_schema_version`, input/export formats and structured proof-verification info; `ModelRegistry.get` gains `ignore_unsupported=False` strict mode (server uses it — config typos raise instead of silently dropping kwargs).
- **Supply chain / packaging / image hygiene** (Phase C) — diff-surrogate pinned to the exact v0.3.0 commit (`6b0ec109…`) with NOTICE entry; NOTICE corrected to state the Docker image DOES redistribute the GPL-3.0 KLayout wheel (the previous "not redistributed" claim was inaccurate while the image installed it); Dockerfile pins the `python:3.12-slim` base by digest, runs as a non-root `openlithohub` user, and adds a `server` target (local-wheel `[server]` install + HTTP healthcheck) published alongside the CLI image; Docker CI now also triggers on `src/**`/`NOTICE` changes; new `package-smoke` CI job builds sdist+wheel, `twine check`s them and smoke-installs the wheel in a clean venv; `publish.yml` adds tag/version consistency, `twine check` and a CycloneDX SBOM artifact; Dependabot covers pip + github-actions + docker; weekly `pip-audit` of the optional extras; a 1–2 minute `perf-sentinel` CI job catches only catastrophic performance regressions (no benchmark claims).
- **Product / docs consistency** (Phase C) — canonical "vendor-neutral computational lithography platform" wording unified across `pyproject.toml`, package docstring, `CITATION.cff`, `docs/index.md` and both READMEs; `fab-ready`/`MRC-compliant` marketing phrasing replaced with `fab-oriented`/`mask-writer-oriented` plus an explicit "not foundry sign-off" disclaimer at every mention; `COMMERCIAL-USE.md` adds "license permission is not technical qualification"; `SECURITY.md` moves to release-based supported versions and stops fully out-scoping exploitable dependency exposures; new `docs/api-stability.md` documents the per-namespace stability contract (public / versioned / provisional / internal) and the README marks `_utils` as internal; dataset audit logs redact secret-bearing kwargs and every download now writes a standardized `DATASET_MANIFEST.json` + `MANIFEST.SHA256` next to the data; community leaderboard validation no longer requires the `submission` label (issue #29 — path-triggered strict validation with the same base-ref-only safety boundary).
- **Industrial Benchmark v1.1 contract hardening — audit-driven artifact authority** (follow-up to the v1 entry below; audit 2026-09-21) — closes the benchmark-authority gaps found by independent review: artifacts are STRICT JSON (`NaN`/`Infinity` rejected at parse time and at write time via `allow_nan=False`; undefined metrics serialize as `null` with an explicit `non_finite_reason`); every artifact carries a `measurement_source` block (full 40-hex commit of a CLEAN checkout — `working_tree_dirty=true` artifacts are refused at write time and by CI — plus SHA-256 of the harness, core module and claim generator) and a `fixture` block (SHA-256, byte size, top cell, layer, die bbox of the measured GDS); honest status taxonomy (`NOT_RUN_MEMORY_POLICY` for policy-blocked dense rows, `INFEASIBLE_ON_REFERENCE_MACHINE` for the machine-relative full-die raster — a policy decision is never laundered into "structurally impossible"); memory units fixed to GiB (2^30) with the full-die size shown as both 1.23 TB and 1.12 TiB; quality comparisons downgraded to scoped facts with absolute + relative deltas until a spatially expanded paired-CI gate exists (a degenerate blank mask can never headline, and its "100% MRC reduction" is withheld); the surrogate quality gate now fails closed on degenerate/non-finite metrics on either side (`QUALITY_COMPARISON_INVALID`); README promises tightened to Industrial-Benchmark-v1 headlines only; CI `lint` job runs `scripts/verify_industrial_artifacts.py` (strict JSON + schema + SHA256SUMS + provenance closure) and `generate_industrial_claims.py --check` on every PR — artifact drift, byte tampering, or injected `Infinity` fail CI. Harness is fully observable/pausable/resumable: per-row fsynced checkpoints, atomic `progress.json` (stage, done/total, in-flight row, ETA), `RUN.log` tee, resume-skip logging, and an orphan-worker watchdog (workers exit if the driver dies).
- **Server cache concurrency + streamed responses** (`src/openlithohub/server/app.py`) — closes audit P1 items on the resident-model cache: refcount acquisition now happens INSIDE the cache-lock critical section as model insertion (closing the race where a concurrent eviction saw refcount 0 and tore down a model about to be used); slow `teardown()` moved outside the cache lock; sidecar `_MODEL_LOCKS`/`_MODEL_REFCOUNTS` state is bounded under distinct-key churn (regression-tested at 200 keys, cap 2); `/v1/optimize` streams the optimized file from disk via `FileResponse` (a multi-GB output no longer transits through RAM a second time) with the per-request scratch dir removed by a background task after the response completes.
- **Industrial Benchmark v1 — artifact-backed industrial claims on real routed silicon** (`benchmarks/industrial/run_industrial_benchmark.py`, `src/openlithohub/benchmark/industrial.py`, `scripts/generate_industrial_claims.py`, artifacts `benchmarks/results/industrial/`, docs `docs/industrial-benchmarks.md` + `docs/generated/industrial-claims.{json,md}`) — a full-chip-class measurement layer that turns existing capabilities into auditable industrial numbers, with a hard rule: **every public number is generated from a checked-in `OpenLithoHub.industrial-benchmark.v1` artifact; no hand-written performance claims**. Protocol: dense full-raster vs tiled-raster vs exact-vector streaming vs certified selective screening on deterministic center crops (4096²–32768² px @ 1 nm/px, plus a streaming-only 65536² row) of the public OpenROAD-routed **Ibex RISC-V core** (sky130hd, PDB Physical Design Database @ `9e1e3399`, fixture SHA-256 recorded); fresh worker process per timed run (per-run peak RSS), ≥ 5 repeats with median/p10/p90 on claim-bearing rows, dense guarded by an explicit memory policy (`INFEASIBLE_STRUCTURAL` instead of silent skip), 512² synthetic + 1024² real-layout correctness witnesses gating the whole run, work-accounting closure checks, per-stage checkpoint/resume. Measured (Apple M5 Pro CPU, 48 GB): streaming peak RSS stays flat **0.35 → 0.40 GB from 4096² to 65536²** while dense grows 1.00 → 19.67 GB (**65.0–98.0% peak-memory reduction**, `IB-MEM-*`); **65536² px (4.29 GPx) streamed end-to-end at 0.38 GB** where dense input alone is 16 GiB (`IB-SCALE-65536`); full-die dense raster **1.23 TB = structurally infeasible** (`IB-DIE-1`); honestly reported against ourselves: dense remains *faster* on CPU at every feasible size (0.22–0.44× streaming/dense median wall ratio, `IB-RT-*`), `levelset-ilt` defaults degenerate to a blank mask on the print-critical ICCAD16 EUV crop (degenerate-output firewall blocks the trivial "100% MRC reduction"), and surrogate-ILT is not faster end-to-end at the matched iteration budget. Quality stage: same-Hopkins-optics comparison of `dummy-identity` / `rule-based-opc` / `levelset-ilt` / `surrogate-ilt` on real sky130hd tiles (1024² @ 1 nm/px, 193 nm NA 1.35) and on ICCAD16 testcase1 (256² clip @ 4 nm/px, 13.5 nm NA 0.33) — EPE, wafer EPE, L2, PV Band, MRC, shot count with degenerate-blank detection and a wafer-EPE-vs-MRC trade-off gate. Claims pipeline: `scripts/generate_industrial_claims.py` derives claim levels (`REPRODUCED_INTERNAL` … `FOUNDRY_CALIBRATED`) with headline admission thresholds (runtime ≥ 1.1×, memory ≥ 20%, quality ≥ 5%, surrogate gated on a 10% quality tolerance) and `--check` fails CI when README-quoted numbers drift from artifacts; README/README_zh first screen rebuilt around vendor-neutral platform positioning, the measured headline table, and an explicit NOT-claimed section (no foundry qualification, no commercial-tool comparisons, no GPU numbers — `docs/self_hosted_deployment.md` GPU tables downgraded to historical/illustrative with a provenance notice).

- **Server service-discovery endpoints** (`src/openlithohub/server/app.py`) — `GET /v1/version` (package/git/torch identity) and `GET /v1/capabilities` (registered models, simulator backends, GPU availability, streaming-pipeline flag) with host-sensitive details excluded; `flow run` gains opt-in observability (`--profile` per-phase table, `--report-json` machine-readable report with a performance-profile section: parse/tile wall, forward wall, metrics wall, peak RSS, forward calls). Pre-existing stale server tests (`test_get_or_load_model_*`) updated for the current 3-tuple `(model, lock, key)` contract.

- **P-054 repository integration — frozen proof surface** (audit PR-1…PR-5) — engineering-enforced provenance for the frozen *Certified Stratified Phase Diagrams for Finite Hopkins Lithography* (DOI 10.5281/zenodo.22843141, implementation basis `348fa5d`): `CertificationCapability` + `proof_level_ceiling` firewall (a `DIAGNOSTIC_ONLY` backend can never support an `INTERVAL_CERTIFIED` PASS — assembler hard-errors instead of downgrading); `ModelIdentity` replaces the global pinned commit (new certificates name their own frozen model; legacy replay keeps `348fa5d` and says so); `SourceSnapshot v2` with exact dyadic raw-weight provenance, `(sy, sx)` source bins, a full normalization record and declared `SpectralRepresentation` (truncated without certified bound → inadmissible; v1 `is_topk_truncated` honestly answers `None`); `proof_artifacts/p054/` canonical entry (manifest/fixture/event/chamber catalogs with the three distinct layers z1–z5, zV, zH, zP and the `1 → 3 → 5 → 4` target sequence; large artifacts stay external, DOI + SHA-256 with verify-before-use); typed frozen phase-diagram query API (`load_frozen_phase_diagram`, layer-typed event certificates, fail-closed `PhaseArtifactNotAvailableError` for numerics that live in the external artifact); `verification-status.json` + `scripts/check_verification_status.py` (stale-doc and rewritten-basis detection) + `scripts/fetch_proof_artifacts.py`; two-job `p054-proof-replay` workflow (offline fast gate vs scheduled full replay where missing artifacts are hard failures); README honesty boundary updated (public-layout engineering benchmarks exist; nothing is wafer qualification without external calibration).
- **B04 Increment 29 — realistic-density routed-block external validity** (`streaming/crop_source.py`) — `ExactVectorCropSource` exposes a rectangular region of interest of a parent exact-run source as its own finite local-coordinate layout (run/window queries translated parent↔local, SHA-256 crop layout hash, `CropProvenance`, parent `verification_metadata` nesting) without materializing the parent raster. Benchmark: `benchmarks/benchmark_b04_inc29_real_density_ibex.py` — attacks Increment 28's empty-context screening on the pinned public PDB-Physical-Design-Database Sky130HD **Ibex** routed GDS (OpenROAD-generated, 15515 standard cells, 0.579 utilization) via deterministic nested center crops at 4096²/8192²/16384² plus a 32768² screen-only diagnostic, reporting real-density active/screened fractions, parser cost, common-size wall-time comparisons and an explicit empty-screening verdict (`STRONG`/`MODERATE`/`WEAK` — a weak verdict is a valid outcome that redirects the next work-avoidance mainline to nonempty-tile screening). Claim firewall: real routed block crops, not the entire Ibex core; synthetic finite-support forward model; no foundry calibration; speedup claims only for returned common-size comparisons.

- **B04 Increment 28 — scale-first work avoidance** (`streaming/screening.py`, integrated `streaming/work_accounting.py`) — fail-closed pre-forward tile screening: a `TileScreeningPolicy` runs before `TileSource.read_window` and the forward model and may skip a tile only with a certified `SCREENED_OUT` decision carrying an exact trusted-core fill value; when verification plugins are attached, the screen must explicitly certify those verifiers (`certifies_verifiers` + `verification_upper_bound`) or the pipeline falls back to the ordinary active path. The built-in `ExactEmptyContextScreeningPolicy` screens an exact-vector tile only when the complete core+halo read window contains no physical runs and the caller certifies both the context/halo and the constant zero-context response. `WorkAccounting` is now driven by the actual `run_streaming` loop: unique active core area (refinement re-runs charge `reused_work_units`, never inflating active area), uniquely-owned screened-out cores, read-window calls/pixels, forward calls/input pixels, screen queries, plus `read_amplification` and `accounted_pct` closure checks; `verify_layout()` exposes `VerificationResult.work_accounting` and records the screening policy in `model_provenance`. Benchmark: `benchmarks/benchmark_b04_inc28_scale_first.py` (512²–16384² synthetic sparse ladder + real-Sky130-motif sparse hierarchical ladder; `dense_full`/`tiled_raster`/`b04_vector`/`b04_selective` modes with wall time, peak RSS, active/screened fractions, forward-call counts and measured crossover fields; explicit claim firewall — architecture benchmark, not calibrated lithography physics). Spec and notes: `docs/notes/B04_Increment28_ScaleBenchmarkSpec.json`, `docs/notes/B04_Increment28_ScaleFirst_WorkAvoidance.md`.
- **B04 Increment 27 — continuous square-aperture lift + Arb fixed-source pupil** (`verify/continuous_square_mask.py`, `verify/arb_fixed_source_pupil.py`) — promotes the theorem mask to the continuous square-pixel aperture with the exact lift `M̂_c(f) = p² sinc_π(p·fₓ) sinc_π(p·f_y) M_point(f)` (`representation_bridge_upper = 0` for the declared `FINITE_EXACT_VECTOR_SQUARE_APERTURE_MASK` semantics; `ContinuousSquareMaskContract` rejects periodic copies, omitted exteriors and dense loaders), and certifies the one-source-point coherent field `E_s(X,z) = ∫_{|u|≤f_p} M̂_c(u−s) e^{πizλ|u|²} e^{2πi(u−s)·X} du` as a rigorous complex ball via python-flint `acb.integral` after mapping the hard pupil to the fixed rectangle ρ∈[0,1], θ∈[0,2π] (Arb normalized `sinc_pi` handles the removable axis zeros without interval division through zero). Independent proof backend — not yet wired into the public verification API. Benchmark: `benchmarks/benchmark_b04_inc27_fixed_source_arb_pupil.py` (generated fixture + sparse public sky130 layer with an Arb complexity guard). Still open: source-plane quadrature, partial-coherence intensity enclosure, continuous-HOPKINS vs SOCS runtime bridge.
- **B04 Increment 26 — global finite spectral sufficient statistic** (`verify/global_finite_spectrum.py`) — streams the complete finite exact-vector layout into arbitrary-physical-frequency Fourier moments `M_q = Σ exp(-2πi(fx_q(x+½)p + fy_q(y+½)p))` via closed geometric-sum per horizontal run; `spatial_halo_error_upper = 0` for the declared finite plane-wave quadrature model (no periodic copies, no omitted exterior, no dense loader); continuous value/gradient/Hessian jets from the same summary; `GlobalFiniteSpectralContract` rejects omitted exteriors, periodic copies, and dense loaders. Benchmark: `benchmarks/benchmark_b04_inc26_global_finite_spectrum.py` (48 non-lattice disk nodes inside NA/λ on public sky130 GDS).
- **B04 Increment 25 — exact-vector proof-facing cutover** (`verify/exact_vector_optics.py`) — freezes the theorem-facing mask semantics as `EXACT_VECTOR_PIXEL_CENTER_INDICATOR` via an `ExactVectorMaskContract` that forbids the legacy dense loader (hard `ValueError` on `dense_loader_used=True`) and requires `representation_bridge_upper == 0`. `exact_vector_normalized_spectrum` builds normalized DFT coefficients directly from canonical source runs (no dense mask materialized); `materialized_reference_spectrum` is a diagnostic-only algebraic oracle. Benchmark: `benchmarks/benchmark_b04_inc25_public_exact_vector_hopkins.py` — pinned sky130 GDS → `KLayoutAlignedRunSource` → exact-vector run spectrum → K24 Hopkins aerial, bypassing `load_layout()` entirely. The legacy dense loader remains supported for public/legacy benchmarking and diagnostics but is no longer an admissible theorem dependency.
- **B04 Increment 23 — raster-bridge isolation** (`verify/raster_bridge.py`) — characterizes the 90032-vs-96258 dense/exact discrepancy observed in the Increment-21 benchmark: on the regenerated 512² fixture the dense loader's entire 6226-pixel symmetric difference is false positives (zero false negatives), the mutual Chebyshev dilation radius is 1 px, and the Euclidean center Hausdorff upper bound is 11.31 nm at 8 nm pixels. The module explicitly does **not** assert loader/equivalence and is not by itself an EPE/contour/Hopkins/process-window certificate (`DIAGNOSTIC/FINITE-GRID ENCLOSURE` firewall). Benchmark: `benchmarks/benchmark_b04_inc23_raster_bridge.py` (fixture regeneration, exact-oracle guard, bridge characterization).
- **B04 Increment 22 — expert semantic hardening** (`streaming/physical_identity.py`, `verify/replay_contract.py`, KLayout semantics extended tests, public real-GDS benchmark) — physical-instance identity keys separate source repetition (distinct cell/array members are distinct owners even with identical transformed geometry) from read repetition (the same owner seen through two tile windows is never double-charged); `KLayoutAlignedRunSource.from_file` now stamps every run with `physical:<sha256>` owner ids derived from instance paths + shape fingerprints + quantization policy; `run_streaming` sources expose optional `verification_metadata(read_bbox, tile_id, core_bbox)` consumed as tile context. `verify/replay_contract.py` adds end-to-end replay provenance (toolchain/halo-decision/layout/quantization records) and the one-way reference-enclosure acceptance criterion. New public real-GDS benchmark (`benchmark_b04_inc22_public_gds.py`, sky130hd fixture from a pinned ORFS commit) with an explicit claim firewall: parser/ownership results only — no speedup or certified-process-window claims.
- **B04 Increment 21 — vector run pipeline (KLayout-gated)** (`streaming/vector_runs.py`, `verify/run_ownership.py`, `verify/run_spectrum_boundary.py`) — physical vector cells (`VectorCell`, KLayout y-up DBU alignment via `rectangle`), `ExactVectorRunSource` (read-window exact scanline oracle) and `KLayoutAlignedRunSource` (tile-ownership charge de-duplication across overlapping reads); `run_ownership` separates load-bearing vs context run views so overlapping tile reads never double-charge the same physical run; `run_spectrum_boundary` routes owned runs into the Increment 19 spectrum accumulator at domain boundaries. `run_streaming` now annotates tile metadata with `b04_owned_runs` for verifier consumption. KLayout-validated: GDS/OASIS fixtures, hierarchy, polygon holes, finite edges, overlapping reads, ownership de-duplication (`max_owned_duplicate_views = 0` on a 512² layout, 64 tiles). Benchmark: `benchmarks/benchmark_b04_inc21_vector_runs.py` (dense-loader vs exact-scanline timing, `NUMERICAL-DIAGNOSTIC`).
- **B04 Increment 19 — direct streamed run spectrum** (`verify/run_spectrum.py`) — closes the Increment-18 `OPEN_IMPLEMENTATION` blocker: horizontal runs accumulate directly into the normalized layout Fourier spectrum via `RunSpectrumAccumulator` (matches dense FFT to <3e-15), and circular convolution `c_E = N² · c_h · c_m` feeds the existing 144-grid alias-free intensity proof — no corrected coherent spatial grid is ever materialized (`corrected_spatial_grid_materialized = false`, status `PASS_DIRECT_STREAMED_SPECTRAL_CONSTRUCTION`). Frozen fixture: coherent midpoint difference vs Increment 18 ≈4.2e-17, Hessian upper 1.2097e-3 nm⁻², L3 upper 5.009e-5 nm⁻³, local cell gradient floor 5.864e-3 nm⁻¹, hidden-loop scale 19.93 nm. Remaining full-chip blocker moves to the input adapter: KLayout/GDS/OASIS geometry must emit canonical horizontal runs before full-canvas rasterization (5 new tests).
- **B04 Increments 16–18 — known-layout run compression, interval derivative run-prefix, and cellwise Fourier envelopes** (`verify/interface_runs.py`, `verify/derivative_runs.py`, `verify/streaming.py`, `verify/cellwise_fourier.py`) — Increment 16: exact known-layout row-run encoding with cyclic row-prefix oracle (`CyclicRowPrefix.sum_runs` matches per-pixel sums to <1e-15) and run-compression stats (36× compression on the frozen fixture); Increment 17: outward-rounded interval derivative run-prefix artifacts for all six intensity jet components (value/gradient/Hessian), with the center-jet certificate PASS at 8 px halo while the cell extension stays OPEN; Increment 18: layout-conditioned alias-free Fourier envelopes — Hessian Frobenius majorant ≈1.211e-3 nm⁻², third-derivative majorant ≈5.109e-5 nm⁻³ (repairs the generic kernel-L1 L3 failure by >100×), recovering continuous transversality (gradient floor >5.82e-3 nm⁻¹) and hidden-loop completeness (curvature scale >19.55 nm vs 11.31 nm diameter) on the pinned 72×72 fixture. Ledger honestly keeps `production_streamed_construction_without_full_grid = OPEN_IMPLEMENTATION`. Large snapshot NPZs are git-ignored release artifacts addressed by SHA-256 in `README_B04_PATCH.md`; artifact-dependent tests skip when unmounted (14 new tests).
- **B04 Increment 15 — certified-halo architecture** (`verify/halo.py`, `verify/full_chip.py`) — theorem-facing halo contracts deliberately separate from `workflow.halo`: sufficient absolute-tail upper bound `U(h) = Σ w_j (2 A_j T_j(h) + T_j(h)^2)` and necessary unrestricted-binary lower bound `L(h) = max_j w_j (T_j(h)/π)^2` over a frozen normalized SOCS kernel family; `minimal_halo_bracket` classifies the epsilon bracket (`CERTIFIED_SUFFICIENT` / `PARTIAL` / `OPEN`); `resolve_final_halo` keeps physics/model/verifier halo requirements separate; `CoreHaloGeometry` is an explicit theorem geometry (never inferred from workflow overlap); `halo_duplicate_compute_ratio` implements the §15 formula. `aggregate_full_chip_status` folds `TileProofStatus` records streaming-style (certified load-bearing FAIL wins, one INCONCLUSIVE blocks PASS, non-load-bearing tiles excluded). On the 72×72 ArF K=24 snapshot, ε=2.916392e-3 yields only the bracket `29 ≤ h* ≤ 36` px — a useful large-layout certified halo for arbitrary binary exteriors is explicitly **not yet obtained**; replayed from the pinned `B04_Increment15_HaloTail_Certificate` artifact (8 new tests).
- **Streaming core/halo tiling architecture** (`openlithohub.streaming`, RFC 0008) — `TileSource` (`TensorTileSource`, `MemmapTensorTileSource` out-of-core, `VectorLayoutTileSource` + `SpatialLayoutIndex` window rasterization), `TileSink` (`TensorTileSink`, `MemmapTileSink`, raster-free `MetricOnlyTileSink`), `TileRequest`/`TileScheduler` with exact-core-cover planning, verifier-driven refinement (`RefinementRequest`, subdivide/increase-halo), `plan_tiling` memory-budget planner, and the `run_streaming` pipeline with `O(tile area + active batch)` peak memory. Halo sizing becomes a policy interface (`HaloPolicy`): `LegacyFixedHaloPolicy`, `PhysicalInteractionHaloPolicy` (RFC 0005 logic), `KernelTailHaloPolicy` (kernel-tail mass criterion, `CERTIFIED_SUFFICIENT` only with a trusted sample), plus `estimate_minimum_halo` (`EMPIRICALLY_STABLE`, explicitly not a certificate) and `HaloStatus` taxonomy. Benchmarks: `benchmarks/benchmark_streaming.py` (whole-raster vs streaming: resident raster, tiles, halo overhead, discrepancy, minimum-halo sweep). Migration notes in `docs/migration-notes/0008-streaming-core-halo.md`.
- **Verification plugin API** (`streaming.verification`) — `VerificationPlugin` protocol (`required_halo/prepare/verify_tile/reduce/finalize` + optional `refine`), `VerificationRegistry` (`register/get/list`, ships a `dummy` verifier), `TileVerificationResult` with B04 coverage-contract fields, `StreamingVerificationReducer` composing the B04 error budget (`E_process + E_spatial + E_halo + E_raster_bridge + E_other`); global PASS is blocked by any FAIL, INCONCLUSIVE tile, `PARTIAL_COVER`, or breached tolerance.
- **B04 source-native verifier foundation** (`verify/source_snapshot.py`, `verify/source_native.py`) — frozen realized discrete-model snapshot (source-bin indices, normalized weights, pupil-support bits, process/grid conventions, mask hash, git commit), exact IEEE-754 → dyadic-rational import with outward rounding, and the opt-in `source_native_full` backend contract (`FieldEnclosure`, `SpatialDerivativeEnclosure`, reference `OutwardRoundedCPUBackend`). No dynamic top-K spectral branch on this path.
- **B04 Increment 14 integration** (`openlithohub.verify`, RFC 0007) — proof-carrying certificate types (`PASS`/`FAIL`/`INCONCLUSIVE` firewall, `ProofLevel` dependency gates), 1-D boundary-root isolation, coverage reducer (`ALL_COMPONENTS_COVERED`/`PARTIAL_COVER`/`INCONCLUSIVE`), expanded-band replay, MVP-1 manifest adapter, certified nominal-contour-reconstruction consumer, and sha256-pinned proof artifacts (Increment 11–14) with replay regression tests. `EXTRACTED_CONTOUR_EPE` requires `ALL_COMPONENTS_COVERED` plus the reconstruction artifact; raster `epe_max_nm` semantics untouched (firewall test).

### Changed

- **R17 streaming correctness migration (B04)** — proof-carrying execution semantics boundary hardening across `streaming/`, intentionally breaking unsound behaviours (each break is covered by a hostile regression test):
  - **Per-verifier reducer sessions (C1a)** — `run_streaming()` no longer folds all verifiers into one shared latest-wins reducer (which let a later verifier's PASS clobber an earlier verifier's FAIL on the same tile). One verifier instance → one `VerifierSession` → one reducer; optional verifier-owned `make_reducer()`; `StreamingVerificationReducer.retire()` for parent removal. `StreamingRunReport` gains canonical `verification_results: Mapping[VerifierIdentity, GlobalVerificationResult]`; the backward `report.verification` facade is None (0 verifiers) / sole result (1) / `MultiVerifierSummary` (n>1), which deliberately carries no generic numeric upper bound.
  - **Typed screen proof facts (C1b)** — the screen is a proof-fact producer, not a verifier authority: the deprecated `TileScreenDecision.certifies_verifiers` blanket flag is never honored (recorded as provenance only). A certified `SCREENED_OUT` adapts to `ExactOutputFact`/`ScreenProofFact` (`EXACT_OUTPUT_SKIP`); a tile skips the expensive path only when no verifiers are attached or every verifier accepts the facts via the new `accept_screen_facts(tile, facts) -> TileVerificationResult | None` capability; any refusal falls back to the ordinary active path.
  - **Sink capability model (C2)** — optional `record_certified_core(tile_id, bbox, *, exact_fill, metadata)` tensor-free certified commit: `MetricOnlyTileSink` never materialises certified cores; tensor/memmap sinks synthesize exact fills and reject verification-only skips (`CertifiedCommitNotRepresentableError`) because verifier PASS does not imply a known output tensor.
  - **Ownership tree + true subdivision (C3a/C3b)** — new `streaming/ownership.py`: `action="subdivide"` now performs a real area-conserving partition (no longer disguised halo growth) with child read windows reconstructed against the GLOBAL domain; children inherit forward history `F(Cj) ← F(P)`; subdivision retires the parent in EVERY verifier session (no automatic parent-PASS→children restriction theorem) and every child must reach a fresh terminal verdict per verifier.
  - **Dual ledgers (C4a/C4b)** — terminal coverage ledger (final-leaf dispositions, fail-closed on gap/overlap/unterminated/double) and ancestor-aware historical work ledger (region-keyed unique-forward union; `unique_forward_pixels` + `pre_forward_avoided_pixels` == full chip) are separate projections. Inc28 keys `active_pixels`/`screened_out_pixels` survive as aliases with identical no-subdivision values (pinned by a C0 freeze test).
  - **Verifier-owned metrics and tolerances (C5)** — `MetricDescriptor(quantity, unit, tolerance)`; `verify_layout()`'s `continuous_epe_upper_nm` / `hausdorff_upper_nm` are typed projections of the unique metric owner (None when unowned, `AmbiguousMetricProjectionError` when disputed); `tolerance_nm` is a documented single-verifier legacy convenience — heterogeneous multi-verifier runs require per-verifier tolerances.

### Fixed
- **KLayout adapter exact-rational DBU snap** (`streaming/vector_runs.py`) — `KLayoutAlignedRunSource.from_file` computed the theorem-facing DBU-per-pixel ratio from `Fraction(str(layout.dbu))` verbatim; foreign GDS files (e.g. the pinned ORFS sky130hd flow GDS) store a units literal whose nearest binary double reads back as `0.0009999999999999998`, making every pixel ratio non-integer and rejecting the file outright. `exact_dbu_nm()` now snaps the short decimal repr to its intended rational (`limit_denominator(10**6)`) before exact arithmetic; regression-tested against the observed artifact.

### Changed
- **Surrogate-ILT model** (`openlithohub.models.surrogate_ilt`) — CNN-accelerated ILT using an on-the-fly trained surrogate forward model. Periodically corrects with the true physics-based forward model, yielding 10–50× speedup during optimisation. Architecture adapted from DiffNano's `NeuralSurrogate`. Supports both Gaussian and Hopkins forward models.
- **VAE-ILT model** (`openlithohub.models.vae_ilt`) — Latent-space ILT via variational autoencoder. Optimises in a compressed latent space for smoother loss landscape and faster convergence. Self-supervised training (no external dataset required). Architecture adapted from DiffNano's `LearnedRepresentation`.
- **diff-surrogate integration** — External `diff_surrogate` library replaces the local convergence fallback for unified surrogate lifecycle management.
- **GAN-OPC v0.4 training pipeline** (`scripts/train_gan_opc.py`) — metric-aligned PVB bandwidth loss (`mean(outer_envelope - inner_envelope)` across 4 dose/focus corners using differentiable sigmoid threshold), UNetV2 (4-level, 7.7M params), memmap dataset cache, gradient accumulation, AMP support, plateau early-stopping with best-checkpoint rollback, BN-drift sliding baseline guard for >50-epoch runs.
- **v0.4 probes** — P3 (PVB bandwidth correctness), P4 (memmap integrity), P5 (AMP numerical consistency), P6 (memory peak + AMP benchmark).
- **v0.4 eval script** (`scripts/_eval_v04_iccad16.py`) — greenlight rules: PVB≤10.9nm AND MRC≤7.5% AND PVBmax≤45nm.

### Changed

- **GAN-OPC PVB loss rewritten** — `_pvb_loss()` (MSE vs design) replaced by `_pvb_bandwidth_loss()` (metric-aligned envelope bandwidth minimisation). The new loss directly minimises `mean(max(resist_corners) - min(resist_corners))` across 4 dose/focus corners, structurally aligned with the eval-time `compute_pvband()` metric.
- **Thread control** — `torch.set_num_threads` and OMP/MKL/OPENBLAS limits auto-detected from `os.cpu_count()` (capped at 12 for AMD 5600G).
- **Model registry** — `register_builtin_models()` now imports surrogate_ilt and vae_ilt.

### Added

- **RDP vertex decimation on OASIS export** — `export_oasis_mbw(vertex_tolerance_nm=...)` runs an iterative anchored Ramer-Douglas-Peucker simplification on each sampled curvilinear polygon. Default `0.0` keeps bit-exact academic behaviour; positive values cut full-chip OASIS data volume (MBMW shot/byte budget) without measurable wafer-image change. Reduction count and ratio logged at INFO.
- **ILT checkpointing** — `LevelSetILTModel.predict(checkpoint_dir=, save_freq=, resume_from=)` periodically `torch.save`s the mask logit, Adam state, and best-loss tracker. Deterministic resume (resume-vs-uninterrupted equality is pinned by test). SLURM preemption / CUDA crash on multi-thousand-iter runs no longer wipes prior progress. Off by default; `save_freq>0` without `checkpoint_dir` raises.
- **SRAF min-area export filter** — `export_oasis_mbw(min_area_nm2=...)` and the matching `workflow.export.{export_oasis,export_gds}` parameter drop sub-resolution polygons via shoelace area before OASIS insert (default `0.0`, Hackathon-safe). Plumbed through `optimize --export-min-area` and the `/v1/optimize` HTTP form so fab-ready exports can clear MRC without touching academic scoring runs. Dropped count logged at INFO.
- **Deterministic mode** — `openlithohub._utils.determinism.set_deterministic()` centralises the four torch backend flags needed for bit-reproducible scoring (`cudnn.deterministic`, `cudnn.benchmark=False`, `allow_tf32=False` on cudnn + matmul). Exposed via `--deterministic` on `openlithohub optimize` and `openlithohub eval`; off by default (the flags carry a real perf cost).
- **HF Hub SHA256 verification** — `ModelHub.download_weights` now verifies an expected `sha256` digest on the HuggingFace Hub path (previously only the direct-URL path did). Mutable revisions (branch names like `main`) trigger a warning so users know the digest can drift between fetches; pin a commit SHA or tag for reproducible scoring. Closes #20.
- **Measured-source / Zernike-pupil I/O on simulators public API** — `load_source_intensity`, `load_zernike_coefficients`, and `zernike_phase_map` are now re-exported from `openlithohub.simulators` (previously defined in `_utils/optics.py` with full test coverage but no `src/` callers). The README and v0.1 milestone advertised this feature; it is now backed by a public import path. Closes #65.
- **Multi-patterning regime guidance** — `levelset-ilt` README and docstring now document that the model targets single-exposure regimes; multi-patterning (LELE / SAQP / SADP) requires upstream colouring + per-mask runs.
- **DRC-vs-MRC count semantics** — eval-aggregation docs and `_repr_html_` panels disambiguate DRC violation count (per-rule hard-fail) vs MRC violation count (geometric width/spacing samples). Same number, different denominator.
- **`openilt` baseline model** (`openlithohub.models.openilt`) — clean-room
  PyTorch reimplementation of the OpenILT SimpleILT formulation
  (MIT-licensed upstream pinned at commit
  [`dabb97c`](https://github.com/OpenOPC/OpenILT/commit/dabb97c6ca3dfd159362e48273c436444c77353b)).
  Optimises the MOSAIC L2 + PVBand objective (Gao et al., DAC 2014) with
  SGD across a 3-corner dose/defocus sweep, distinct from `levelset-ilt`'s
  single-corner Adam loop. Reuses the existing Gaussian / Hopkins forward
  models. Closes #17.
- **L2 wafer-error metric** (`openlithohub.benchmark.metrics.l2_error`) —
  Neural-ILT canonical printability metric: `compute_l2_error()` returns
  `L2ErrorResult` with `l2_error_pixels` and `l2_error_nm2` between the
  forward-simulated wafer image and the target. Complements EPE for
  callers training against the same loss the upstream Neural-ILT paper
  reports.
- **Wafer-level EPE via forward physical simulation** —
  `compute_epe(..., simulate=True)` passes the predicted mask through the
  Hopkins/Gaussian forward model before extracting contours, producing
  the contest-canonical "wafer EPE" rather than the previous mask-vs-mask
  contour distance. Wired through `openlithohub eval run`.
- **GDSII export** (`openlithohub.workflow.export_gds`) — companion to
  `export_oasis`. GDSII is the academic/contest lingua franca (ICCAD,
  SPIE benchmarks). Manhattan masks dump as rectangles; curvilinear
  masks vectorise to polygons via klayout (GDSII has no native curve
  primitive).
- **ICCAD'13 gauge file IO** (`openlithohub.workflow.gauges`) — round-trip
  reader/writer for the contest gauge format alongside the existing
  Calibre `.gg` and CSV parsers, so contest-style `(x, y, angle, target)`
  EPE-evaluation tables drop into the gauge pipeline directly.
- **ONNX-runtime CI smoke test** — extends the existing `openlithohub
  export` ONNX path with a CI smoke test that loads the exported model
  via `onnxruntime` and verifies a single forward pass agrees with the
  PyTorch reference, catching dynamo/onnxscript regressions before they
  reach users.
- **DEF/LEF layout ingestion** — `openlithohub.workflow.parse_layout`
  now accepts `.def` and `.lef` inputs in addition to OASIS/GDSII.
  Pass `lef_files=[...]` to feed cell abstracts when reading a DEF file.
  Closes the gap from RTL-to-GDSII flows (Innovus / ICC2 / OpenROAD)
  that emit DEF as their canonical interchange format.
- **OpenAccess layer-purpose helper** (`openlithohub.workflow.layer_purpose`) —
  canonical purpose-name → integer map mirroring the OpenAccess (Si2)
  default registry plus a permissive `classify_purpose()` alias resolver.
  Lets downstream tooling branch on `(layer, datatype, purpose_name)`
  whether the input came from Cadence (oaPurpose) or OASIS (datatype).
- **Croissant dataset metadata** — dataset adapters expose
  `croissant_name` / `croissant_description` / `croissant_license_url` /
  `croissant_url` / `croissant_citation` properties so OpenLithoHub
  datasets can emit
  [Croissant](https://github.com/mlcommons/croissant) JSON-LD for
  ML-data discoverability.
- **Anamorphic demag flags for High-NA EUV** — `ProcessNode` gains
  `demag_scan` / `demag_slit` (both default to 4.0) and an
  `is_anamorphic` property. ASML's High-NA EXE:5000 class (NA=0.55) is
  8× along scan, 4× along slit; recording demag here unblocks downstream
  anamorphic imaging and reticle-area accounting.
- **imec-style stochastic defect classification**
  (`openlithohub.benchmark.metrics.compute_stochastic_defect_classes`,
  `StochasticDefectRates`) — per-class failure rates in failures/cm²
  for the four canonical EUV stochastic-defect classes (microbridge,
  break, missing contact, merging contact), following the imec
  defectivity-rate convention. Complements the existing aggregate
  `compute_stochastic_robustness`. Includes a shared `_NominalState`
  cache and per-component bridge/break detection for accurate counting
  on tiles with multiple disconnected line segments.
- **Object-oriented API façade** (`openlithohub.api`) — `Mask`,
  `LitheEngine`, and `Report` re-exported at the package root
  (`from openlithohub import Mask, LitheEngine`). Thin wrapper over the
  existing functional API for fab-/EDA-shaped callers who think in masks
  and engines, not tensors and registries. `Mask` is a frozen dataclass
  carrying `(tensor, pixel_size_nm, layer)` with explicit and
  suffix-sniffing constructors (`Mask.from_oasis`, `Mask.from_pt`,
  `Mask.from_npy`, `Mask.from_gds`, `Mask.load`); `LitheEngine` exposes
  `optimize` / `evaluate` / a public `load_layout` plus a teardown
  lifecycle so model resources release cleanly; `Report` aggregates
  metrics, compliance, and tile/halo provenance. The functional API is
  unchanged. Closes #10.
- **Differentiable curvilinear MRC loss** (`openlithohub.benchmark.metrics.curvilinear_mrc_loss`) —
  three-term penalty (min-CD via soft morphological opening, min-spacing
  on the inverted mask, min-curvature via boundary-band gradient
  magnitude) that drops into ILT / level-set / Neural-ILT training loops.
  PDK-first contract: pass `pdk="asap7"` / `pdk="freepdk45"` / a
  `PdkRules`, or supply explicit `min_width_nm` / `min_spacing_nm` /
  `pixel_size_nm`; per-rule kwargs win over the preset. Mirrors the
  binary verdict in `compliance.mrc.check_mrc` so loss and verdict agree
  on what a violation is. Closes #8.
- **SRAF non-printing penalty** (`openlithohub.benchmark.metrics.sraf_print_penalty`) —
  differentiable squared-ReLU loss that punishes SRAF-region aerial
  intensity rising above a configurable `print_threshold - margin`.
  Drop-in for any `torch.optim` ILT loop; complements the post-hoc
  `compliance.mrc` check by catching the failure mode while gradients
  still flow.
- **Process-window-aware OPC workflow** (`openlithohub.workflow.process_window`) —
  `ProcessWindowCorner` dataclass, `DEFAULT_PW_CORNERS` (5-corner
  dose × focus sweep), `pw_aerial_images`, and `pw_fidelity_loss`
  (weighted-MSE across corners). `LevelSetILTModel.predict()` gains
  opt-in `process_window: bool = False` and `pw_corners` kwargs so
  callers can co-optimise against the corner sweep instead of the
  nominal point. Metadata records `process_window` and
  `pw_corner_count`. Defaults stay nominal — no API break.
- **Auto-calibration notebook** (`notebooks/auto_calibration.ipynb`) —
  end-to-end demo of inverting measured-vs-simulated CD error onto
  resist-threshold and Gaussian-σ parameters using `torch.optim.Adam`.
  Runs on CPU in <30 s; pre-fit MAE 1.999 px → post-fit MAE ~0 px on
  the synthetic gauge table.
- **Sharded CI test job** — `.github/workflows/ci.yml` splits the
  pytest run into 5 directory shards (`models`, `workflow`,
  `benchmark`, `data-utils`, `other`) × 3 Python versions = 15
  parallel jobs, each running `pytest -n auto` (workflow shard runs
  serially to avoid spawn-context nesting). Wall-clock dropped from
  30+ min monolithic to ~9 min sharded.
- **`openlithohub serve` HTTP micro-service** (`openlithohub.server`) —
  FastAPI app exposing `GET /v1/health`, `GET /v1/models`, and
  `POST /v1/optimize` so fab-side schedulers (Slurm, LSF) and legacy
  C++/Perl pipelines can drive the optimization engine without
  embedding the Python interpreter. Models are loaded lazily and
  cached in-process; new `[server]` extra pulls in
  `fastapi` / `uvicorn` / `python-multipart`. See the `serve` section
  of the CLI reference for the curl example.
- **Jupyter `_repr_html_` for result dataclasses** —
  `PredictionResult`, `MRCResult`, `CurvilinearMRCResult`, `DRCResult`,
  `MonteCarloFailureResult`, and `SimulatorResult` now render as
  inline HTML panels (pass/fail badge, key/value table, violation
  rows, mask thumbnail) when displayed in Jupyter / Colab / VS Code.
  Helpers live in `openlithohub.jupyter._html` and degrade gracefully
  to plain `repr` when matplotlib is unavailable.
- **RFC 0003 — Standard MRC rule-deck schema**
  (`docs/rfcs/0003-mrc-rule-deck-schema.md`). A single JSON/TOML
  format covering every parameter the OpenLithoHub MRC checkers
  consume (`min_width_nm`, `min_spacing_nm`, `min_curvature_radius_nm`,
  `min_feature_area_nm2`) plus provenance/notes. New
  `openlithohub.benchmark.compliance.load_rule_deck()` validates the
  file against the in-tree schema (Draft 2020-12) and exposes
  `RuleDeck.kwargs_manhattan()` / `kwargs_curvilinear()` adapters to
  the existing `check_mrc` / `check_curvilinear_mrc` functions. Ships
  with a worked example (`benchmark/compliance/rule_decks/freepdk45_metal1.json`).
- **Measured-source / Zernike-pupil I/O** (`openlithohub._utils.optics`) —
  load lithography source maps and pupil aberrations from common formats
  for use with the Hopkins/SOCS forward model.
- **Calibre / CSV gauge parser** (`openlithohub.workflow.parse_gauge`) —
  ingests Calibre `.gg` and CSV gauge files and refuses unrecognized
  headers (rather than silently falling back to a wrong canonical
  column order, which would produce incorrect EPE numbers).
- **`openlithohub export` CLI** — exports trained models to
  ONNX / TorchScript / TensorRT-ready artifacts. Uses the dynamo ONNX
  path with a TorchScript fallback for models that aren't yet
  `torch.export`-able (e.g. NeuralILT). New `[export]` extra pulls in
  `onnxscript`.
- **End-to-end leaderboard submission test** — drives the full
  `auto-leaderboard.yml` pipeline (yaml load → schema validate →
  on-disk JSON) and asserts hostile YAML cannot inject extra fields,
  override `submission_id`, or smuggle Python objects.
- **`scripts/build_litho_tiny.py`** — deterministic 100-pair generator
  emitting an HF-ready parquet + dataset card under `out/litho-tiny/`.
- **HuggingFace authentication guide** (`docs/hf-auth.md`) — single-page
  walkthrough for unblocking gated Hub datasets (request access →
  `huggingface-cli login` / `HF_TOKEN` → verify), with corporate-proxy
  notes linking to `networking.md`. The `LithoSim` adapter's HTTP 401
  remediation now points users at this page.
- **`describe_simulators()`** in `openlithohub.simulators` — public
  `(name, class)` accessor for the simulator registry. Used by the CLI
  (`simulate list-backends --verbose`) to print the implementing class
  path so users can locate the source without grepping the registry.

### Changed

- **MRC `actual_nm` reports feature spine, not edge-pixel distance** — `compliance.mrc.check_mrc` now samples the local distance-transform maximum within the violating component (feature spine) instead of the edge-pixel value. The reported number is now the actual narrow-feature width, matching what foundry MRC docks would print.
- **Hopkins dose application** — `simulate_aerial_image_hopkins` no longer multiplies dose into both the aerial and the binarisation threshold (the previous double-application made `dose` cancel under the constant-threshold-resist path). Dose now affects the aerial image only; threshold is dose-independent. Closes #52.
- **Polar-grid Jacobian on Hopkins illumination samples** — source-sample weighting now applies the polar-grid Jacobian, fixing a systematic bias toward on-axis samples. Canonical Hopkins aerial mean baselines were rebaselined as part of the fix. Closes #29.
- **ILT receptive field lifted from 0 → 64** — `levelset-ilt` and `openilt` now declare a 64-px receptive field, so `--halo auto` accounts for the ILT spread when computing tile halos. Closes #75.
- **EUV H-V CD bias measured in shadowed-mask domain** — `compute_euv_hv_cd_bias` now operates on the shadowed-mask image rather than the pre-shadow aerial, matching the physical observable. Closes #24.
- **Eval-aggregation per-metric weighting + empty-mask floor** — aggregations now apply explicit per-metric weights and floor empty-mask tiles to a pass result instead of NaN-dropping silently.
- **Stochastic resist threshold consistency** — `--threshold` plumbed through `eval` / `optimize` / stochastic metrics so all three see the same value; default `0.225` everywhere. Closes #19, #33.
- **OASIS export single-tile shortcut** — when a layout is smaller than `tile_size`, the tiling pipeline now skips redundant tile dispatch and runs a single in-memory pass.
- **`align_resolution` binary-safe + 4D + deterministic** — re-binarises after rescale so masks stay {0,1}, supports 4D batched tensors, and is deterministic on non-integer scale factors.
- **`openlithohub simulate list-backends --verbose`** — adds a
  `--verbose`/`-v` flag that prints `name  module.ClassName` for every
  registered backend; bare invocation stays script-friendly (one name
  per line).

- **`--compile` defaults to `True`** on the `eval` and `optimize` CLI
  commands, with a graceful fallback to eager when `torch.compile`
  fails (Windows / non-Triton environments stay alive). The existing
  `--no-compile` escape hatch is preserved.
- **`README.md`** — prominent star CTA at the top and a JIT-acceleration
  bullet calling out the default `torch.compile` wrap.
- **`mypy --strict` enforced in CI**; pre-existing type errors cleared.

### Fixed

- **Monte-Carlo dose jitter applied as post-hoc aerial scaling** — previously `dose_jitter_sigma` was plumbed via `config.dose`, which (because `threshold = cfg.threshold * cfg.dose`) cancelled out and produced no observable jitter. Jitter now scales the aerial image and threshold offset *outside* the simulator config so the cancellation cannot bite. Closes #54.
- **Simultaneous bridge + break detection** — `_bridge_and_break_versus` builds nominal and trial component-label maps and detects each axis independently (a single trial can register on both); previously a trial that bridged one pair *and* broke a third left the net component count unchanged and was silently classified as a no-op. `failure_probability` is now clamped to `[0, 1]` instead of summing past 1. Closes #55.
- **forward_model 1-px axis raises instead of silent replicate fallback** — `simulate_aerial_image` on a degenerate H=1 or W=1 input now raises `ValueError`; the previous silent replicate-padded fallback produced meaningless aerials. Closes #10.
- **`auto_crop` replicate padding** — boundary crops near image edges now use replicate padding instead of zero-padding, eliminating a halo bias at the crop edge. Closes #32.
- **Process-window caveats documented** — `process_window` workflow docstring now states that the 5-corner sweep is a coarse approximation; production callers should pin their own corner set. Closes #27.
- **SVRF micron thresholds in `eda_bridge`** — SVRF rule decks emitted by the EDA bridge now use micron units (not nanometres) per Calibre conventions; curvilinear scope is documented. Closes #50, #51.
- **TorchScript `--verify` round-trip** — `openlithohub export run --verify` reloads the TorchScript artifact and checks the forward pass agrees with the PyTorch reference, catching dynamo / scripting regressions before they reach users.
- **Symmetric EPE + Hungarian hotspot match** — `compute_epe` now averages predicted-vs-target and target-vs-predicted contour distances; hotspot matching uses Hungarian assignment instead of greedy nearest-neighbour. Gauges report single-edge EPE consistently.
- **PVB Gaussian-vs-SOCS forward model documented** — PV-Band metric uses a fast Gaussian-PSF approximation, *not* the Hopkins/SOCS path used by `compute_l2_error` / `compute_wafer_epe`. The benchmarks doc now states this explicitly so callers do not assume PVB and L2 share a forward pass.
- **vis/contours figure leak** — opt-in `close=True` on `plot_contours` and friends so notebook callers don't leak matplotlib `Figure`s.
- **Polygon raster hole semantics** — `rasterize_polygons` now treats CCW outer rings as solid and CW inner rings as holes (consistent with OGC simple-features), not foreground regardless of orientation.
- **`contour_trace` small-feature handling** — single-pixel and single-row features are no longer silently dropped by the tracer.
- **BSpline diagnostics** — `BSplineCurve.evaluate` now raises with a clear message when control points are colinear (previously produced NaN coordinates downstream).
- **Trust-root manifest helpers + lithosim revision pin** — adapter integrity warnings on dataset open, manifest helpers exposed, and the LithoSim adapter pins a default revision so `huggingface_hub` cache misses don't silently pull `main`.
- **LitheEngine threads node-bound simulator into wafer metrics** — the OO façade no longer constructs an ad-hoc default simulator inside `evaluate`; the node-bound one configured on the engine is reused so wafer EPE and L2 see consistent optics.
- **VSB shot count via rectilinear decomposition** — `vsb_shot_count` now decomposes the mask into axis-aligned rectangles; the previous `perimeter² / area` heuristic over-counted Manhattan masks and under-counted curvilinear ones.
- **`typecheck` CI on `fb5c9b4`** — dropped two now-unused
  `# type: ignore` comments (`lithobench.py:234`, `ganopc.py:298`) and
  added `multivolumefile.*` / `py7zr.*` to the mypy
  `ignore_missing_imports` overrides. The two ignores were needed
  locally (where `py7zr` is installed and provides typing) but unused
  in CI (where the lazy optional deps are not installed and mypy
  treats the modules as `Any`). Commit `2aa14cb`.
- **Stochastic defect counting + NaN-safe aggregation** — fixed
  net-component-count formula in `compute_stochastic_robustness` so
  trials that simultaneously bridge some lines and break others
  contribute to both bridge and break probabilities. Aggregations in
  `eval run` are now NaN-safe; perimeter computation is border-safe so
  tiles touching the image edge no longer skew per-cm² rates.
- **`OASIS.MBW` → `OASIS.MASK` (SEMI P39)** — corrected naming in
  `workflow/export.py` and `workflow/layer_purpose.py` after community
  feedback that "MBW" is colloquial; SEMI P39 is the canonical name for
  the OASIS mask-data extension. Behaviour unchanged.
- **Contact email unified** — all CLA / SECURITY / DATA-LICENSES /
  COMMERCIAL-USE / community routing now points at
  `support@openlithohub.com`, with `conduct@openlithohub.com` reserved
  exclusively for Code-of-Conduct reports.
- **DRC notch detection** — `compliance.drc._find_notch_violations` now
  rejects background components that touch the image border (those are
  open exterior, not enclosed notches), eliminating false positives at
  tile boundaries. Notch semantics clarified in the docstring and
  covered by new constructed-violation tests.
- **`mypy --strict` regression in `compliance.drc`** — switched from
  `tensor.unique()` to `torch.unique(tensor)` so the typed-call gate
  passes (the bound-method overload is currently unannotated upstream).
- **`pip-audit` CI gate** — 11 unfixed PYSEC torch advisories triaged
  via `.github/pip-audit-ignore.txt` (each requires attacker-controlled
  inputs to specific torch APIs not reachable from OpenLithoHub's data
  path; revisit quarterly).
- **`contour_trace` truncation** — bound raised from `4*(h+w)` to
  `2*h*w` so serpentine boundaries no longer truncate silently.
- **Manhattan tracer X/T-junction ambiguity** — resolved by always
  picking the right-turn edge, keeping foreground consistently on
  the right; new diagonal-touch test.
- **Leaderboard schema lockdown** — `extra='forbid'`, URL-field
  validation, bounded string lengths; hostile-input tests added.
- **Leaderboard tracker** — type-checks `entries` on read;
  `secrets.token_hex(4)` for collision-free submission IDs.
- **`ModelHub._resolve_and_vet`** now returns all vetted IPs and the
  caller iterates with fallback, so dual-stack hosts work in
  IPv6-broken CI.
- **`Iccad16Dataset`** — warns per skipped row and raises if every row
  is malformed (was silent).
- **`workflow.gauges`** — refuses Calibre `.gg` files without a
  recognizable header (was silent fallback to canonical column
  order producing wrong EPE numbers).
- **Comprehensive code-review pass (2026-09)** — full-repo audit fixing
  30+ confirmed defects:

  - *Curvilinear SDF init* — `_approx_distance` convolved instead of
    min-relaxing, collapsing the signed distance field to zero after one
    pass; replaced with a proper chamfer min-propagation EDT (interior
    negative, exterior positive). `final_epe` in the optimize info dict
    now records the process-window EPE instead of the total loss.
  - *Anamorphic SMO* — removed the extra `ifftshift` that shifted the
    aerial image by half a field; anamorphic magnification now applied in
    the frequency domain (y cutoff × `mag_y/mag_x`) instead of a
    center-based `grid_sample` resampling that smeared a corner-aligned
    PSF; source parameters get gradients through a soft-edged pupil
    (previously the hard aperture comparison cut the graph and the joint
    "source-mask" optimisation never updated the source); fixed the x/y
    meshgrid axis swap in `mask_3d_shadow_correction`; corrected the TE/TM
    contrast-factor ordering to match the documented physics; clamped the
    three-beam modulation depth so the envelope stays non-negative;
    `ShotCountCost.evaluate` respects the configured pixel size.
  - *Stochastic metrics* — dropped the `softplus` on the aerial image in
    the Poisson rate model (it biased every dark pixel to `ln 2 × dose`
    photons, printing spurious resist in unexposed areas) in
    `stochastic_loss` and `coverage_gate`; the conformal coverage gate now
    computes the calibration quantile on the external-predictor path too
    (it silently degraded to 1.0) and predicts over the same defocus range
    it calibrated on; `StochasticProcessWindow` reports the longest
    contiguous passing focus range and passes defocus to custom forward
    models that accept it; `StochasticAwareLoss` is callable as
    documented.
  - *Tiling / Schwarz* — `_inject_boundary_data` now derives overlaps
    from tile origins (anchored edge tiles never exchanged boundary data
    before) and starts each Schwarz round from the tile's own previous
    solution; `sweep_overlap_convergence` labels results with the true
    cumulative iteration count; `cross_tile_epe/contour_residual` accept
    tile origins to compare only geometrically adjacent pairs; the SRAF
    consistency denominator counts union-of-SRAF pixels instead of the
    whole patch; `_squeeze` raises on non-singleton batch dims instead of
    looping forever; `stitch_tiles` no longer leaves a black seam at
    `overlap=1`.
  - *Data adapters* — `Iccad16Dataset` rasterization now delegates to the
    canonical `rasterize_cell_layer` (the in-house y-up fill produced
    vertically mirrored masks vs. `load_layout`, and trapezoid-bbox
    filling over-filled non-Manhattan geometry); ORFS tile origins report
    the correct lower-left y in layout nm; `load_layout` rasterizes GDS
    path shapes instead of dropping them and rejects `pixel_nm <= 0`;
    `data show` PNG export is no longer vertically mirrored; an invalid
    user-supplied layermap JSON now warns instead of breaking the
    package import.
  - *Server / CLI / leaderboard* — the model-cache LRU no longer tears
    down a model another request is using (refcount + deferred teardown);
    `--quencher > 0` is now rejected with `--submit` (matching the
    documented incompatibility); `--limit` and `--writer` validate their
    values; submission IDs are generated from a path-safe charset;
    `simulate` loads `.npy` with `allow_pickle=False`; macOS
    `multiproc_predict` no longer crashes/deadlocks — shared-memory block
    names are hashed under the 31-char POSIX limit and workers are
    spawned (fork of a torch-initialised process deadlocks on darwin).
  - *3D stochastic model* — SE-kernel device placement (CUDA convolutions
    no longer crash), even kernel sizes honoured, LCDU ensemble statistics
    no longer mix pattern variance into trial variance, Pearson
    correlation uses matching population moments, line-collapse
    monotonicity check direction corrected, failure-correlation length
    reports the half-window for strongly correlated maps.
  - *Performance* — Manhattan contour edge detection vectorised (was a
    ~33M-iteration Python loop on 4096² layouts); ASAP7 / FreePDK45
    adapters parse the GDS once instead of per cell; GPU benchmark runs
    the real forward model on the requested device and resets peak-memory
    stats per config; shortest-run helper vectorised without per-pixel
    device syncs; diffusion β-schedule `cumprod` computed once.
  - *Train script* — the GAN-OPC memmap cache is built in a temp file and
    atomically renamed (an interrupted run no longer leaves a truncated
    cache silently treated as valid); `component_history` is actually
    populated so the PVB plateau monitor and saved metadata work.

## [0.1.0a2] - 2026-05-19

First public alpha. Establishes the `openlithohub` PyPI name; install
with `pip install --pre openlithohub` until a stable `0.1.0` is cut.
API surface is **not** stable.

### Added

- **First PyPI release** — `openlithohub-0.1.0a2` published via GitHub
  Actions trusted publishing (`.github/workflows/publish.yml`) on
  every `v*` tag. `hatch-vcs` derives the version from the git tag.
- **PDK layer registry (`openlithohub.data._layers`)** — single source
  of truth for the (layer, datatype) pairs each adapter rasterizes by
  default (`asap7=10/0`, `freepdk45=11/0`, `orfs_asap7=20/0`). Each
  adapter's `DEFAULT_DESIGN_LAYER` re-exports the registry entry.
- **Docs link-boundary lint (`scripts/lint_docs_links.py`)** — new
  Docs-CI step that fails when a Markdown link in `docs/**` resolves
  outside `docs/`, catching the class of bug that only `mkdocs build
  --strict` surfaces (and only after a page is added to nav).
- **End-to-end URL-cache test for `ModelHub.download_weights`** —
  locks the on-disk shape of URL-keyed cache entries and asserts that
  `list_cached → clear_cache` round-trips cleanly.

### Changed

- **`ModelHub` class docstring** documents the three identifier
  shapes that flow through the cache (`owner/repo`, `owner--repo`,
  `url--<hex>`); auto-rendered onto `docs/api/models.md` via
  mkdocstrings.
- **`mkdocs-material`** pinned to `>=9.4,<10` to avoid the
  backwards-incompatible mkdocs-material/mkdocs 2.0 series that
  drops the plugin system this site depends on (mkdocstrings,
  mkdocs-gen-files).

### Fixed

- **`ModelHub.clear_cache` path traversal** — caller-supplied
  `model_id` now passes through the same `_safe_cache_segment`
  validator as `download_weights`, so a `..` cannot escape `cache_dir`
  and `rmtree` a sibling. URL-keyed entries (`url--<hex>`) are
  accepted in their on-disk form so `list_cached` output round-trips.
- **`OrfsArtifactDataset` docstring layer mismatch** — corrected from
  `(10, 0)` to `(20, 0)` to match `DEFAULT_DESIGN_LAYER` (post-route
  ORFS-ASAP7 numbers M1 differently from the cell-library source).
- **`docs/README_zh.md` license link** — replaced relative `../LICENSE`
  with the canonical GitHub URL so adding the page to the mkdocs nav
  later does not break `mkdocs build --strict`.
- **Pytest deprecation noise** — narrow `filterwarnings` entry
  silences torch's internal `torch.jit.script_method` deprecation
  (14 hits in `tests/test_utils/test_hopkins.py`) without masking
  other warnings.

## [0.1.0a1] - 2026-05-19 [YANKED]

Tag exists in git history but the publish workflow failed at the OIDC
exchange step due to a case-mismatched PyPI trusted publisher
registration. No artifact reached PyPI. Superseded by `0.1.0a2`.

## [0.1.0-pre] - pre-release work

### Added

- **Real PDK rollout (issue #4)** — three new dataset adapters that bring OpenLithoHub onto industrial layouts:
    - **`Asap7Dataset`** (`openlithohub.data.asap7`) — loads the BSD-3-Clause [ASAP7 7nm predictive PDK](https://github.com/The-OpenROAD-Project/asap7), exposes a canonical 4-cell smoke set (`INVx1`, `NAND2x1`, `NOR2x1`, `DFFHQNx1`), gated by `--accept-license`. Adds a klayout-based GDS rasterizer reused by the FreePDK45 and ORFS adapters.
    - **`FreePdk45Dataset`** (`openlithohub.data.freepdk45`) — loads FreePDK45 + NanGate Open Cell Library from the [mflowgen mirror](https://github.com/mflowgen/freepdk-45nm); exposes the canonical 4-cell smoke set (`INV_X1`, `NAND2_X1`, `NOR2_X1`, `DFF_X1`); stacked-license disclosure since the mirror ships no LICENSE file.
    - **`OrfsArtifactDataset`** (`openlithohub.data.orfs`) — loads ASAP7-routed RTL→GDSII outputs from [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts), cuts the routed block into 2 µm × 2 µm and 5 µm × 5 µm tiles (canonical AI-OPC inference windows), defaults to ORFS metal1 layer 20/0.
- **`build-asap7-mock-alu` GitHub Actions workflow** (`.github/workflows/build-asap7-mock-alu.yml`) — runs ORFS in the `openroad/orfs` container against pinned commit `74b5f96` and uploads the routed GDS as a workflow artifact (~25 min for `mock-alu`). Companion `scripts/build_riscv_alu.sh` for local Linux runs.
- **CLI `--dataset {asap7,freepdk45,orfs}`** + `--accept-license` and `--tile-nm` flags on `openlithohub eval run`. The CLI now supports five datasets total (LithoBench, LithoSim, ASAP7, FreePDK45, ORFS).
- **Phase-3 baseline (`baselines/orfs-mock-alu-{2um,5um}.json`)** — first numbers against a real ASAP7-routed RISC-V mock-alu. PVB mean 15.07 nm (729 × 2 µm tiles) / 14.98 nm (121 × 5 µm tiles) at `pixel_nm=4.0`.
- **Before/after PNG** at `docs/assets/orfs-mock-alu-tile.png` (design / rule-OPC mask / resist contour) embedded in `docs/benchmarks.md`.
- **RFC 0001 — Layout-MAE base model** (`docs/rfcs/0001-base-model.md`) and **RFC 0002 — Layout Tokens** (`docs/rfcs/0002-layout-tokens.md`) lock in the v0.2 path: a small ViT-S MAE pretrained on rasterised PDK layouts as the open backbone, and a polygon-vertex tokeniser that round-trips losslessly and replaces the diffusion stub with an autoregressive sequence model.
- **Rule-based synthetic layout generator (`openlithohub.synth`)** — PDK-aware patterns (FreePDK45, ASAP7) for SRAM, contact arrays, and randomly routed metal that pass MRC by construction, plus `openlithohub synth` CLI for batch export and a `DiffusionLayoutGenerator` stub pinned to RFC 0001 + 0002.
- **EUV 3D-mask shadow proxy + Monte Carlo failure metric** (`openlithohub.benchmark.metrics.euv_3d`, `openlithohub.benchmark.metrics.monte_carlo`) — first-order anisotropic shadowing operator parameterised by absorber thickness and chief-ray azimuth, plus a higher-fidelity Monte Carlo failure path that runs against any registered simulator backend.
- **Vendor-neutral simulator hook API (`openlithohub.simulators`)** — `BaseSimulator` ABC with a Hopkins reference adapter (`hopkins_sim`) shipping in-tree and config-validated stubs for Calibre nmOPC and Tachyon, exposed via `openlithohub simulate` CLI.
- **Mini-Hackathon (2026-Q3) charter + leaderboard track** (`docs/hackathon.md`) — frozen test split, hard MRC/DRC gate, separate `track` field on leaderboard submissions.
- **Auto-Leaderboard CI** (`.github/workflows/auto-leaderboard.yml`) — claim-and-verify-by-numbers workflow that validates `submissions/*.yaml` against the BenchmarkResult schema. Submission template at `submissions/_template/example-model.yaml`; full guide at `docs/leaderboard-submission.md` (now also documents the optional `track` field).
- **Community charter** (`docs/community.md`) — Discord-only (English-first), launching 2026-Q3. Channel layout, etiquette, moderator policy, onboarding flow.
- **v0.1 launch announcement** (`docs/announcements/2026-05-launch.md`) — paste-ready copy for X / LinkedIn / Zhihu / HuggingFace Forum.
- **AI-engineer terminology guide** (`docs/lithography-for-ai-engineers.md`) — bridges ML vocabulary and lithography terminology for newcomers.
- **Multi-stage KLayout Docker build** — slimmer image, separate build/runtime stages.
- **OpenLithoHub logo** in README and MkDocs (light + dark variants).
- **Paper-ready visualization (`openlithohub.vis`)** — `plot_contours`, `plot_pv_band`, and the `paper_style` context manager (with `IEEE_STYLE` and `SPIE_STYLE` presets) emit IEEE / SPIE column-width figures with a colorblind-safe palette, vector PDF defaults, and Type-42 fonts.
- **Hermetic dummy layout generator** — `openlithohub.data.generate_dummy_layout`, `generate_dummy_pair`, and `DummyLayoutSpec` produce deterministic, DRC-clean synthetic layouts with only NumPy and PyTorch — usable in CI and Colab without the `[workflow]` extras.
- **EDA bridge templates (`openlithohub.workflow.eda_bridge`)** — `BridgeRules`, `emit_calibre_svrf`, `emit_icv_runset`, and `emit_bridge_bundle` write minimal Calibre nmDRC and Synopsys IC Validator runsets next to an exported OASIS file.
- **Colab quickstart** — `notebooks/quickstart.ipynb` runs install → dummy layout → metrics → paper figure end-to-end on Colab's stock runtime.
- **Spaces leaderboard tab** — `spaces/app.py` now ships a third tab that renders the JSON leaderboard with a refresh button.
- **Rule-based OPC model** — analytic per-edge bias OPC baseline registered as `rule-based-opc`.
- **Differentiable Hopkins forward model** — partial-coherent imaging via SVD-truncated SOCS (`openlithohub._utils.hopkins`), supporting circular / annular / dipole / quasar illumination, defocus, and per-(params, grid) kernel caching. End-to-end auto-differentiable so it can drop into AI-OPC training and ILT loops.
- **`LevelSetILTModel.forward_model="hopkins"`** — opt-in switch from the default Gaussian PSF to the new Hopkins SOCS model, with optional `HopkinsParams` override.
- **`differentiable_threshold`** — standalone sigmoid-based resist threshold helper exposed from `openlithohub._utils`.
- **Baseline reference numbers** — `scripts/generate_baselines.py` runs `dummy-identity`, `rule-based-opc`, `levelset-ilt`, and `neural-ilt` against eight synthetic 64×64 layouts (or LithoBench when `--data-root` is supplied) and writes `baselines/results.json` + `baselines/results.md`.
- **ICCAD'16 Problem C hotspot dataset (`openlithohub.data.Iccad16Dataset`)** — klayout-based OASIS rasterizer for the ICCAD 2016 EUV hotspot benchmark. Returns `LithoSample(design, mask=None, ...)` with hotspot annotations and clip-site bboxes in `metadata`.
- **GAN-OPC paired-mask dataset (`openlithohub.data.GanOpcDataset`)** — loader for the ~4875 paired `(target, OPC mask)` 2048×2048 PNGs from Yang et al. *GAN-OPC* (TCAD'20), suitable for AI-OPC training.
- **Hotspot detection metric (`compute_hotspot_detection`)** — distance-tolerant greedy point matching → recall / precision / F1, configurable via `match_radius_nm`.
- **Hotspot baseline pipeline (`scripts/run_hotspot_baseline.py`)** — end-to-end wiring of `Iccad16Dataset` → predictor → metric across three sanity baselines (empty / saturated grid / clip-centers); writes `hotspot_results.{json,md}`.
- **`docs/benchmarks.md`** — new docs page covering baseline numbers, reproduction, and the differentiable forward models.
- **LevelSet-ILT model** — iterative gradient-descent mask optimization using differentiable forward model
- **Neural-ILT model** — U-Net based single-pass mask prediction with pretrained weight support
- **Model Hub** — download and cache pretrained weights from HuggingFace Hub or direct URLs
- **DTCO Process Node Config** — physical parameters for 3nm-euv, 5nm-euv, 7nm, 45nm nodes
- **Resist simulation** — chemically-amplified resist model with acid diffusion and quencher
- **Jupyter integration** — `%load_ext openlithohub.jupyter` magic commands and display helpers
- **PyPI publish workflow** — automated package publishing on version tags
- **Docker image** — containerized deployment via GitHub Container Registry
- **Performance benchmarks** — pytest-benchmark suite for critical paths
- **py.typed marker** — PEP 561 type information support
- `[models]` and `[jupyter]` optional dependency groups
- 73 new tests (217 total), covering utils, models, process nodes, and integration

### Fixed

- `distance_transform` infinite loop on all-foreground masks (pre-existing bug)
- CLI `--node` parameter now auto-configures pixel size and MRC thresholds from process node presets

### Changed

- Project scaffold with 5-layer architecture
- Abstract interfaces: `DatasetAdapter`, `LithographyModel`
- CLI skeleton: `openlithohub eval`, `openlithohub optimize`
- Benchmark metric stubs: EPE, PV Band, shot count, stochastic robustness
- Compliance check stubs: MRC, DRC
- Workflow stubs: layout parsing, tiling, contour extraction, OASIS export
- Leaderboard schemas with Pydantic models
- Model registry with decorator-based registration
- Dummy identity model for pipeline testing
- CI pipeline (GitHub Actions): lint + test on Python 3.10/3.11/3.12
- Pre-commit hooks (ruff, trailing whitespace, YAML/TOML checks)
- MkDocs Material documentation configuration
- Full MkDocs documentation site (getting started, architecture, CLI reference, API docs)
- GitHub Actions workflow for docs deployment to GitHub Pages
- HuggingFace Spaces web playground (Gradio-based interactive demo)
  - Synthetic pattern evaluation (line/space, contact holes, SRAM)
  - Upload custom masks for EPE/MRC evaluation
  - Edge contour visualization overlay
  - Public leaderboard view
