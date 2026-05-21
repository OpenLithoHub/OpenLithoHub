# RFC 0006 — MCP (Model Context Protocol) Bridge

| | |
|-|-|
| Status | Draft |
| Author | OpenLithoHub maintainers |
| Created | 2026-05-21 |
| Targets | v0.4 |
| Related | `openlithohub.cli.eval_cmd`, `openlithohub.cli.serve_cmd`, `openlithohub.benchmark.report`, `openlithohub.benchmark.compliance.{drc,mrc}`, `openlithohub.workflow.process_window`, `openlithohub.workflow.tiling`, RFC 0003 (MRC Rule Deck Schema), RFC 0005 (Halo Sizing) |

## Summary

Expose OpenLithoHub's evaluation and optimization surface to LLM agents
(Claude Desktop, Cursor, custom MCP clients) through a thin Model
Context Protocol bridge. The bridge is a JSON-pass-through: every
structural decision lives in the CLI / engine, every framing decision
lives in the MCP transport, and nothing in between gets to invent
schema.

The wedge is conversational lithography verification — a fab engineer
asks "are there any DRC violations in the lower-left corner?" and the
agent calls `evaluate(report_level=detailed)`, gets back a structured
`violations[]` array, and narrates the answer against the layout. The
follow-up `optimize(violation_id=...)` call closes the loop without the
agent re-deriving context from a free-text report.

This RFC defines three gates the bridge must clear before merge, the
schema decisions the engine must implement first, and the control-flow
contract that ties them together.

## Background — what the bridge buys

Today, an agent that wants to use OpenLithoHub has two options:

1. **Shell out to `openlithohub eval`**, parse text output. Brittle —
   any CLI cosmetic change breaks the agent.
2. **Call the FastAPI engine in `serve_cmd.py`**, parse a JSON response.
   Better, but the JSON shape is `{metric_name: scalar}` (a leaderboard
   row), not a violation list — agents cannot ground a "where is the
   violation" question in coordinates.

MCP gives us a third option: a typed tool surface (`evaluate`,
`optimize`, `simulate`) where every call has a JSON-Schema'd request
and response, the agent sees the schema at `tools/list` time, and
streaming/cancellation are part of the protocol rather than ad-hoc.

The catch is that MCP-over-stdio is fragile in ways that scientific
Python tooling routinely violates (stdout contamination, no abort
plumbing on long ops, opaque `progressToken` semantics). The bridge
must close those failure modes structurally, not by hoping callers
behave.

## Current state (factual, verified 2026-05-21)

- **`openlithohub eval --format=json`** exists in
  `src/openlithohub/cli/eval_cmd.py:31` and
  `src/openlithohub/benchmark/report.py:22`. It emits
  `json.dumps(aggregated_metrics, indent=2)` — a flat
  `{metric_name: scalar}` dict. No `violations[]`, no rule-IDs, no
  per-violation coordinates.
- **DRC/MRC violation records exist internally.**
  `src/openlithohub/benchmark/compliance/mrc.py:27` defines
  `violations: list[dict[str, float]]` on `MRCResult`; `drc.py` is
  parallel. The aggregator throws them away before serialization — the
  data is there, the contract isn't.
- **Tile loop** in `src/openlithohub/workflow/process_window.py:89`
  iterates `for corner in corners:` over process-window samples;
  `src/openlithohub/workflow/tiling.py:142` iterates
  `for tile, result in tiles:`. Neither loop checks a cancellation
  token — once started, an `optimize` call runs to completion.
- **No MCP server exists yet.** `serve_cmd.py` only boots a FastAPI
  engine; there is no stdio MCP transport in tree.
- **No JSON-Schema versioning.** The `eval --format=json` output has no
  `schema_version` field, no JSON-Schema document, no consumer-side
  validator.

## Design

### 1 · Three frozen gates (G1, G2, G3)

The bridge does not merge until all three pass. Each gate has a
falsifiable acceptance test.

#### G1 — stdio hygiene

MCP-over-stdio frames JSON-RPC on the server's `stdout`. Any stray
`print()`, `tqdm` progress bar, deprecation warning routed to `stdout`,
or rich-console output from the wrapped engine breaks the protocol —
the client sees malformed frames and disconnects.

**Hard rules:**

1. The **in-process path** (preferred — lower latency, no subprocess
   overhead) must redirect `openlithohub`'s logger handlers off
   `sys.stdout` *before* installing the MCP stdio transport. Logging
   goes to `stderr` or to a structured artifact path. The MCP transport
   owns `stdout` exclusively.
2. The **subprocess path** (fallback for engines that genuinely need
   process isolation) must capture both `stdout` and `stderr` via
   `subprocess.run(..., stdout=PIPE, stderr=PIPE)` — never `inherit`.
   The wrapper parses child stdout into `violations[]` and forwards
   child stderr to the MCP server's own stderr or to artifacts.
3. **Acceptance test (regression gate):** run `openlithohub eval`
   against a fixture and pipe its stdout to:
   ```
   python -c "import sys, json; [json.loads(l) for l in sys.stdin if l.strip()]"
   ```
   Every line must parse as JSON. One failure = bridge rejected.

#### G2 — long-run control + cancellation

OPC tile loops run for tens of seconds to minutes. An LLM may fire
`evaluate()`, watch the partial stream, decide to pivot, and the engine
must actually stop — not just be ignored. Most MCP servers ship
streaming-out without a real abort path; the engine keeps burning GPU
on a request nobody reads.

**Hard rules:**

1. **Two long-op contracts, both supported:**
   - **Path A — `progressToken`:** when the client supplies one in the
     request, the engine emits incremental progress notifications. Fast
     path for clients that implement it (Claude Desktop does; not all
     do).
   - **Path B — `job_id` + `poll`:** the documented contract regardless
     of client. `evaluate(...)` returns a `job_id` immediately;
     `poll(job_id)` returns `{status, progress, result?}`. Clients that
     don't speak `progressToken` use this path; clients that do can
     still use it (defensive plumbing). This sidesteps client
     variability entirely.
2. **Cancel-on-disconnect is a P0 acceptance criterion.** A
   cancellation token threads through `workflow/process_window.py` and
   `workflow/tiling.py`'s tile loops; each iteration checks the token
   and the forward-model kernels (Hopkins SOCS) check it at a
   granularity that frees the GPU within ~1 s of the disconnect signal.
3. **Acceptance test:** start a long `optimize`, kill the MCP client
   mid-stream, assert the engine process's GPU memory drops within
   ~2 s and the job's terminal state is `cancelled` (not `completed`,
   not `running`). Readback proof, not a log line.

#### G3 — structured-output contract at the CLI layer

The bridge does **not** parse anything. If the agent needs structured
violations, that's a CLI/engine responsibility, not bridge middleware.
This is the architectural constraint that decides whether the bridge
stays a thin shim (good) or becomes a domain-translator with its own
versioning problem (bad).

**Hard rules:**

1. **`openlithohub eval --report-level=detailed`** (new flag) extends
   the JSON contract to include a `violations[]` array alongside the
   existing aggregate metrics. The leaderboard path
   (`--report-level=aggregate`, default) stays bit-identical to today's
   output — no leaderboard regression.
2. **The bridge is a JSON-pass-through with a schema validator.** The
   MCP `evaluate` tool's contract is "return whatever the engine
   emitted, after schema validation, unmodified." Any structural
   transformation is a CLI/engine responsibility.
3. **Acceptance test:** the same fixture round-trips identical bytes
   through three paths — direct CLI, MCP in-process bridge, MCP
   subprocess bridge — modulo only the MCP envelope. No string
   substitution, no re-keying, no float reformatting.

### 2 · Violation schema

The shape `eval --report-level=detailed` emits:

```json
{
  "schema_version": "1.0.0",
  "metrics": {
    "epe_mean_nm": 1.42,
    "epe_max_nm": 4.91,
    "mrc_pass_rate": 0.998,
    "drc_pass_rate": 1.0
  },
  "violations": [
    {
      "violation_id": "v1a2b3c4d5e6f7g8h",
      "violation_group_id": "9z8y7x6w5v4u3t2s",
      "rule_id": "M1.W.MIN",
      "rule_deck": "asap7-v1.2",
      "severity": "error",
      "coordinate_frame": "post_transform_canvas",
      "location_nm": [12450.0, 8300.0],
      "transforms": [{"op": "tile_stitch", "params": {"origin": [0, 0]}}],
      "context": {"tile_id": "T_2_3", "tile_local_nm": [450.0, 300.0]},
      "dedup_strategy": "canonical",
      "dedup_provenance": {
        "rule_deck_hash": "sha256:abc...",
        "radius_nm": 18.0,
        "rule_id": "M1.W.MIN"
      }
    }
  ]
}
```

#### 2a · `coordinate_frame` enum

Free-form strings invite silent shifts of the fix to the wrong
location. Pinned values:

| Value | Meaning |
|-------|---------|
| `input_layout_native` | User-supplied layout's reference frame, pre-rasterization, pre-tile, pre-rotation |
| `post_transform_canvas` | Engine working canvas, after internal rotation/mirror/tile-stitching |
| `tile_local` | Inside a specific tile (requires sibling `tile_origin_nm` in `post_transform_canvas` frame) |
| `wafer_global` | Reserved for future EUV pipelines that resolve to wafer coordinates |

The schema requires both the enum value and, where applicable, a
`transforms: [{op, params}]` chain so a downstream visualizer can
round-trip back to the user's original frame without guessing. Repair
agents that don't understand the transform chain are required to
reject the violation rather than silently shift the fix.

#### 2b · `violation_group_id` derivation

Tile-boundary violations: when a defect straddles two tiles, each tile
emits its own record (the rule check runs per-tile; the rule deck
doesn't know the partitioning). Records that fall within
`dedup_radius_nm` of a sibling get the same `violation_group_id`.
Repair agents iterate `violations[]`, group by `violation_group_id`,
and act only on the canonical record per group — the rest are reported
for traceability. This prevents the "second fix looks like a
regression introduced by the first" failure mode.

**Derivation is deterministic from inputs**, not opaque UUID:

```python
violation_group_id = blake2b(
    rule_deck_hash
    + dedup_radius_nm.to_bytes(...)
    + canonical_location_nm[0].to_bytes(...)
    + canonical_location_nm[1].to_bytes(...)
    + rule_id.encode()
).hexdigest()[:16]
```

Stable across runs given identical inputs. Stable under tile
re-partitioning (dedup runs *after* tiles are stitched in canvas
frame). Re-derivable by any consumer that has the rule deck pinned —
critical for repair agents resuming a workflow days later. The
`dedup_provenance` sibling on every record (not just canonical ones)
lets a stale agent audit-replay the partition without re-running eval.

Cross-vendor authority falls out: vendor-A and vendor-B rule decks
produce different group IDs because their `rule_deck_hash` differs,
even on byte-identical violation coordinates.

`dedup_radius_nm` is a per-rule property in the rule deck (DRC width
violations: ~`min_feature_nm`; MRC: 0 since they're per-edge not
per-region). The aggregation step runs *before* serialization — the
bridge sees pre-deduped records. **No bridge-side dedup logic.**

#### 2c · `dedup_strategy: canonical | shadow`

Each record is one of two:

- **`canonical`** — first record in its group; `optimize` resolves to
  this record's geometry and fixes it.
- **`shadow`** — additional record(s) in the same group; reported for
  traceability so an agent can reason about "this defect was *also*
  observed from tile T2's perspective." Shadow records are not
  independent repair targets.

Default `optimize(violation_id=...)` resolution: regardless of whether
the supplied ID points to a canonical or shadow record, the engine
resolves to the group's canonical record and fixes that. The response
lists all group members as "resolved by group fix" so the agent's
traceability ledger stays consistent. Opt-out:
`optimize(violation_id=v37, scope="record_only")` for the rare case an
agent genuinely wants per-record geometry.

### 3 · Capabilities discovery

Agent planners cannot parse markdown at runtime. Per-engine-path
detail-level support is exposed structurally on the MCP tool descriptor
returned by `tools/list`:

```json
{
  "report_levels": ["aggregate", "detailed"],
  "engine_paths": {
    "drc.standard":   {"detailed": "single_pass"},
    "mrc.standard":   {"detailed": "single_pass"},
    "pvband.4corner": {"detailed": "rerun_required"},
    "epe.gauge":      {"detailed": "unsupported"},
    "euv.stochastic": {
      "detailed": "conditional",
      "precondition": {
        "layout_area_nm2_max": 5e10,
        "violation_count_max": 50000,
        "on_exceeded": "degrade_to_aggregate",
        "signaled_via": "degraded_to_aggregate: true on response"
      }
    }
  },
  "schema_version": "1.0.0",
  "max_violations_per_response": 10000
}
```

The four values:

| Value | Meaning |
|-------|---------|
| `single_pass` | Detailed output is free — DRC/MRC already build the list internally; the aggregator just doesn't throw it away |
| `rerun_required` | Detailed output requires a separate engine pass — e.g. PVB pipelines that throw away geometry between rule check and aggregator |
| `conditional` | Detailed output works below an explicit precondition (layout size, violation count); above the cutoff the engine signals `degraded_to_aggregate: true` and falls back. **Not silent.** |
| `unsupported` | The path cannot produce per-violation output without a structural change — the agent's planner must not ask |

Error-code-and-retry was rejected: it doubles the round-trip cost on
the *expected* path. `report_level_unsupported` still exists as a
fallback error for engine/descriptor drift (e.g. engine downgrade in
production), but it is defensive plumbing, not the documented
contract.

### 4 · `schema_version` policy

Asymmetric `additionalProperties` — the pattern every JSON-Schema
ecosystem converges on after a decade of pain.

- **Response schemas** (engine → bridge → agent): `additionalProperties:
  true`. Adding a new violation field is a **minor** bump; pinned
  consumers ignore unknown keys; new consumers light up the new field.
- **Request schemas** (agent → bridge → engine): `additionalProperties:
  false`. Typo in a field name is a hard error, not silently ignored.
  The failure mode of "I called `evaluate(report_levle=detailed)` and
  got default behavior" is exactly what this prevents.
- **Major bumps** reserved for: removing/renaming fields, changing
  field semantics (e.g. switching `location_nm` from canvas to native
  frame without an enum), tightening enum values. Any of these breaks
  pinned consumers and deserves the major-bump signal.
- **Enum widening** is `additionalProperties`-style too. Consumers are
  required (in the schema doc, hard requirement) to handle an
  `unknown` fallback case — `switch`/`match` must have a default
  branch. Adding `coordinate_frame: wafer_global` in a future minor
  bump must not crash pinned consumers.
- **`schema_version` rides on every response payload**, not just the
  tool descriptor — so even cached responses round-trip safely.

### 5 · Session semantics & `violation_id` resolution

When `optimize` consumes a `violation_id` returned from a prior
`evaluate`, two paths:

- **Fast path (server-side cache).** Session = MCP connection
  lifetime, from `initialize` to disconnect. Cache keyed by
  `(session_id, violation_id) → (coordinate_frame, transforms[],
  canonical_location_nm, rule_deck_hash)`. ~megabytes per session,
  evicted on disconnect.
- **Cache-miss fallback (self-describing).** `violation_id` encodes
  `schema_version || rule_deck_hash || violation_group_id ||
  record_index` (base32, ~40 chars). On `cache_evicted`, the engine
  re-resolves from the ID alone provided the layout artifact is still
  pinned (the agent hints `layout_artifact_id` on the `optimize`
  call). If the layout isn't pinned, the error is `layout_not_pinned`
  rather than `cache_evicted` — different recovery paths.

Cross-session resumption works whenever
`(rule_deck_hash, layout_artifact_id, violation_id)` is reproducible —
which is the deterministic-derivation property from §2b. The session
cache is a latency optimization, not a correctness requirement.

### 6 · Control flow

```
                  ┌──────────────────────────────────────────┐
                  │  MCP client (Claude Desktop / Cursor)    │
                  └───────────────┬──────────────────────────┘
                                  │  JSON-RPC over stdio
                                  ▼
              ┌───────────────────────────────────┐
              │  MCP server (openlithohub-mcp)    │  ── owns stdout
              │  ─ stdio transport                │
              │  ─ schema validator (JSON-Schema) │
              │  ─ session cache                  │
              │  ─ progressToken / job_id router  │
              └───────────────┬───────────────────┘
                              │  validated request
                              ▼
              ┌───────────────────────────────────┐
              │  CLI / in-process engine adapter  │  ── logger → stderr
              │  ─ openlithohub eval / optimize   │
              │  ─ --report-level={agg, detailed} │
              │  ─ cancellation token injection   │
              └───────────────┬───────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │  Engine                                                │
   │  ┌────────────────────┐   ┌─────────────────────────┐ │
   │  │ workflow/tiling    │──▶│ Hopkins / SOCS forward  │ │
   │  │   for tile in ...: │   │ (cancel-token check)    │ │
   │  │     check_token()  │   └─────────────────────────┘ │
   │  └─────────┬──────────┘                                │
   │            ▼                                           │
   │  ┌────────────────────────────────────────────────┐   │
   │  │ benchmark/compliance/{drc,mrc}                 │   │
   │  │   per-tile violations: list[dict[...]]         │   │
   │  └─────────┬──────────────────────────────────────┘   │
   │            ▼                                           │
   │  ┌────────────────────────────────────────────────┐   │
   │  │ Post-tile aggregator                           │   │
   │  │   ─ stitch tiles in post_transform_canvas      │   │
   │  │   ─ dedup by (rule_id, radius_nm, location_nm) │   │
   │  │   ─ assign violation_group_id (deterministic)  │   │
   │  │   ─ tag canonical | shadow                     │   │
   │  └─────────┬──────────────────────────────────────┘   │
   │            ▼                                           │
   │  ┌────────────────────────────────────────────────┐   │
   │  │ benchmark/report.generate_report               │   │
   │  │   ─ aggregate metrics (always)                 │   │
   │  │   ─ violations[] (when --report-level=detailed)│   │
   │  │   ─ schema_version stamp                       │   │
   │  └─────────┬──────────────────────────────────────┘   │
   └────────────┼───────────────────────────────────────────┘
                │  JSON bytes (validated)
                ▼  ── cancel/disconnect short-circuits at every ▲
              [ MCP server emits to client unmodified ]
```

#### 6a · Cancellation pseudocode

The token threads through three layers; each one checks at a
granularity that frees the GPU within ~1 s of disconnect.

```python
# workflow/tiling.py — outer tile loop
def run_tiled(layout, model, halo_px, *, cancel_token):
    tiles = tile_layout(layout, halo_px=halo_px)
    results = []
    for tile in tiles:
        cancel_token.check()                    # boundary 1: per tile
        result = model.predict(tile, cancel_token=cancel_token)
        results.append(result)
    return stitch(results)

# models/<engine>.py — kernel-level checkpoint
def predict(self, tile, *, cancel_token):
    for corner in self.process_window_corners:
        cancel_token.check()                    # boundary 2: per corner
        aerial = simulate_aerial_image(...)     # ~100ms-1s on GPU
    # boundary 3: if simulate_aerial_image is itself >1s, it owns
    # an internal token check before the GPU launch and after each
    # batch — the kernel API takes cancel_token as a kwarg.

# bridge — token lifecycle tied to MCP request
def evaluate_handler(request, mcp_session):
    token = CancellationToken()
    mcp_session.on_disconnect(token.cancel)
    mcp_session.on_cancel_notification(request.id, token.cancel)
    try:
        return run_eval(..., cancel_token=token)
    except Cancelled:
        return {"status": "cancelled", "job_id": request.job_id}
```

The token is a plain object, not asyncio-specific; the engine code
calls `token.check()` synchronously and raises `Cancelled` if set.
Async wrappers in the bridge translate to MCP semantics. No engine
code needs to know about MCP.

## Hard constraints

1. **No bridge-side parsing or transformation.** The bridge validates
   schema and forwards bytes. Any structural change to the violation
   shape lives in `benchmark/report.py`, never in the bridge.
2. **No leaderboard regression.** `eval --report-level=aggregate`
   (default) emits bit-identical output to today's
   `eval --format=json`. The leaderboard CI path is unchanged.
3. **No `stdout` writes from the engine when MCP-stdio is active.**
   Logging routed to `stderr` or artifacts, before the transport is
   installed. Verified by the JSON round-trip regression test.
4. **Cancellation is structural, not advisory.** A cancellation that
   doesn't free GPU within ~2 s is a bug, not a slow path. Verified by
   the GPU-memory readback acceptance test.
5. **`violation_group_id` is deterministic from inputs.** No opaque
   UUIDs. Cross-session resumption depends on this.
6. **Schema versioning is asymmetric.** `additionalProperties: true`
   on responses, `false` on requests. Enum widening requires the
   `unknown` fallback contract.
7. **The bridge ships behind a stdio-only MCP transport in v0.4.**
   HTTP-streamable MCP is out of scope until v0.5 — different
   stdout-contamination story, different cancellation primitives.

## Verification

### Unit / contract tests

- `tests/test_bridge/test_schema.py`:
  - JSON-Schema document validates a fixture `violations[]` response.
  - `additionalProperties: false` on requests rejects unknown fields.
  - `additionalProperties: true` on responses accepts new fields.
  - Enum-widening fallback: a response with
    `coordinate_frame: wafer_global` does not crash a pinned consumer.
- `tests/test_bridge/test_dedup.py`:
  - 100 violations × 4 tiles × 12 group collisions: assert 12 distinct
    canonical records, 88 shadows, group IDs deterministic across runs.
  - `optimize(violation_id=shadow_record)` resolves to canonical group
    fix; `scope="record_only"` opts out.
- `tests/test_bridge/test_capabilities.py`:
  - `tools/list` response includes `engine_paths` matrix.
  - `conditional` precondition triggers `degraded_to_aggregate: true`
    when layout area exceeds the cutoff.

### G1 acceptance — stdio hygiene

```bash
openlithohub eval --report-level=detailed --fixture tests/fixtures/asap7_small.oas \
  | python -c "import sys, json; [json.loads(l) for l in sys.stdin if l.strip()]"
```

Exit 0 = pass. One unparseable line = pass becomes fail; bridge merge
blocked. Runs in CI on every PR that touches `cli/eval_cmd.py`,
`benchmark/report.py`, or any engine path.

### G2 acceptance — long-run + cancel

```python
def test_cancel_frees_gpu():
    job = mcp_client.call("evaluate", {"layout": large_fixture, "report_level": "detailed"})
    time.sleep(0.5)  # let engine start
    mem_before = nvidia_smi_memory()
    mcp_client.disconnect()
    time.sleep(2.0)
    mem_after = nvidia_smi_memory()
    assert mem_after < mem_before - 100_000_000  # ≥100 MB freed
    assert engine_job_status(job.job_id) == "cancelled"
```

CPU-only fallback for CI: assert process RSS drops, assert the tile
loop's iteration counter stops advancing, assert job state is
`cancelled` not `running`.

### G3 acceptance — bridge is a pass-through

```python
def test_byte_identity_through_paths():
    fixture = "tests/fixtures/asap7_small.oas"
    direct = subprocess.run(["openlithohub", "eval", "--report-level=detailed",
                             "--format=json", fixture], capture_output=True).stdout
    in_proc = mcp_client_inproc.call("evaluate", {"layout": fixture, "report_level": "detailed"})
    sub_proc = mcp_client_subproc.call("evaluate", {"layout": fixture, "report_level": "detailed"})
    assert json.loads(direct) == in_proc["result"] == sub_proc["result"]
```

Modulo the MCP envelope, every path produces identical structured
content. No string substitution. No re-keying. No float reformatting.

### Falsifiable acceptance fixture (cited from issue #11)

`tests/fixtures/mcp_acceptance/100x4tiles_12collisions.json`:

- 100 violations across 4 tiles
- 12 `violation_group_id` collisions
- 2 `scope` values (`group`, `record_only`)
- = 24 acceptance assertions on a single canned response

Replayable, falsifiable, exercises every failure surface this RFC
addresses.

## Out of scope

- **HTTP-streamable MCP transport.** Different stdout/stderr semantics,
  different cancellation primitives. v0.5 at earliest.
- **Multi-session shared cache.** Each MCP connection has its own
  `violation_id` cache; no cross-connection reuse. Cross-session
  resumption is via deterministic re-derivation, not shared state.
- **Authenticated MCP servers.** Local stdio assumes a trusted client
  (the user's own Claude Desktop / Cursor process). Network-exposed
  authenticated MCP is a separate RFC.
- **`simulate` and `synth` tool surfaces.** The first bridge ships
  `evaluate` and `optimize` only. Adding more tools is incremental and
  doesn't change the gates.
- **Streaming partial `violations[]`.** `evaluate` is request/response;
  long ops use `job_id` + `poll`, not partial-array streams. Partial
  streaming is an API-shape question revisited in v0.5.
- **Bridge-side observability.** Tracing, metrics, and structured
  logging for the bridge process itself are punted to the same RFC as
  HTTP-streamable transport.

## Implementation

Phased — the CLI prerequisite ships first, the bridge ships second.
Both are independently useful (the CLI work helps non-MCP consumers
like CI annotators and DRC visualizers).

### Phase 1 — CLI prerequisite (blocks bridge)

- `src/openlithohub/cli/eval_cmd.py`: `--report-level={aggregate, detailed}`
  flag. Default `aggregate` (no leaderboard regression).
- `src/openlithohub/benchmark/report.py`: emit `violations[]` when
  detailed. `schema_version` stamp on every JSON response. Aggregate
  path bit-identical to today.
- `src/openlithohub/benchmark/compliance/{drc,mrc}.py`: stop discarding
  per-violation records at the aggregator boundary. Add
  `coordinate_frame`, `transforms[]`, `dedup_provenance` to each
  record.
- `src/openlithohub/benchmark/compliance/dedup.py` (new): post-tile
  aggregation, deterministic `violation_group_id` derivation,
  `canonical | shadow` tagging.
- `src/openlithohub/benchmark/compliance/rule_deck.py`: per-rule
  `dedup_radius_nm` field.
- `docs/api/violation-schema.md` (new): JSON-Schema document, versioned.
- `tests/test_benchmark/test_report_detailed.py`: round-trip, dedup,
  group-ID determinism, bit-identical aggregate path.

### Phase 2 — bridge (after Phase 1 lands)

- `src/openlithohub_mcp/` (new package, in same repo for now):
  - `transport.py`: stdio MCP transport, owns `stdout`, redirects
    engine logger before install.
  - `tools.py`: `evaluate`, `optimize` tool definitions, JSON-Schema
    validation.
  - `session.py`: per-connection cache, `violation_id` resolution,
    `cancel_evicted` / `layout_not_pinned` error codes.
  - `cancel.py`: `CancellationToken`, MCP disconnect / cancel
    notification → token.
  - `capabilities.py`: per-engine-path matrix, served on `tools/list`.
- `src/openlithohub/workflow/{tiling,process_window}.py`: thread
  `cancel_token` kwarg through `for tile in ...` / `for corner in ...`
  loops. Token-aware kernels in `models/{neural_ilt,levelset_ilt}.py`.
- `tests/test_bridge/`: schema, dedup, capabilities, G1/G2/G3
  acceptance gates, fixture-based falsifiable check.
- `docs/cli-reference.md`: document the MCP server entry point
  (`openlithohub mcp` or `python -m openlithohub_mcp`).

### Phase 3 — client smoke transcripts (after Phase 2)

- `examples/mcp/claude-desktop-config.json`: drop-in config for
  Claude Desktop.
- `examples/mcp/transcripts/`: replayable JSON-RPC transcripts for
  representative agent flows (eval → optimize, eval → cancel,
  eval → poll → optimize).
- `docs/lithography-for-ai-engineers.md`: link the transcripts as the
  worked example for "how an agent uses OpenLithoHub."

## Acknowledgements

The schema decisions in this RFC were shaped by the design discussion
on [issue #11](https://github.com/OpenLithoHub/OpenLithoHub/issues/11):

- **`@m13v`** — flagged cancel-on-disconnect as the v2 rewrite vector;
  raised cancellation to a P0 acceptance gate (G2).
- **`@Ilya0527`** — stdio contamination as a hard rule (G1); the
  thin-shim-vs-domain-translator fork that froze G3; the
  `coordinate_frame` enum + transform chain; tile-boundary dedup with
  `violation_group_id` and the canonical/shadow distinction;
  `conditional` as the fourth capabilities-matrix value;
  deterministic group-ID derivation; the falsifiable
  100×4×12-collisions fixture.
- **`@reaworks-ops`** — the two-gate framing (stdio hygiene + long-run
  control) that organized the design before G3 was added.
