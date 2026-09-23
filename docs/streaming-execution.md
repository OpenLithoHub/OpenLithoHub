# Streaming Execution (PR-C)

OpenLithoHub's product execution path has a single spine: one planner
decides **dense vs streaming** for every optimize request, and both the
CLI (`optimize run`), the Python API (`LitheEngine.optimize`), the
synchronous HTTP endpoint (`POST /v1/optimize`) and the async job API
(`POST /v1/jobs/optimize`) go through it. There is no per-surface
execution topology and no silent fallback.

## The planner

`openlithohub.workflow.execution.plan_execution` returns a typed decision:

| Reason | Meaning |
| --- | --- |
| `DENSE_SMALL_LAYOUT` | `auto` under the dense memory policy: dense is allowed. |
| `DENSE_REQUESTED_EXPLICITLY` | The caller explicitly forced `--execution-mode dense` (documented override, including above the policy). |
| `STREAMING_SUPPORTED` | Explicit streaming request on a fully supported combination. |
| `STREAMING_REQUIRED_BY_MEMORY_POLICY` | `auto` above the dense memory policy with a fully supported streaming combination. |
| `STREAMING_UNSUPPORTED_INPUT` / `..._OUTPUT` / `..._MODEL` | The combination cannot stream. |

The two streaming-unsupported reasons carry honest, specific detail. A
request may still **explicitly** stream over in-memory endpoints
(`LitheEngine.optimize(..., execution_mode="streaming")`); the plan then
records that resident memory stays O(layout) because the caller already
materialises input and output — the exact-core scheduling semantics still
apply.

## Dense memory policy

```text
OPENLITHOHUB_MAX_DENSE_BYTES   # max bytes of one fp32 full raster the
                               # dense path may materialise under `auto`;
                               # default 8 GiB; 0 = unlimited
```

This is **policy, not benchmark authority**. Under `auto`:

```text
estimated raster bytes <= policy and supported   -> dense
estimated raster bytes >  policy and supported   -> streaming
estimated raster bytes >  policy and unsupported -> FAIL CLOSED (HTTP 400 /
                                                    CLI error), never a
                                                    silent full-chip raster
```

## Streaming branch

The streaming branch is the RFC 0008 authority
(`openlithohub.streaming.run_streaming`): windowed reads of core+halo,
the model applied per window, only the trusted core committed to the
sink, per-tile tensors discarded. It never calls the dense loader or
`stitch_tiles` and never retains per-tile results — this is enforced by
structural hostile tests, not by convention.

| | Supported | Notes |
| --- | --- | --- |
| Input `.npy` | yes | `np.load(mmap_mode="r")` — out-of-core, header-only probing. |
| Input `.gds` / `.oas` | yes* | Exact vector scanline source (`KLayoutAlignedRunSource`); parser-only, no rasterization. *Requires integer DBU-per-pixel and pixel-aligned geometry; anything else is streaming-unsupported (dense under policy, fail-closed above it). |
| Input `.pt` | no | Inherently dense representation; explicit dense only (under policy). |
| Output raster artifact (`.npy`) | yes | `MemmapTileSink(npy=True)` — disk-backed, self-describing, never fully resident. |
| Output Manhattan OASIS (`vsb`) | yes | `StreamingManhattanTileSink` — each trusted core is converted to owned Manhattan rectangles exactly once; memory is O(core + rectangle count). |
| Output curvilinear OASIS (`mbmw`) | no for large jobs | Explicit dense under the policy; `auto` fails closed above it rather than silently materializing a full output raster. |

The live matrix is served by `GET /v1/capabilities` under `streaming`
and populated from code-owned tables (`workflow.execution.streaming_capability_matrix`),
not from documentation.

## Input rasterization conventions (honest asymmetry)

The dense path rasterizes GDS/OASIS through the historical PIL pipeline
(`data.io.load_layout`, round-to-nearest projection, inclusive polygon
fill). The streaming path uses the exact pixel-center scanline semantics
of the B04 vector authority. Both are internally pinned by tests, but
they differ at polygon boundary rows/cols — typically a one-pixel band.
Dense/streaming **bitwise** equivalence is therefore asserted for `.npy`
inputs (identical input rasters); for GDS/OASIS the streaming artifact is
asserted against the exact-vector reference, and the convention
difference is documented instead of hidden.

## CLI surface

```bash
# planner decides; prints the decision before executing
openlithohub optimize run -i chip.npy -m dummy-identity -o out.npy \
    --execution-mode auto

# force the streaming branch; .npy output = raster artifact
openlithohub optimize run -i chip.gds -m dummy-identity -o out.oas \
    --writer vsb --execution-mode streaming
```

`--num-gpus > 1` keeps the legacy multi-GPU dense parallel path (dense
only by contract).

## What this is not

* Not a new benchmark: Industrial Benchmark v1.1 and the P-054 proof
  surface are untouched. The memory-scaling test added in PR-C is an
  engineering regression check, not a published performance claim.
* Not durability: jobs remain process-local; durable job stores and
  multi-worker execution come later.
* Not foundry sign-off: streaming export has the same fab-orientation
  disclaimers as dense export.
