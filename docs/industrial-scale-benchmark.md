# Industrial Scale Benchmark — Charter (FROZEN protocol definitions)

Independent scale/multi-GPU track of the PR-G benchmark family. This
charter freezes the benchmark's problem definition, lanes, forward
profiles, scale ladder, status vocabulary and authority boundaries
BEFORE any GPU measurement. The protocol authority core lives in
`src/openlithohub/benchmark/industrial_scale.py`; fixture preparation in
`scripts/prepare_industrial_scale_fixture.py`; the harness in
`benchmarks/industrial-scale/` (S2+).

## Authority isolation (independent namespace)

This track never touches Industrial Benchmark v1.1, Industrial Benchmark
v2, or P-054/B04 authority:

```text
schema:            OpenLithoHub.industrial-scale-benchmark.v1
run config:        OpenLithoHub.industrial-scale-run-config.v1
fixture manifest:  OpenLithoHub.scale-fixture.v1
results root:      benchmarks/results/industrial-scale/
authority_scope:   SCALE_CHARACTERIZATION
not_v1_1_authority = true
not_v2_authority   = true
not_foundry_calibrated = true
commercial_tool_comparison = NONE
```

Until the formal scale protocol is frozen and executed on a CUDA host,
every result is:

```text
PROVISIONAL
DEVELOPMENT DRY RUN
NOT PERFORMANCE AUTHORITY
```

## Problem definition

The scale benchmark does NOT answer:

```text
Is OpenLithoHub faster than commercial OPC/ILT tools?
Does it reach foundry sign-off?
Does it have production lithography accuracy?
```

It answers exactly:

```text
1. How large can a real routed vector layout be processed?
2. Is full-layout rasterization avoided?
3. Do host RSS / VRAM stay bounded?
4. What is the single-GPU throughput?
5. How does 1/2/3-GPU scaling behave?
6. Where is the bottleneck: geometry / transfer / forward / sink?
```

## Two lanes — never mixed into one number

### Lane A — Large-Layout Streaming Scale

```text
vector GDS → exact-vector query → streaming tiles
→ bounded forward → out-of-core sink
```

Core metrics:

```text
layout_equivalent_pixels   vector_gds_bytes      end_to_end_wall_s
peak_host_rss_bytes        peak_vram_bytes       n_tiles
n_forward_batches          screened_tiles        forwarded_pixels
output_bytes               throughput_gpx_s
```

### Lane B — Multi-GPU Scaling

Same fixture, same tile geometry, same forward kernel, same output
semantics at 1 / 2 / 3 GPUs:

```text
T1, T2, T3
speedup_2 = T1 / T2        speedup_3 = T1 / T3
efficiency_2 = speedup_2 / 2    efficiency_3 = speedup_3 / 3
```

Every run also records the stage breakdown:

```text
geometry_wall_s  queue_wall_s  h2d_wall_s  forward_wall_s
d2h_wall_s       sink_wall_s   end_to_end_wall_s
per_gpu_peak_vram   per_gpu_forward_tiles
```

Kernel-only time must never be headlined alone.

## Forward profiles

| Profile | Purpose | Headline eligibility |
|---|---|---|
| `P0_IDENTITY` | scheduler / sink / ownership / I/O smoke | NEVER a performance headline |
| `P1_FINITE_SUPPORT` | deterministic finite-support kernel: large-layout throughput, batching, multi-GPU scaling, correctness parity | main profile |
| `P2_HOPKINS_BOUNDED` | compute-heavy characterization on bounded windows/tile sets only | optional stress, never first acceptance |

P1 is deterministic, has a bounded receptive field, bounded memory, a
CPU-checkable small-witness parity, and is GPU-friendly. P2 always uses
a fixed grid, fixed kernels, fixed optical parameters, the same tile
size and dtype, and the same correctness witness.

## Scale ladder and statuses

Development ladder: `4096², 8192², 16384², 32768², 65536², 131072²,
262144², full-die`. No host has to run every rung, and no rung may be
silently skipped — the status vocabulary is:

```text
SUCCESS | FAILED | FAILED_CORRECTNESS
NOT_RUN_MEMORY_POLICY | NOT_RUN_TIME_POLICY
NOT_RUN_ENVIRONMENT | UNSUPPORTED
```

## Full-layout firewall

The large-layout lane must never construct an H×W resident raster of the
layout. If code attempts a full-layout input or output tensor, the run
FAILS. Allowed: tile-local tensors, bounded batches, memmap output,
streaming Manhattan/OASIS output.

## Fixture authority

Fixtures come from the frozen PDB lineage
(`SJTU-YONGFU-RESEARCH-GRP/PDB-Physical-Design-Database` @
`9e1e3399b1b707f26fee853bce1ff91ab466ce24`):

* **Ibex** — the v1.1/v2 authority lineage (`layout/sky130hd/ibex/ibex.gds`,
  top cell `ibex_core`).
* **Microwatt** — split-chunk stress fixture
  (`layout/sky130hd/microwatt/`, top cell `microwatt`); chunks are
  canonically sorted, concatenated, re-opened and verified before use.

Every fixture is prepared by
`scripts/prepare_industrial_scale_fixture.py` into a validated
`OpenLithoHub.scale-fixture.v1` manifest: SHA-256, byte size, top cell,
DBU, exact bbox, full layer enumeration, explicit selected layer, die
size in pixels, and the DENSE FLOAT32 RASTER EQUIVALENT derived from
bbox/pixel. That equivalent is a derived number — it must never be
worded as processed bytes.

The benchmark layer is an explicit per-design decision frozen into the
manifest. Microwatt must never inherit Ibex's `66:44` by assumption.

## Claim boundaries

Never:

```text
"processed 40 TB of GDS"            (if it is a dense equivalent)
"3× GPU = unified VRAM"             (device memory is not unified)
"full-chip Hopkins sign-off"
"foundry validated"
"commercial-tool competitive"
```

Allowed future headline shapes (only with formal, verifier-closed
artifacts):

```text
real routed layout streamed end-to-end without constructing a
resident full-layout raster

peak host RSS remained bounded across the declared scale ladder

N-GPU end-to-end throughput speedup under the frozen
tile-sharded protocol
```

## Environment lock

Every run — including CPU development dry runs — records a complete
`scale environment lock` (device inventory with UUID/PCI bus/VRAM/compute
capability on GPU hosts, driver, CUDA/cuDNN builds, TF32 state, and the
`nvidia-smi topo -m` facts). The lock hash enters the run identity.
CPU-worker-emulation runs can never satisfy a formal cuda-backend run.
