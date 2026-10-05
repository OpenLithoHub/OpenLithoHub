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

Reference GPU host (current campaign): **1×RTX 4090**. The historical
3×RTX 3080 multi-GPU campaign is superseded for measurement; Lane B
multi-GPU remains a retained engineering capability with measurement
DEFERRED.

The scale benchmark does NOT answer:

```text
Is OpenLithoHub faster than commercial OPC/ILT tools?
Does it reach foundry sign-off?
Does it have production lithography accuracy?
```

The current 1×RTX4090 campaign answers exactly:

```text
1. How large a real routed vector layout can be processed with bounded
   host/GPU working memory?
2. Does the pipeline avoid resident full-layout rasterization?
3. What is single-RTX4090 end-to-end throughput on the frozen P1 profile?
4. How does throughput / VRAM usage scale with microbatch size on ONE GPU?
5. Where are the bottlenecks: geometry / H2D / forward / D2H / sink /
   end-to-end?
6. How does the Microwatt stress fixture behave under the same frozen
   single-GPU protocol?
```

The current campaign must NOT claim 1/2/3-GPU scaling, multi-GPU
efficiency, or multi-GPU speedup: only one physical GPU is available.''

## Three lanes — never mixed into one number

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

### Lane B — Multi-GPU Scaling (capability RETAINED, measurement DEFERRED)

Lane B remains implemented end to end (tile-sharded multi-worker
scheduler, exactly-once ownership, bounded queues, CPU-emulated hostile
matrix) as an engineering capability for future multi-GPU hardware.
On the current 1×RTX4090 host its formal measurement is DEFERRED: it
keeps requiring gpu_count >= 2 in the formal blockers, and CPU-emulated
1/2/3-worker runs prove scheduler semantics only — never GPU scaling.
The quantities below are the frozen Lane B definitions for when such
hardware exists:

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

### Lane C — Single-GPU Saturation (current formal lane)

Measures single-GPU P1 throughput saturation while varying ONLY the
microbatch size on one fixed physical GPU (cuda:0, gpu_count = 1); all
other semantics stay fixed.  Frozen setup:

```text
fixture:          Ibex scale fixture, layer 66:44
window:           8192
forward profile:  P1_FINITE_SUPPORT
tile:             1024, halo: 64, dtype: fp32
sink:             memmap_npy
warmup:           2, repeats: 5
microbatch ladder: 1, 2, 4, 8, 16, 32   (FROZEN before any measurement)
baseline:         microbatch = 1
```

Every rung records: microbatch, status, aggregate median/p10/p90/n,
throughput_gpx_s, peak host RSS, VRAM allocated/reserved peaks, n_tiles,
n_forward_batches, output_bytes, correctness witness, timing method.
SUCCESS requires aggregate_n == repeat_count == 5 and the witness pass.
No best-of-N.  The FULL ladder is reported — the baseline is
microbatch = 1 and no rung is suppressed.  Lane C executes on the
single-device production spine (never the multi-worker executor).

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
  (`layout/sky130hd/microwatt/split/`, top cell `microwatt`), FROZEN:
  reconstructed GDS sha256
  `b0253af06f35d1a8b11c2a47f70aac89f33be53e8f28da844b35c2ad6cc92a6d`
  (554,770,926 bytes; 11 chunks, canonically ordered, byte-verified
  against the pinned PDB tree, independently re-concatenated and
  KLayout re-opened).  Die 3,020,000 × 3,610,000 px @ 1 nm/px;
  dense float32 raster-equivalent 43,608,800,000,000 bytes (derived,
  never materialized).  Audited selected layer: **66:44** — 27.5M
  shape instances spanning 98.8% × 99.4% of the die; the same
  sky130hd li1 routed class as the frozen v1.1/v2 lineage.  The choice
  was made by the source-owned rule BEFORE any benchmarking; no
  candidate was benchmarked for the selection.

Every fixture is prepared by
`scripts/prepare_industrial_scale_fixture.py` into a validated
`OpenLithoHub.scale-fixture.v1` manifest: SHA-256, byte size, top cell,
DBU, exact bbox, full layer enumeration, explicit selected layer, die
size in pixels, and the DENSE FLOAT32 RASTER EQUIVALENT derived from
bbox/pixel. That equivalent is a derived number — it must never be
worded as processed bytes.

The benchmark layer is an explicit per-design decision frozen into the
manifest. Microwatt must never inherit Ibex's `66:44` by assumption —
and since PR-V3.1.3 that rule is machine-checked, not prose.

## Parameter authority (per design)

Every design-specific parameter carries a source-owned authority record
(`openlithohub.benchmark.scale_parameter_authority`). No parameter may
be inherited from another design by analogy: the only valid bases are
`lineage` (live-measured for THIS design in the frozen v1.1/v2 lineage)
or `direct_audit` (committed audit evidence re-derived from the frozen
GDS bytes). Anything else is UNRESOLVED and blocks formal use for that
design. CI enforces the consistency of this table, the fixture
manifests, the audit evidence and the numbers quoted in these docs
(`tests/test_benchmark/test_scale_parameter_authority.py`).

| Design | Parameter | Value | Basis | Evidence |
|---|---|---|---|---|
| Ibex | selected layer | `66:44` | `lineage` — the v1.1/v2 authority runs live-measured this GDS on this layer | frozen GDS sha256 `5b706ac4…4664c` (source-owned record; regenerated per #94) |
| Ibex | pixel | 1.0 nm/px | `lineage` — v1.1/v2 measurement grid | — |
| Microwatt | selected layer | `66:44` | `direct_audit` — committed audit re-derived from the frozen GDS bytes | `benchmarks/results/industrial-scale/audits/microwatt-layer-audit.json` |
| Microwatt | pixel | 1.0 nm/px | `lineage` — scale-track protocol grid | — |

Until PR-V3.1.3 the Microwatt selection rationale existed only in
prose; it is now regenerable evidence
(`scripts/audit_industrial_scale_fixture_layer.py`): 27,487,849
flattened shape instances — second densest of 41 non-empty layers by
this metric (densest is 67:44) — spanning 98.8% × 99.4% of the die;
15,342 cell-local GDS elements. The audit also freezes the metric
definitions, so "instances" can never silently change meaning again.

Per-design fixture-tracking semantics are frozen in the same module:
the two Microwatt manifests are committed, so the tracked-tree gate is
a live tripwire on them; the Ibex fixture directory and every GDS byte
are host-local (gitignored), so the same gate is structurally blind to
them and their integrity rests entirely on manifest SHA-256
revalidation. The asymmetry is a protocol property, not an accident of
`.gitignore`.

### Adding a new design — fixed review questions

Before a new design enters the scale track, answer these in writing.
Every answer of the form "we assumed it from <previous design>" marks
an UNRESOLVED parameter until it gets its own authority record:

1. Which parameters did we inherit from the previous design by
   analogy? List them explicitly; each needs `lineage` or
   `direct_audit` evidence or must not be used.
2. What does the tracked-tree gate see for this design's fixture —
   which paths are committed, which host-local? Declare it in
   `fixture_tracking`; `.gitignore` must never imply protocol
   semantics.
3. Does the reference host still fit? Record resource margins per
   design BEFORE the first run: GDS bytes, dense float32
   raster-equivalent, expected memmap output volume, and free
   disk/RAM headroom.
4. Which stale artifacts from earlier eras linger on measurement
   hosts, and which SHA gate rejects them? (Known instance: the
   standin-era 144,566-byte `ibex.gds`, sha256 `9b1790b9…` — no
   authority; `--expected-gds-sha256` is the gate.)
5. Is the operator checklist in
   `docs/industrial-scale-gpu-runbook.md` §9 updated for this design
   before the first GPU handoff?

## Claim boundaries

Never:

```text
"processed 40 TB of GDS"            (if it is a dense equivalent)
"multi-GPU scaling"                 (single-GPU host — Lane B deferred)
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

peak host RSS / VRAM remained bounded across the declared scale ladder

single-GPU microbatch saturation behaviour under the frozen
Lane C protocol
```

## Environment lock

Every run — including CPU development dry runs — records a complete
`scale environment lock` (device inventory with UUID/PCI bus/VRAM/compute
capability on GPU hosts, driver, CUDA/cuDNN builds, TF32 state, and the
`nvidia-smi topo -m` facts). The lock hash enters the run identity.
CPU-worker-emulation runs can never satisfy a formal cuda-backend run.
