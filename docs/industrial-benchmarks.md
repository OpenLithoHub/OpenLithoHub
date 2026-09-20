# Industrial Benchmarks (v1)

<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

This page defines **Industrial Benchmark v1**: the measurement protocol
OpenLithoHub uses to back any industrial-facing performance statement.
Everything here is designed around one rule:

> **Every public number must be regenerable from a checked-in artifact.
> No hand-written performance claims.**

- Harness: [`benchmarks/industrial/run_industrial_benchmark.py`](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/benchmarks/industrial/run_industrial_benchmark.py)
- Artifacts: `benchmarks/results/industrial/*.json` (schema
  `OpenLithoHub.industrial-benchmark.v1`)
- Generated claims: [`docs/generated/industrial-claims.md`](generated/industrial-claims.md)
- Claim generator: `scripts/generate_industrial_claims.py` (use `--check`
  to fail CI when the README quotes drift from artifacts)

## What it measures

Four questions, in priority order:

1. **Runtime** — dense full-raster vs tiled-raster vs exact-vector
   streaming vs selective streaming (certified empty-context screening),
   on identical inputs. Runtime rows report `parse_wall_s`,
   `execution_wall_s` and `end_to_end_wall_s` separately; speedups are
   `baseline_median / candidate_median` over ≥ 5 repeats with p10/p90
   spread. Best-of-N is never reported.
2. **Peak memory** — fresh worker process per timed run, `ru_maxrss`
   medians. Dense input is `O(W·H)`; streaming memory is
   `O(tile_area + active_batch)`. Where dense is infeasible under the
   harness memory policy, that is recorded as `INFEASIBLE_STRUCTURAL`
   rather than skipped silently.
3. **Quality** — registered models on real-layout tiles with the *same*
   Hopkins SOCS optical model, same resist threshold, same pixel size,
   same input, same metric implementations (EPE, wafer EPE, L2, PV Band,
   MRC, shot count). Quality is always reported next to runtime; a fast
   method that loses quality is shown as a trade-off, not hidden.
4. **Scaling** — the largest layout streamed end-to-end at bounded
   memory, and structural feasibility arithmetic for dense rasterization
   (die pixels × bytes vs physical RAM).

## Dataset and provenance

Real routed silicon layout, not a synthetic showcase:

| Field | Value |
|---|---|
| Design | `ibex_core` — 32-bit RISC-V CPU core, two-stage pipeline |
| PDK | `sky130hd` (OpenROAD-flow-scripts flow) |
| Source | `SJTU-YONGFU-RESEARCH-GRP/PDB-Physical-Design-Database`, commit `9e1e3399b1b707f26fee853bce1ff91ab466ce24` |
| Files | `layout/sky130hd/ibex/ibex.gds` + `ibex.json` (15,515 cells, 0.579 std-cell utilization, 305,113 µm² die) |
| Layer | `66:44`, rasterized at 1 nm/px by the exact-vector run source |
| Integrity | fixture SHA-256 recorded in every artifact; measurement fingerprints mix dataset hash + harness arguments |

The die is 555,355 nm per side. At 1 nm/px a dense raster would need
**1.23 TB** — structurally infeasible on benchmark hardware (48 GB), and
recorded as such.

## Benchmark protocol

- Every timed run executes in a **fresh worker process**, so peak RSS is
  attributable to that run alone and torch/thread state never leaks
  between rows.
- Ladder sizes are center crops of the real die: 4096², 8192², 16384²,
  32768² px, plus a streaming-only 65536² row. Dense modes are guarded by
  a memory policy (`--dense-max-bytes`, default 30 GB): sizes whose dense
  input+intermediates cannot fit are *measured as infeasible*, not
  skipped.
- Claim-bearing rows use ≥ 5 repeats (median / p10 / p90). The largest
  scaling-only rows record fewer repeats and carry their own `repeats`
  field; headline claims are derived only from full-repeat rows.
- Correctness witnesses gate the whole benchmark: a 512² synthetic
  witness and a 1024² real-layout witness verify that selective
  streaming output matches the dense reference (`max_abs_error ≤ 2e-6`).
- Work accounting must close (`accounted_pct == 100`) on streaming rows.

## Physics claim scope

Two forward models are used, and the artifacts say which:

- **Architecture ladder** (runtime + memory): a deterministic 9×9
  separable finite-support zero-preserving blur — identical to the B04
  INC28 benchmark so results stay comparable. This measures work
  avoidance and memory scaling, *not* lithography physics.
- **Quality stage**: built-in Hopkins SOCS (193 nm, NA 1.35, σ 0.7, 24
  kernels, CTR threshold 0.225) with identical parameters for every
  compared method. Numbers are benchmark-relative and
  **NOT_FOUNDRY_CALIBRATED**.

No commercial tool (Calibre, Tachyon, cuLitho) is benchmarked or
compared anywhere in this harness.

## Claim levels and firewall

Claims generated from artifacts carry a provenance level:

- `REPRODUCED_INTERNAL` — measured by this repo's harness on documented
  hardware (everything currently produced by Industrial Benchmark v1)
- `PUBLIC_BENCHMARK` — same protocol, executed on public third-party
  infrastructure
- `THIRD_PARTY_REPRODUCED` — independently reproduced by an external party
- `FOUNDRY_CALIBRATED` — calibrated against wafer/SEM data (does not exist
  for this project)

Headline admission rules enforced by the generator: runtime ratios are
headline-eligible only at ≥ 1.1×; memory claims at ≥ 20% reduction;
quality claims at ≥ 5% reduction of the bad metric; surrogate-vs-levelset
runtime ratios only when quality stays within a 10% degradation tolerance
at the matched budget. Everything else is published as a scoped fact.

Standing firewall (enforced culturally and by the claims generator):

- No foundry qualification without wafer/SEM calibration.
- No commercial-tool speedup or quality claim without a matched external
  benchmark.
- No neural-model quality claim from degenerate blank masks.
- No synthetic benchmark marketed as production wafer performance.
- No plugin research backend marketed as third-party validated.
- No speedup claim without matching quality/tolerance context.

## Reproducing

```bash
# One command produces all industrial benchmark artifacts:
python benchmarks/industrial/run_industrial_benchmark.py \
  --gds /path/to/ibex.gds \
  --out benchmarks/results/industrial

# Regenerate public claims from the artifacts:
python scripts/generate_industrial_claims.py

# Fail if README-quoted numbers drift from artifacts:
python scripts/generate_industrial_claims.py --check
```

Stages are individually checkpointed under
`benchmarks/results/industrial/checkpoints/` — an interrupted run resumes
from completed rows instead of restarting. Progress prints one line per
timed run.

## Known limitations (measured)

- **Exact-vector window/screen queries** build a per-row polygon index on
  first touch; cost grows with tile rows × layout polygons. On the
  reference machine a 32768² die-center tile costs ~56 s of screening
  before any forward work, and a full-die screening survey is estimated
  (mean per-tile × 289 grid tiles, labeled `ESTIMATE_NOT_MEASUREMENT` in
  the artifact) at roughly an hour of CPU. A spatial index over runs is
  the highest-leverage next step for die-scale exhaustive surveys.
- **GPU tiers are NOT yet measured.** The reference hardware for v1 is a
  CPU machine; any GPU number in older docs is historical and not
  artifact-backed (see `docs/self_hosted_deployment.md`).
- Surrogate-ILT's on-the-fly training budget is reduced in the harness
  (16 samples × 3 epochs) relative to library defaults, which are
  intractable at 1024 px on CPU; surrogate comparisons are therefore
  scoped to the recorded budget, never generalized.
