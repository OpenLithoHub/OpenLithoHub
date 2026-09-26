# GPU Issue G2-4090 — Large-Layout Single-GPU Scale (Lane A)

> Characterization evidence only. Not foundry calibrated, no commercial-tool comparison.

## Prerequisite

```text
FROZEN_SCALE_COMMIT:
<FULL_40_HEX_SHA>

G1-4090: PASS
```

## Frozen fixture facts

```text
Ibex:            layer 66:44 (v1.1/v2 lineage), manifest committed with the scale freeze
Microwatt:       GDS sha256 b0253af06f35d1a8b11c2a47f70aac89f33be53e8f28da844b35c2ad6cc92a6d
                 554,770,926 bytes; top cell microwatt; layer 66:44
                 die 3,020,000 × 3,610,000 px @ 1 nm/px
                 dense float32 raster-equivalent 43,608,800,000,000 bytes
                 (derived — never materialized)
```

## Execution (frozen; do not alter after seeing results)

Runbook **Command A** — Lane A ladder on the single 4090:

```bash
python benchmarks/industrial-scale/run_scale_benchmark.py \
  --fixture-manifest benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json \
  --gds benchmarks/results/industrial-scale/fixtures/ibex/ibex.gds \
  --lanes a \
  --windows 4096,8192,16384,32768 \
  --device cuda:0 --device-backend cuda --gpu-count 1 \
  --forward-profile P1_FINITE_SUPPORT --sink memmap_npy \
  --tile 1024 --halo 64 --microbatch 8 \
  --repeats 5 --warmup 2 \
  --formal \
  2>&1 | tee measurement-logs/lane-a-4090.txt
```

Command C-style Microwatt bounded stress (optional, same frozen protocol,
Microwatt fixture, `--windows 4096,8192`) only after the Ibex ladder is
complete. A `NOT_RUN_MEMORY_POLICY` rung is a VALID outcome — record it;
never shrink a window to make it fit unless that window is in the frozen
ladder.

## Acceptance per SUCCESS row

```text
status = SUCCESS
correctness_witness_pass = true
aggregate_n == repeat_count (== 5)
timing_method = cuda_synchronized
full-die-scale rows use the out-of-core memmap sink only
```

A slow result is valid evidence. Do not rerun with different parameters
to obtain a better number.

## Attach

Run workspaces, `verify_industrial_scale_artifacts.py --require-formal`
PASS per workspace, evidence bundle SHA-256, final clean-tree
confirmation, complete (non-selective) rung report.
