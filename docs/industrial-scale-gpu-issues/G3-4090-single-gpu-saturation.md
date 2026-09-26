# GPU Issue G3-4090 — Single-GPU Microbatch Saturation (Lane C)

> Characterization evidence only. This lane is a SINGLE-GPU claim by
> definition — it must never be labeled multi-GPU scaling.

## Prerequisite

```text
FROZEN_SCALE_COMMIT:
<FULL_40_HEX_SHA>

G1-4090: PASS
G2-4090: PASS (the REQUIRED Ibex single-GPU baseline; optional Microwatt
stress targets are not a prerequisite for this issue)
```

## Frozen protocol

```text
fixture:          Ibex scale fixture, layer 66:44
window:           8192
GPU:              cuda:0, gpu_count = 1
forward profile:  P1_FINITE_SUPPORT
tile:             1024
halo:             64
dtype:            fp32
sink:             memmap_npy
warmup:           2
repeats:          5
microbatch ladder: 1, 2, 4, 8, 16, 32   (FROZEN — never change after
                                         observing results)
baseline:         microbatch = 1
```

## Execution

Runbook **Command C-Lane** — the full ladder, unmodified:

```bash
python benchmarks/industrial-scale/run_scale_benchmark.py \
  --fixture-manifest benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json \
  --gds benchmarks/results/industrial-scale/fixtures/ibex/ibex.gds \
  --lanes c \
  --windows 8192 \
  --device cuda:0 --device-backend cuda --gpu-count 1 --worker-count 1 \
  --forward-profile P1_FINITE_SUPPORT --sink memmap_npy \
  --tile 1024 --halo 64 \
  --repeats 5 --warmup 2 \
  --microbatch-ladder 1,2,4,8,16,32 \
  --formal \
  2>&1 | tee measurement-logs/lane-c-4090.txt
```

## Acceptance

```text
every frozen rung present exactly once (no silent missing rung)
each SUCCESS rung: correctness_witness_pass = true
                   aggregate_n == repeat_count == 5
                   median / p10 / p90 present
                   VRAM allocated/reserved facts present
                   timing_method = cuda_synchronized
verifier --require-formal PASS
```

The FULL ladder is reported — never select the best rung and suppress
the others. Derived quantities (throughput gain and wall speedup
relative to microbatch = 1) are maintainer-computed from the artifacts.

## Attach

Run workspace(s), verifier PASS, evidence bundle SHA-256, final
clean-tree confirmation, complete rung report (no selective summary).
