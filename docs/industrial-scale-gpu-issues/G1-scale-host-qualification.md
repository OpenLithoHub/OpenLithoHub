# GPU Issue G1 — Scale Host Qualification (3×RTX 3080)

## Purpose

Qualify the scale host: driver/CUDA/cuDNN/PyTorch, KLayout, fixture
preparation, single-GPU correctness, and the 3-GPU topology — BEFORE
any scale measurement authority is attempted.

This issue is NON-AUTHORITATIVE and publishes no performance claims.

## Prerequisite

```text
FROZEN_SCALE_COMMIT:
<FULL_40_HEX_SHA>
```

Fill only with the post-freeze scale-track merge commit. Do not
substitute `main` / `latest` / any other commit.

Blocked by:
- the scale-track freeze (charter + harness + verifier + dry run merged).

Frozen fixtures (verified 2026-09-25):

```text
PDB commit:      9e1e3399b1b707f26fee853bce1ff91ab466ce24
Ibex layer:      66:44
Microwatt GDS:   b0253af06f35d1a8b11c2a47f70aac89f33be53e8f28da844b35c2ad6cc92a6d (554,770,926 bytes)
Microwatt layer: 66:44 (audited: 27.5M instances spanning 98.8% x 99.4% of die)
```

## Execution

Follow `docs/industrial-scale-gpu-runbook.md` steps 1–4 EXACTLY
(environment, fixture preparation, preflight) and then run the SINGLE
correctness smoke:

```bash
python benchmarks/industrial-scale/run_scale_benchmark.py \
  --fixture-manifest benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json \
  --gds benchmarks/results/industrial-scale/fixtures/ibex/ibex.gds \
  --lanes b --windows <FROZEN_SCALING_WINDOW> \
  --device cuda:0 --device-backend cuda --gpu-count 3 --worker-count 3 \
  --forward-profile P1_FINITE_SUPPORT --sink memmap_npy \
  --tile <FROZEN_TILE> --halo <FROZEN_HALO> --microbatch <FROZEN_MICROBATCH> \
  --repeats <FROZEN_REPEATS> --warmup <FROZEN_WARMUP> \
  --provisional
```

Acceptance: `status = SUCCESS`, `correctness_witness_pass = true`,
worker topology covers all tiles, per-GPU peaks recorded, tracked tree
clean after the run.

## Attach

`nvidia-smi -q`, `nvidia-smi topo -m`, `pip freeze`, preflight output,
run workspace archive, final clean-tree confirmation.

## Exit states

```text
PASS: SCALE HOST QUALIFIED — G2/G3 MAY PROCEED
FAIL: attach evidence and stop; a source fix re-freezes everything
```
