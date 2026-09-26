# GPU Issue G1-4090 — Scale Host Qualification (1×RTX 4090)

> Non-authoritative qualification. NO performance claims.

## Prerequisite

```text
FROZEN_SCALE_COMMIT:
<FULL_40_HEX_SHA>
```

Fill with the current Scale frozen source (`NEW_SCALE_4090_FROZEN_SOURCE`).
Do not substitute `main` / `latest` / any other commit.

## Frozen fixtures (verified 2026-09-25)

```text
PDB commit:      9e1e3399b1b707f26fee853bce1ff91ab466ce24
Ibex layer:      66:44
Microwatt GDS:   b0253af06f35d1a8b11c2a47f70aac89f33be53e8f28da844b35c2ad6cc92a6d (554,770,926 bytes)
Microwatt layer: 66:44 (audited: 27.5M instances spanning 98.8% × 99.4% of die)
```

## Expected host

```text
1 × NVIDIA GeForce RTX 4090 (24 GB)
Linux, CUDA-enabled PyTorch, cuDNN, KLayout
```

## Procedure

1. Follow `docs/industrial-scale-gpu-runbook.md` steps 1–3 (environment,
   both fixture preparations, frozen facts above).
2. Scale preflight (must PASS):
   ```bash
   python scripts/preflight_industrial_scale.py \
     --ibex-manifest benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json \
     --gds-ibex benchmarks/results/industrial-scale/fixtures/ibex/ibex.gds \
     --microwatt-manifest benchmarks/results/industrial-scale/fixtures/microwatt/fixture-manifest.json \
     --gds-microwatt benchmarks/results/industrial-scale/fixtures/microwatt/microwatt.gds \
     --device cuda:0 \
     --expected-commit <FROZEN_SCALE_COMMIT> \
     | tee measurement-logs/scale-preflight.txt
   ```
3. Single-GPU bounded qualification smoke (provisional, non-authoritative):
   ```bash
   python benchmarks/industrial-scale/run_scale_benchmark.py \
     --fixture-manifest benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json \
     --gds benchmarks/results/industrial-scale/fixtures/ibex/ibex.gds \
     --lanes c --windows 8192 \
     --device cuda:0 --device-backend cuda --gpu-count 1 --worker-count 1 \
     --forward-profile P1_FINITE_SUPPORT --sink memmap_npy \
     --tile 1024 --halo 64 --microbatch 8 \
     --repeats 5 --warmup 2 \
     --microbatch-ladder 8 \
     --provisional
   ```

Acceptance: `SCALE PREFLIGHT: PASS`; smoke `status = SUCCESS`,
`correctness_witness_pass = true`, GPU identity/VRAM facts recorded,
tracked tree clean after the run.

Fail-closed on: no CUDA GPU, cuda:0 unavailable, GPU identity materially
different from the recorded environment, fixture hash mismatch, source
commit mismatch, dirty tree.

## Attach

`nvidia-smi -q`, `nvidia-smi topo -m`, `pip freeze`, preflight output,
smoke workspace archive, final clean-tree confirmation.
