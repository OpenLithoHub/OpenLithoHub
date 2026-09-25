# Industrial Scale — CPU Development Dry Run Record (S12/S13)

> **PROVISIONAL / DEVELOPMENT DRY RUN / NOT PERFORMANCE AUTHORITY.**
> This record proves the frozen scale protocol executes end to end on a
> CPU-only host. Every number below is a development artifact. Nothing
> here is v1.1 authority, v2 authority, a headline claim, or comparable
> to any external tool. Multi-worker CPU emulation proves scheduler
> semantics ONLY — it is never multi-GPU performance.

## Scope executed (freeze-gate items S12)

Fixture: **v1.1-lineage Ibex center crop** (`ibex_crop_16384`, 16384² px
@ 1 nm/px, layer 66:44 among 32 enumerated layers), re-verified through
`scripts/prepare_industrial_scale_fixture.py` rather than trusted from
history:

```text
fixture sha256:    9b1790b9ad8ce3f878e67c53f8a51be912076b6a0291fe70dd6ddcec534699f9
top cell:          IBEX_CROP_16384
die size px:       16384 × 16384
dense equivalent:  1,073,741,824 bytes (1 GiB hypothetical full float32
                   raster — DERIVED, never materialized)
FIXTURE PREP: PASS
```

Commands (all `--device cpu`, `--formal` deliberately NOT passed):

```bash
# A: Lane A ladder, P1_FINITE_SUPPORT
run_scale_benchmark.py --lanes a --windows 4096,8192 --tile 1024 \
  --halo 64 --microbatch 8 --repeats 3 --warmup 1 \
  --forward-profile P1_FINITE_SUPPORT --sink memmap_npy
# B3: Lane B, 3 CPU-emulated workers
run_scale_benchmark.py --lanes b --windows 4096 --worker-count 3 …
# B1: Lane B baseline, 1 worker
run_scale_benchmark.py --lanes b --windows 4096 --worker-count 1 …
# P0: identity plumbing
run_scale_benchmark.py --lanes a --windows 4096 \
  --forward-profile P0_IDENTITY --repeats 2 --warmup 1 …
```

## Dry-run acceptance results

| Gate | Result |
|---|---|
| fixture prepare PASS | ✅ `FIXTURE PREP: PASS` |
| run identity + environment lock PASS | ✅ four distinct identities |
| streaming PASS (all runs exit 0) | ✅ 4/4 workspaces |
| no full-layout tensor (memmap sink only) | ✅ out-of-core `.npy` outputs |
| correctness witness PASS | ✅ every SUCCESS row |
| 1-worker / 3-worker output parity | ✅ **byte-identical** `w4096-r2.npy` |
| artifact family complete (8 members) | ✅ in each `runs/<id>/family/` |
| verifier PASS (structural tier) | ✅ 4/4 families |
| formal tier refuses CPU provisional | ✅ `--require-formal` → FAIL (expected) |
| claim generator emits only provisional | ✅ `PROVISIONAL_INTERNAL`, nothing admitted |
| headline firewall (`ISC-*` banned in README) | ✅ generator `--check` passes on clean READMEs |

## Observed facts (provisional, Apple M5 Pro CPU — NOT authority)

* Lane A 8192² rung: 64 tiles, median ≈ 2.59 s end-to-end, peak host
  RSS ≈ 0.71 GiB — flat versus window size, as the charter predicts.
* Lane B 3 workers: shard plan 6/5/5 across workers, resident reorder
  watermark ≤ 3 (the backpressure cap), and outputs **bit-identical**
  to the 1-worker run.
* P0_IDENTITY rows are emitted `headline_eligible = false`.

## S13 status (Microwatt): FROZEN

The real Microwatt fixture freeze landed with the scale fixture
authority (see `benchmarks/results/industrial-scale/fixtures/microwatt/`
and the GPU runbook): reconstructed GDS sha256
`b0253af06f35d1a8b11c2a47f70aac89f33be53e8f28da844b35c2ad6cc92a6d`
(554,770,926 bytes from 11 canonically ordered chunks, byte-verified
against the pinned PDB tree), top cell `microwatt`, DBU 1.0 nm,
bbox [0, 0, 3020000, 3610000] DBU, 41 layers audited, selected layer
**66:44** (audited: 27.5M instances spanning 98.8% × 99.4% of die —
the same li1 class as the v1.1/v2 lineage), dense float32
raster-equivalent 43,608,800,000,000 bytes (derived, never
materialized). Microwatt still must not enter any benchmark before the
GPU-phase issues execute the frozen commands.
