# Industrial Benchmark v2 — Formal Measurement Runbook

Operational runbook for executing the formal v2 measurement on a
controlled NVIDIA host, per the PR-G Phase 2B/2C split. The protocol
authority itself lives in `src/openlithohub/benchmark/industrial_v2.py`
and `benchmarks/industrial-v2/run_v2_benchmark.py`; this page is the
operator procedure.

## Status

```text
Phase 2A (merged):  V2 PROTOCOL READY / FORMAL GPU MEASUREMENT PENDING
Phase 2B:           freeze the clean measurement commit (this page)
Phase 2C:           canonical artifacts + generated claims + README
```

## Core rules

```text
No code changes during formal measurement.
No canonical family without all mandatory tiers PASS.
No claim without verifier closure.
No README number without generated admission.
```

Any fix during a run → new commit → new run identity → **restart the
formal run**. Never continue across a code change.

## Host requirements

* One NVIDIA CUDA-capable GPU (single GPU is sufficient; multi-GPU is
  not required for initial v2 authority).
* Stable driver; VRAM adequate for the selected Tier B/C window.
* ≥ 16 GiB host RAM, ≥ 16 GiB free disk.
* KLayout Python API installed; real routed Ibex GDS fixture available.

## Procedure

1. **Clone the exact measurement commit** (the Phase 2B freeze; record
   `git rev-parse HEAD`) and confirm `git status --porcelain` is empty.
2. **Isolated environment**: fresh venv, install the host's CUDA-enabled
   PyTorch build, then `pip install -e ".[server,workflow]"`. Capture
   `pip freeze` and the torch/CUDA facts. Never edit dependencies
   mid-run.
3. **Fixture identity**: `sha256sum ibex.gds` — the same real routed
   Ibex lineage as v1 (`benchmarks/results/industrial/fixtures/`
   provenance). Never substitute a different GDS mid-run.
4. **Preflight** (must print `PREFLIGHT: PASS`, otherwise stop):
   ```bash
   python scripts/preflight_industrial_v2.py --gds /path/to/ibex.gds --device cuda:0
   ```
5. **Save operator evidence** (preflight output, `nvidia-smi -q`, git
   status, commit) under `measurement-logs/`. Canonical facts remain
   harness-owned — operator logs are context, never claim inputs.
6. **Run the formal tiers through the harness only** — never ad-hoc
   Python, never shell `time`:
   ```bash
   python benchmarks/industrial-v2/run_v2_benchmark.py \
       --tiers a,b,c --gds /path/to/ibex.gds --device cuda:0 \
       --dtype fp32 --repeats 5 --warmup 2 --batch 8 --tile 1024 \
       --formal
   ```
   The harness owns the run workspace
   (`runs/<run_identity>/`), fresh-process repeats, synchronized CUDA
   timing, allocated/reserved GPU peaks, correctness witnesses, and the
   environment lock.
7. **Acceptance**: Tier A exact parity (runs, run ids, contributor ids,
   raster, ownership metadata); Tier B real CUDA execution with
   `correctness_witness_pass=true` and synchronized timing; Tier C
   frozen Hopkins config with cold/warm separated. Any FAILED tier → no
   canonical family; keep the run as exploratory evidence.
8. **Post-run integrity**: `git status --porcelain` must still be clean.
9. **Promote only through `promote_canonical_family()`** — never `cp`.
10. **Verify**:
    ```bash
    python scripts/verify_industrial_v2_artifacts.py
    # required: V2 VERIFIER: PASS
    ```
11. **Generate claims** (only after verifier PASS):
    ```bash
    python scripts/generate_industrial_v2_claims.py
    # writes docs/generated/industrial-v2-claims.{json,md}
    ```
12. **README/README_zh** may cite only ADMITTED `IB2-*` claims from the
    generated outputs, keeping the boundaries: hardware-specific, not
    foundry calibrated, no commercial-tool comparison. Then:
    ```bash
    python scripts/generate_industrial_v2_claims.py --check
    # required: CLAIMS CHECK: OK
    ```
13. **Publication PR** stays narrow: canonical family, generated claims,
    README/README_zh, CHANGELOG. No feature code.

## Honest-outcome rules

* A weak or absent Tier A speedup simply stays non-headline (§27). Do
  not tune the benchmark to manufacture a result.
* A slower GPU end-to-end row is valid diagnostic evidence (§28) — it
  names the next bottleneck. Never convert a kernel-only speedup into an
  end-to-end claim.
* If Tier C GPU stays unsupported, canonical publication remains blocked;
  changing mandatory-tier policy is itself a protocol change (new commit,
  new identity, full rerun) and must not be done after seeing results
  (§29).
* CPU-only hosts: schema/unit tests PASS, Tier A provisional allowed,
  Tier B/C formal BLOCKED, canonical publication BLOCKED. Never emulate
  CUDA and call it formal measurement (§35).

## Provisional dry-run record (Phase 2B)

A CPU-only provisional dry run was executed during Phase 2B on the
checked-in Ibex 32768² fixture (`--tiers a,c --repeats 3`) and exposed +
fixed four harness defects (worker argv, layer selection, Tier C
simulator-result access, Tier A cache-warm measurement bias). Final
recorded outcome, non-authoritative:

* Tier A (window 4096, real fixture): `SUCCESS`,
  `correctness_witness_pass=true`, indexed-vs-reference exact parity,
  first-touch discovery speedup ≈ **135×** vs the pre-G1 full-flat scan,
  **99.6%** of flat scans avoided.
* Tier B: `NOT_RUN_ENVIRONMENT` (no CUDA) — exactly per protocol.
* Tier C: `UNSUPPORTED` (GPU-resident path) with the CPU SOCS finite
  witness `PASS` — exactly per protocol.

Provisional workspaces live under `runs/` and are gitignored; only the
verified canonical family enters git (Phase 2C).
