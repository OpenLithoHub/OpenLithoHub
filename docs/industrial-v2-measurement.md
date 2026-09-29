# Industrial Benchmark v2 — Formal Measurement Runbook

Operational runbook for executing the formal v2 measurement on a
controlled NVIDIA host, per the PR-G Phase 2B/2C split. The protocol
authority itself lives in `src/openlithohub/benchmark/industrial_v2.py`
and `benchmarks/industrial-v2/run_v2_benchmark.py`; this page is the
operator procedure. It is written to be executable by an operator who
has never participated in PR-G development.

## Status

```text
Phase 2A  (merged):    V2 PROTOCOL READY / FORMAL GPU MEASUREMENT PENDING
Phase 2B   (merged):   measurement harness + CPU-only provisional dry run
Phase 2B.1 (merged):   measurement-authority repair (blockers A–K)
Phase 2B.2 (this page): declared-vs-executed repair A–F; executable CUDA Tier C
Phase 2C:              canonical artifacts + generated claims + README
```

**Historical vs current Tier C.** During the Phase 2B CPU-only dry run,
Tier C recorded `UNSUPPORTED` and older revisions of this page said the
GPU Tier C path was not implemented. That text is historical: since
Phase 2B.1 the GPU-resident SOCS Hopkins path is implemented and
verified, and Tier C executes on CUDA like Tier B. Do not follow any
external copy of this page that still calls the current Tier C GPU path
`UNSUPPORTED`.

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
* ≥ 16 GiB host RAM.
* **≥ 16 GiB free disk — this is the single authoritative threshold and
  the preflight enforces it** (`disk free >= 16 GiB` must PASS). The
  32768² fp32 window alone spans ~4 GiB per tensor, and the run keeps
  the fixture, per-identity workspaces and results on disk.
* KLayout Python API installed; real routed Ibex GDS fixture available.

## Tier C timing contract (what the harness actually does)

Tier C is a fixed-configuration Hopkins compute tier executed on the
GPU-resident SOCS path. Every claim-bearing timing observation follows
the fresh-worker contract:

```text
fresh worker process
→ untimed setup / warmup (device-resident kernels, FFT tables, warmup executions)
→ ONE synchronized measured steady-state GPU execution
→ ONE timing observation (gpu_warm_wall_s)
```

The driver performs the configured fresh-process repeats and computes
the claim-bearing statistics itself: `median / p10 / p90 / n` over the
per-worker observations (`aggregate_*` in the tier row). **Never
best-of-N**: no worker takes a minimum over in-process repetitions, and
a SUCCESS row with fewer observations than measured repeats fails
closed. Cold start (first GPU call including SOCS kernel construction)
is recorded per repeat as `gpu_cold_wall_s` — a separate diagnostic
fact that never enters the warm steady-state headline statistic.

## Tier C bounded-SOCS memory authority (GPU Authority Repair v3)

Tier C executes the **exact bounded-memory SOCS decomposition**
(`exact_block_gram_topk_v1`): a streamed two-pass Gram construction —
frequency-column blocks are generated, accumulated into the (n_src ×
n_src) complex128 Gram matrix, one eigendecomposition selects the
top-K modes, and a second streamed pass reconstructs the kernels
directly into the output slices. **Full H and full Vh are never
resident.** The legacy dense full-H `torch.linalg.svd` runtime has been
removed from every production, benchmark and preflight path with **no
fallback** (no flag, no environment variable, no re-enable).

The driver computes the **SOCS memory plan** once — from the real frozen
optical configuration (actual `n_src`, `n_freq`, `K`) and quiescent
device memory facts — BEFORE the run identity exists, binds its
canonical SHA-256 into `RunConfigV2`, and writes it to the workspace as
`socs-memory-plan.json`. Every fresh worker receives that exact plan,
validates its hash and semantic inputs, and executes it — **workers
never replan**. Free memory is checked against the plan floor before
every chunk allocation (`MEMORY_PLAN_HEADROOM_VIOLATION` stops before
allocation), and an unexpected CUDA OOM after a planner PASS is a
`PLANNER_CONTRACT_VIOLATION` that fails the run — **CUDA OOM is never
adaptive control flow**, and no shrink-and-retry exists.

Chunk selection is **capacity-optimal** under the frozen policy (the
largest aligned chunk satisfying the conservative peak-memory
inequality) and depends only on the formal dimensions, dtype, K, GPU
memory facts and frozen planner constants — never on benchmark results.
The planned-vs-observed memory peak (`memory_plan_peak_witness_pass`)
and the fresh-worker environment fingerprint
(`worker_environment_witness_pass`) are locked facts of every formal
Tier-C SUCCESS row, enforced by the verifier.

The preflight has two distinct Tier-C gates (v3 §14): the
**analytical formal-grid memory plan** on the ACTUAL frozen grid (never
a stand-in grid — a small-grid smoke proves nothing about formal-grid
capacity), and a **bounded real-CUDA block-path probe** through the
exact worker implementation. Formal preflight PASS means: the formal
problem is analytically feasible under the frozen policy AND the real
bounded block path executes on CUDA.

## Procedure

1. **Clone the exact measurement commit** (the frozen candidate
   measurement commit; record `git rev-parse HEAD`) and confirm
   `git status --porcelain` is empty.
2. **Isolated environment**: fresh venv, install the host's CUDA-enabled
   PyTorch build, then `pip install -e ".[server,workflow]"`. Capture
   `pip freeze` and the torch/CUDA facts. Never edit dependencies
   mid-run.
3. **Fixture identity**: `sha256sum ibex.gds` — the same real routed
   Ibex lineage as v1 (`benchmarks/results/industrial/fixtures/`
   provenance). Never substitute a different GDS mid-run. The fixture
   is multi-layer; the harness executes the declared GDS layer
   (`--layer`, default `66:44` sky130hd li1) in Tier A and Tier B, and
   the layer is part of the run identity.
4. **Preflight** (must print `PREFLIGHT: PASS`, otherwise stop):
   ```bash
   python scripts/preflight_industrial_v2.py --gds /path/to/ibex.gds --device cuda:0
   ```
5. **Save operator evidence** (preflight output, `nvidia-smi -q`, git
   status, commit) under `measurement-logs/`. Canonical facts remain
   harness-owned — operator logs are context, never claim inputs.
   `measurement-logs/` is ignored by the committed root `.gitignore`
   (GPU Authority Repair §11): the tracked tree stays verifiably clean
   with no operator-created nested `.gitignore`.
6. **Run the formal tiers through the harness only** — never ad-hoc
   Python, never shell `time`:
   ```bash
   python benchmarks/industrial-v2/run_v2_benchmark.py \
       --tiers a,b,c --gds /path/to/ibex.gds --device cuda:0 \
       --dtype fp32 --repeats 5 --warmup 2 --batch 8 --tile 1024 \
       --layer 66:44 \
       --formal
   ```
   The harness owns the run workspace
   (`runs/<run_identity>/`), the frozen SOCS memory plan, fresh-process
   repeats, synchronized CUDA timing, allocated/reserved GPU peaks,
   correctness witnesses, the environment lock, and — on a formal run —
   the fail-closed build of the exact canonical family (including
   `industrial-v2-socs-memory-plan.json`) inside the workspace
   (`run-summary.json` then records `canonical_build_blockers`; it must
   be `[]`).
7. **Acceptance**: Tier A exact parity (runs, run ids, contributor ids,
   raster, ownership metadata); Tier B real CUDA execution with
   `correctness_witness_pass=true` and synchronized timing; Tier C GPU
   steady-state execution with the fresh-worker single-observation
   contract above (`timing_observations == 1` per repeat). Any FAILED
   tier → no canonical family; keep the run as exploratory evidence.
8. **Post-run integrity**: `git status --porcelain` must still be clean.
9. **Promote only through the operator CLI** — never `cp`, never an
   ad-hoc Python snippet:
   ```bash
   python scripts/promote_industrial_v2_artifacts.py \
       --workspace benchmarks/results/industrial-v2/runs/<RUN_ID> \
       --canonical-root /path/to/canonical-staging
   ```
   The CLI is fail-closed: it requires the exact complete seven-member
   family, re-runs the verifier and the formal blockers (including a
   live dirty-tree check), refuses provisional/incomplete/drifted
   families, never touches the frozen v1.1 root, and never silently
   overwrites an existing unrelated canonical authority (re-promoting
   the same byte-identical family is idempotent). Required output:
   `V2 PROMOTION: PASS`.
10. **Verify**:
    ```bash
    python scripts/verify_industrial_v2_artifacts.py \
        --canonical-root /path/to/canonical-staging
    # required: V2 VERIFIER: PASS
    ```
11. **Generate claims** (only after verifier PASS):
    ```bash
    python scripts/generate_industrial_v2_claims.py \
        --canonical-root /path/to/canonical-staging
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
* A failed or unavailable CUDA tier blocks canonical publication;
  changing mandatory-tier policy is itself a protocol change (new
  commit, new identity, full rerun) and must not be done after seeing
  results (§29).
* CPU-only hosts: schema/unit tests PASS, Tier A provisional allowed,
  Tier B/C formal BLOCKED (rows record `NOT_RUN_ENVIRONMENT`), canonical
  publication BLOCKED. Never emulate CUDA and call it formal measurement
  (§35).

## Historical record (non-authoritative): Phase 2B CPU-only dry run

A CPU-only provisional dry run was executed during Phase 2B on the
checked-in Ibex 32768² fixture (`--tiers a,c --repeats 3`) and exposed +
fixed four harness defects (worker argv, layer selection, Tier C
simulator-result access, Tier A cache-warm measurement bias). Final
recorded outcome, **non-authoritative and predating the GPU-resident
Tier C implementation**:

* Tier A (window 4096, real fixture): `SUCCESS`,
  `correctness_witness_pass=true`, indexed-vs-reference exact parity,
  first-touch discovery speedup ≈ **135×** vs the pre-G1 full-flat scan,
  **99.6%** of flat scans avoided.
* Tier B: `NOT_RUN_ENVIRONMENT` (no CUDA) — exactly per protocol.
* Tier C: `UNSUPPORTED` **as of that historical run only** (the
  GPU-resident path did not exist yet; the CPU SOCS finite witness
  passed) — exactly per protocol at the time. Current Tier C executes on
  CUDA; see the timing contract above.

Provisional workspaces live under `runs/` and are gitignored; only the
verified canonical family enters git (Phase 2C).
