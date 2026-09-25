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
   (`runs/<run_identity>/`), fresh-process repeats, synchronized CUDA
   timing, allocated/reserved GPU peaks, correctness witnesses, the
   environment lock, and — on a formal run — the fail-closed build of
   the exact seven-member canonical family inside the workspace
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
