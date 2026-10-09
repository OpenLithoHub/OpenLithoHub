# Industrial Scale Benchmark — GPU Handoff Runbook

For the operator executing the scale/multi-GPU characterization on the
3×RTX 3080 host. **You never choose parameters, never edit source, and
never tune anything**: every command below is frozen once the SHAs are
filled. If anything fails, STOP, attach the evidence, and report — a
failure is diagnostic evidence, never permission to change the protocol.

> PLACEHOLDER RULE: every `FROZEN_...` and `sha256` value marked
> `<...>` below is filled by the maintainer at freeze time. Do not
> substitute `main`, `latest`, or any other commit.

## 0. What this run is

```text
reference GPU host: 1 × NVIDIA RTX 4090
authority_scope = SCALE_CHARACTERIZATION
Lane A: large-layout single-GPU streaming scale
Lane C: single-GPU microbatch saturation (frozen ladder 1,2,4,8,16,32)
Lane B: multi-GPU capability RETAINED — measurement DEFERRED on this host
NOT v1.1 authority / NOT v2 authority / NOT foundry calibrated
NO commercial-tool comparison
Current campaign must NOT claim multi-GPU scaling/speedup/efficiency.
Results are characterization evidence, not marketing claims.
Historical 3×RTX 3080 protocol provenance is retained in the charter
and the superseded issue/templates — it is not the current protocol.
```

## 1. Host requirements (frozen formal Scale host policy, §22)

The formal host policy is FROZEN IN SOURCE
(`openlithohub.benchmark.industrial_scale.SUPPORTED_SCALE_GPU_MODELS`)
BEFORE the rerun — the preflight and the formal family builder both
enforce it fail-closed:

* GPU model: `NVIDIA GeForce RTX 4090 Laptop GPU` (this campaign's
  reference host) **or** `NVIDIA GeForce RTX 4090` (desktop) — both
  enumerated explicitly;
* VRAM ≥ 15 GiB; compute capability 8.9; device count ≥ 1 (`cuda:0`);
* OS: Windows 11 or Linux (both supported — the preflight records the
  actual platform);
* working NVIDIA driver + CUDA-enabled PyTorch + cuDNN visible; the
  driver/device identity is captured by source-owned code (nvidia-smi
  first, a Windows registry fallback when NVML fails) — an empty or
  unattributed identity blocks the formal run;
* KLayout Python API (installed with the repository);
* ≥ 32 GiB host RAM recommended; free disk ≥ the prepared fixture size
  + ~64 GiB for run outputs;
* No MPS, no ROCm, no CPU emulation of CUDA — a CPU forward under a
  CUDA-enabled PyTorch can never pass the preflight's actual-CUDA probe.

Record, before anything else:

```bash
cd "$(git rev-parse --show-toplevel)"   # anchor EVERY command at the OpenLithoHub
                                        # repo root — never the PDB sibling checkout
mkdir -p measurement-logs               # git IGNORES this dir but never creates it;
                                        # a fresh clone fails the first tee without it
nvidia-smi -q | tee measurement-logs/nvidia-smi-q.txt
nvidia-smi topo -m | tee measurement-logs/nvidia-smi-topo.txt
git rev-parse HEAD | tee measurement-logs/git-head.txt
git status --porcelain | tee measurement-logs/git-status-before.txt
```

Required: `HEAD` equals the frozen scale commit; `git status` empty.
`measurement-logs/` is ignored by the committed root `.gitignore`
(GPU Authority Repair §11), so evidence logs never dirty the tracked
tree and no operator-created nested `.gitignore` is needed. Re-check
repo root + HEAD + tree together after ANY track switch (V2 freezes at
`d9ec685a…`, Scale at `b8948572…` — a checkout done in the wrong
working directory is the classic silent failure).

## 2. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
# install the CUDA-enabled torch build matching this host's driver FIRST
pip install -e ".[server,workflow]"
pip freeze | tee measurement-logs/pip-freeze.txt
command -v python | tee measurement-logs/python-path.txt
python -c "import klayout.db; print('KLayout Python API: OK in this interpreter')" \
  | tee measurement-logs/klayout-identity.txt
```

ONE interpreter for everything: the KLayout gate, the preflight and
every benchmark command must run in the python recorded in
`python-path.txt` — never switch environments mid-campaign, never
install packages to rescue a failing gate.

## 3. Fixture preparation (frozen source, verified bytes)

Clone the frozen PDB lineage and prepare both fixtures exactly once:

```bash
git clone https://github.com/SJTU-YONGFU-RESEARCH-GRP/PDB-Physical-Design-Database.git
git -C PDB-Physical-Design-Database checkout 9e1e3399b1b707f26fee853bce1ff91ab466ce24
git -C PDB-Physical-Design-Database rev-parse HEAD   # must equal the commit above

python scripts/prepare_industrial_scale_fixture.py \
  --source-root PDB-Physical-Design-Database \
  --design ibex \
  --selected-layer 66:44 \
  --pixel-nm 1.0 \
  --expected-commit 9e1e3399b1b707f26fee853bce1ff91ab466ce24 \
  | tee measurement-logs/fixture-ibex.txt

python scripts/prepare_industrial_scale_fixture.py \
  --source-root PDB-Physical-Design-Database \
  --design microwatt \
  --selected-layer 66:44 \
  --pixel-nm 1.0 \
  --expected-commit 9e1e3399b1b707f26fee853bce1ff91ab466ce24 \
  | tee measurement-logs/fixture-microwatt.txt
```

Each must print `FIXTURE PREP: PASS`. Record both manifest
`gds_sha256` values. The layers above are the maintainer's audited,
frozen decisions — recorded in the fixture manifests (committed for
microwatt; host-local for ibex) and anchored in source
(`openlithohub.benchmark.scale_parameter_authority`) — never chosen on
the GPU host.

### Frozen fixture facts (maintainer-verified 2026-09-25)

Ibex (`layout/sky130hd/ibex/ibex.gds`, top cell `ibex_core`,
layer 66:44): the v1.1/v2 authority lineage. Frozen GDS sha256
(regenerated from the pinned PDB tree, #94):

```text
5b706ac417f994d357ff78627a01baad8724b808b5e66ed7fe32a022f904664c
```

The ibex fixture directory is host-local (gitignored) — the frozen
sha256 above and `--expected-gds-sha256` are the only gates against a
stale local `ibex.gds` (a 144,566-byte standin-era file, sha256
`9b1790b9…`, is known to linger on measurement hosts and has NO
authority; see §9).

Microwatt (`layout/sky130hd/microwatt/split/`, top cell `microwatt`,
11 split chunks `microwatt.gds.part_[aa..ak]`):

```text
GDS sha256:            b0253af06f35d1a8b11c2a47f70aac89f33be53e8f28da844b35c2ad6cc92a6d
GDS bytes:             554,770,926
DBU:                   1.0 nm
bbox (DBU):            [0, 0, 3020000, 3610000]
die size px @1nm/px:   3,020,000 × 3,610,000
equivalent pixels:     10,902,200,000,000
dense float32 raster-  43,608,800,000,000 bytes
equivalent:            (hypothetical — DERIVED, never materialized)
layers present:        41 (full inventory in the committed fixture manifest)
selected layer:        66:44
```

Layer-decision rationale (source-owned rule, fixed BEFORE any
benchmark; per-design authority records in
`openlithohub.benchmark.scale_parameter_authority`): 66:44 is the same
sky130hd li1 routed-layout semantic class as the frozen v1.1/v2
lineage; the audit shows it is non-empty (27,487,849 flattened shape
instances — second densest of 41 non-empty layers by this metric,
densest is 67:44) and spans 98.8% × 99.4% of the die. For Ibex the
basis is `lineage` (the v1.1/v2 runs live-measured this layer); for
Microwatt it is `direct_audit` — no analogy inheritance.
Marker/fill/boundary layers (14:0, 81:*, 235:*, 236:0) were rejected
as non-representative. No candidate was benchmarked before selection.

Verification evidence committed with this freeze:
`benchmarks/results/industrial-scale/fixtures/microwatt/fixture-manifest.json`
and `pdb-split-manifest.json` (per-chunk git-blob sha1 from the pinned
PDB tree; every chunk byte-verified against it), plus the PR-V3.1.3
committed layer audit
`benchmarks/results/industrial-scale/audits/microwatt-layer-audit.json`
(regenerable via `scripts/audit_industrial_scale_fixture_layer.py` —
the artifact that turns the numbers above from prose into evidence).

## 4. Preflight

```bash
python scripts/preflight_industrial_v2.py \
  --gds benchmarks/results/industrial-scale/fixtures/ibex/ibex.gds \
  --device cuda:0 | tee measurement-logs/preflight.txt

python scripts/preflight_industrial_scale.py \
  --ibex-manifest benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json \
  --gds-ibex benchmarks/results/industrial-scale/fixtures/ibex/ibex.gds \
  --microwatt-manifest benchmarks/results/industrial-scale/fixtures/microwatt/fixture-manifest.json \
  --gds-microwatt benchmarks/results/industrial-scale/fixtures/microwatt/microwatt.gds \
  --device cuda:0 \
  --expected-commit <FROZEN_SCALE_COMMIT> \
  | tee measurement-logs/scale-preflight.txt
```

Required: `PREFLIGHT: PASS` AND `SCALE PREFLIGHT: PASS`. Otherwise STOP.

## 5. Frozen measurement commands

> FILL AT FREEZE: `FROZEN_SCALE_COMMIT` below is the only remaining
> placeholder (filled at handoff).  The window ladders, tile/halo/
> microbatch and repeat/warmup counts below are the FROZEN protocol
> (aligned with the v2 formal protocol) — do not alter them after
> seeing results.
>
> FROZEN SCALE COMMIT: <FULL_40_HEX_SHA>

### Command A — Lane A, single GPU, WINDOW-SCOPED frozen ladder

GPU Authority Repair §19: Lane A is **window-scoped** — each frozen
window is its OWN formal invocation with its own canonical family and
its own verifier PASS. An interrupted run restarts at ONLY the missing
window; arbitrary rows are never stitched together after the fact.

Run each window exactly once, unmodified:

```bash
for W in 4096 8192 16384 32768; do
  python benchmarks/industrial-scale/run_scale_benchmark.py \
    --fixture-manifest benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json \
    --gds benchmarks/results/industrial-scale/fixtures/ibex/ibex.gds \
    --lanes a \
    --windows $W \
    --device cuda:0 --device-backend cuda --gpu-count 1 \
    --forward-profile P1_FINITE_SUPPORT --sink memmap_npy \
    --tile 1024 --halo 64 --microbatch 8 \
    --repeats 5 --warmup 2 \
    --formal \
    2>&1 | tee measurement-logs/lane-a-w$W.txt
done
```

After ALL FOUR window families verify (`SCALE VERIFIER: PASS — formal
tier`), bind the campaign:

```bash
python scripts/build_industrial_scale_lane_a_campaign.py \
  --window 4096=<WS_4096>/family --window 8192=<WS_8192>/family \
  --window 16384=<WS_16384>/family --window 32768=<WS_32768>/family \
  --out measurement-logs/industrial-scale-lane-a-campaign.json

python scripts/verify_industrial_scale_lane_a_campaign.py \
  --campaign measurement-logs/industrial-scale-lane-a-campaign.json \
  --window 4096=<WS_4096>/family --window 8192=<WS_8192>/family \
  --window 16384=<WS_16384>/family --window 32768=<WS_32768>/family
```

Required: `LANE-A CAMPAIGN VERIFIER: PASS — four frozen windows, one
authority`. The campaign manifest binds one measurement source SHA, one
fixture SHA, one environment-lock SHA and one frozen protocol across
all four windows — mixed anything is a hard FAIL.

### Command B — Lane C, single-GPU microbatch saturation (frozen ladder)

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

Baseline is microbatch = 1. The FULL ladder (1,2,4,8,16,32) is
reported — never select the best rung and suppress the others.

**Lane B (multi-GPU scaling) is DEFERRED on this host.** Do NOT emulate
it by mapping several workers onto one device — that measures host-side
contention, not multi-GPU scaling, and must never be labeled
"multi-GPU". The scheduler/executor remain retained capabilities (CPU
hostile tests prove scheduler semantics only).

### Command C — optional Microwatt bounded stress (only after G1 passes)

```bash
python benchmarks/industrial-scale/run_scale_benchmark.py \
  --fixture-manifest benchmarks/results/industrial-scale/fixtures/microwatt/fixture-manifest.json \
  --gds benchmarks/results/industrial-scale/fixtures/microwatt/microwatt.gds \
  --lanes a \
  --windows 4096,8192 \
  --device cuda:0 --device-backend cuda --gpu-count 1 \
  --forward-profile P1_FINITE_SUPPORT --sink memmap_npy \
  --tile 1024 --halo 64 --microbatch 8 \
  --repeats 5 --warmup 2 \
  --formal \
  2>&1 | tee measurement-logs/microwatt.txt
```

`NOT_RUN_MEMORY_POLICY` / `NOT_RUN_TIME_POLICY` rows are VALID outcomes
for rungs the device cannot serve — record them, never retry with
smaller windows to "make it fit" unless that smaller window is in the
frozen ladder.

Microwatt wording rule: the 43,608,800,000,000-byte figure is the
hypothetical dense float32 raster EQUIVALENT derived from the exact
bbox — never word it as "processed a 43.6 TB file".

## 6. Verify + evidence packaging

```bash
for WS in <RUN_WORKSPACES>; do
  python scripts/verify_industrial_scale_artifacts.py --root "$WS" --require-formal
  # required: SCALE VERIFIER: PASS — family closed (formal tier)
done

python scripts/generate_industrial_scale_claims.py --check
# required: SCALE CLAIMS CHECK: OK (no ISC-* in any README)
```

Then build the deterministic authority bundle (GPU Authority Repair
§25) — run config, environment lock, fixture member, row JSONs,
run-summary, canonical family + SHA256SUMS, preflight/verifier outputs,
git HEAD/clean-tree evidence, environment identity capture and the
Lane-A campaign manifest. Raw sink buffers stay LOCAL and are recorded
by SHA-256 + byte count only (never upload tens of GB of `.npy`):

```bash
python scripts/package_gpu_authority_evidence.py \
  --workspace <RUN_WORKSPACE> \
  --preflight-log measurement-logs/scale-preflight.txt \
  --verifier-log measurement-logs/verifier.txt \
  --campaign-manifest measurement-logs/industrial-scale-lane-a-campaign.json \
  --out-dir evidence/
# required: EVIDENCE PACKAGING: BUILT — gpu-authority-evidence-<run-id>.tar.gz
# record the printed archive sha256
```

The packaging is deterministic: identical inputs produce byte-identical
archives.

## 7. STOP conditions

STOP and attach evidence if ANY of:

* preflight fails;
* a frozen command exits nonzero;
* `SCALE VERIFIER` fails;
* any SUCCESS row lacks its correctness witness;
* `git status --porcelain` becomes non-empty at any point — this is a
  REPRODUCIBILITY-VERIFICATION failure (tracked protocol state changed
  since the freeze), not a tool bug; never repair it with
  `git checkout --` (see §9 for what the gate can and cannot see);
* you feel tempted to edit source, parameters, or artifacts.

Upload: the `gpu-authority-evidence-<run-id>.tar.gz` bundle(s) + their
SHA-256 + the final `git status --porcelain` (must be empty) + the
completion report from the GPU issue you are executing.

## 8. Explicit non-goals

No README edits, no headline decisions, no commercial-tool
comparisons, no protocol/threshold/repeat changes, no benchmark source
edits, no v1.1/v2/P-054 mutation. Those are maintainer-owned. If code
must change: STOP — a source fix means a new commit, new run identity,
new dry run, new freeze, and re-running every GPU issue from zero.

## 9. Operator checklist — what the scripts cannot check (PR-V3.1.3)

The fail-closed gates verify everything mechanically checkable; these
three classes of defect live exactly where scripts stop. Check each by
hand BEFORE starting a campaign — every past closure-round finding in
this class was preventable here.

**1. Per-design gate semantics — what "tracked tree clean" can and
cannot see.** The same `git status` gate has different reach per
design, and that is protocol, not accident:

```text
ibex:      whole fixture dir host-local (gitignored) → gate sees NOTHING;
           integrity = manifest SHA-256 revalidation + --expected-gds-sha256
microwatt: the two manifests are TRACKED → gate is a live tripwire on them;
           the 554,770,926-byte GDS is gitignored → SHA-256 equality only
```

Source of truth:
`openlithohub.benchmark.scale_parameter_authority` (declared, CI-checked
against the real `.gitignore`). When the preflight prints the
"tracked tree clean" FAIL, it now explains this semantics itself — read
it; do not reach for `git checkout --`.

**2. Resource margins, per design, before the first window.** Confirm
free disk ≥ prepared fixture + ~64 GiB outputs (§1) AND, for the
design being measured: GDS bytes (ibex ≈ per frozen manifest;
microwatt 554,770,926), dense float32 raster-EQUIVALENT (microwatt
43,608,800,000,000 bytes — derived, never materialized), expected
memmap output volume per window, and host RAM headroom. A "NOT_RUN_
MEMORY_POLICY" row is a valid outcome (§5 Command C); an operator who
did not check margins first is not diagnosing, just discovering.

**3. Machine hygiene — stale-state inventory before fixture prep.**

```bash
ls -la benchmarks/results/industrial-scale/fixtures/*/   # sizes must match the frozen bytes
git -C "$PDB_ROOT" rev-parse HEAD                        # must equal 9e1e3399b1b707f26fee853bce1ff91ab466ce24
git -C "$PDB_ROOT" status --porcelain                    # must be empty (operator authority gate, #94)
```

Known hazards: the standin-era 144,566-byte `ibex.gds`
(sha256 `9b1790b9…`, no authority), leftover `pdb-standin/` fixture
directories, and PDB clones from earlier campaigns. Rule: never repair
a stale clone or fixture in place — regenerate from a clean frozen
sibling checkout; the frozen sha256 values (ibex `5b706ac4…`,
microwatt `b0253af0…`) are the only acceptance test for prepared
bytes. The layer audit tool fails closed on any sha mismatch by
construction.
