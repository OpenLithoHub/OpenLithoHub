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
authority_scope = SCALE_CHARACTERIZATION
Lane A: large-layout streaming scale (single device)
Lane B: 1/2/3-GPU scaling (same fixture, same protocol)
NOT v1.1 authority / NOT v2 authority / NOT foundry calibrated
NO commercial-tool comparison
Results are characterization evidence, not marketing claims.
```

## 1. Host requirements

* 3× NVIDIA RTX 3080 (10 GB each) — or a subset; the frozen commands
  name the exact GPU count per run;
* working NVIDIA driver + CUDA-enabled PyTorch + cuDNN visible;
* KLayout Python API (installed with the repository);
* ≥ 32 GiB host RAM recommended; free disk ≥ the prepared fixture size
  + ~64 GiB for run outputs;
* Linux. No MPS, no ROCm, no CPU emulation of CUDA.

Record, before anything else:

```bash
mkdir -p measurement-logs
nvidia-smi -q | tee measurement-logs/nvidia-smi-q.txt
nvidia-smi topo -m | tee measurement-logs/nvidia-smi-topo.txt
git rev-parse HEAD | tee measurement-logs/git-head.txt
git status --porcelain | tee measurement-logs/git-status-before.txt
```

Required: `HEAD` equals the frozen scale commit; `git status` empty.

## 2. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
# install the CUDA-enabled torch build matching this host's driver FIRST
pip install -e ".[server,workflow]"
pip freeze | tee measurement-logs/pip-freeze.txt
```

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
frozen decisions recorded in the committed fixture manifests — never
chosen on the GPU host.

### Frozen fixture facts (maintainer-verified 2026-09-25)

Ibex (`layout/sky130hd/ibex/ibex.gds`, top cell `ibex_core`,
layer 66:44): the v1.1/v2 authority lineage.

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
benchmark): 66:44 is the same sky130hd li1 routed-layout semantic
class as the frozen v1.1/v2 lineage; the audit shows it is non-empty
(27,487,849 shape instances, second densest of 41 layers) and
spans 98.8% × 99.4% of the die. Marker/fill/boundary layers
(14:0, 81:*, 235:*, 236:0) were rejected as non-representative. No
candidate was benchmarked before selection.

Verification evidence committed with this freeze:
`benchmarks/results/industrial-scale/fixtures/microwatt/fixture-manifest.json`
and `pdb-split-manifest.json` (per-chunk git-blob sha1 from the pinned
PDB tree; every chunk byte-verified against it).

## 4. Preflight

```bash
python scripts/preflight_industrial_v2.py \
  --gds benchmarks/results/industrial-scale/fixtures/ibex/ibex.gds \
  --device cuda:0 | tee measurement-logs/preflight.txt
```

Required: `PREFLIGHT: PASS`. Otherwise STOP.

## 5. Frozen measurement commands

> FILL AT FREEZE: `FROZEN_SCALE_COMMIT`, window lists and repeat
> counts below are the frozen protocol. Do not alter them after seeing
> results.

### Command A — Lane A, single GPU, Ibex ladder

```bash
python benchmarks/industrial-scale/run_scale_benchmark.py \
  --fixture-manifest benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json \
  --gds benchmarks/results/industrial-scale/fixtures/ibex/ibex.gds \
  --lanes a \
  --windows <FROZEN_IBEX_LADDER> \
  --device cuda:0 --device-backend cuda --gpu-count 1 \
  --forward-profile P1_FINITE_SUPPORT --sink memmap_npy \
  --tile <FROZEN_TILE> --halo <FROZEN_HALO> --microbatch <FROZEN_MICROBATCH> \
  --repeats <FROZEN_REPEATS> --warmup <FROZEN_WARMUP> \
  --formal \
  2>&1 | tee measurement-logs/lane-a.txt
```

### Command B — Lane B, 1/2/3-GPU scaling (repeat per GPU count)

```bash
for GPUS in 1 2 3; do
python benchmarks/industrial-scale/run_scale_benchmark.py \
  --fixture-manifest benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json \
  --gds benchmarks/results/industrial-scale/fixtures/ibex/ibex.gds \
  --lanes b \
  --windows <FROZEN_SCALING_WINDOW> \
  --device cuda:0 --device-backend cuda --gpu-count $GPUS --worker-count $GPUS \
  --forward-profile P1_FINITE_SUPPORT --sink memmap_npy \
  --tile <FROZEN_TILE> --halo <FROZEN_HALO> --microbatch <FROZEN_MICROBATCH> \
  --repeats <FROZEN_REPEATS> --warmup <FROZEN_WARMUP> \
  --formal \
  2>&1 | tee measurement-logs/lane-b-$GPUS-gpu.txt
done
```

### Command C — optional Microwatt bounded stress (only after G1 passes)

```bash
python benchmarks/industrial-scale/run_scale_benchmark.py \
  --fixture-manifest benchmarks/results/industrial-scale/fixtures/microwatt/fixture-manifest.json \
  --gds benchmarks/results/industrial-scale/fixtures/microwatt/microwatt.gds \
  --lanes a \
  --windows <FROZEN_MICROWATT_LADDER> \
  --device cuda:0 --device-backend cuda --gpu-count 1 \
  --forward-profile P1_FINITE_SUPPORT --sink memmap_npy \
  --tile <FROZEN_TILE> --halo <FROZEN_HALO> --microbatch <FROZEN_MICROBATCH> \
  --repeats <FROZEN_REPEATS> --warmup <FROZEN_WARMUP> \
  --formal \
  2>&1 | tee measurement-logs/microwatt.txt
```

`NOT_RUN_MEMORY_POLICY` / `NOT_RUN_TIME_POLICY` rows are VALID outcomes
for rungs the device cannot serve — record them, never retry with
smaller windows to "make it fit" unless that smaller window is in the
frozen ladder.

## 6. Verify + bundle

```bash
for WS in <RUN_WORKSPACES>; do
  python scripts/verify_industrial_scale_artifacts.py --root "$WS" --require-formal
  # required: SCALE VERIFIER: PASS — family closed (formal tier)
done

python scripts/generate_industrial_scale_claims.py --check
# required: SCALE CLAIMS CHECK: OK (no ISC-* in any README)

tar -czf industrial-scale-evidence.tar.gz measurement-logs \
  benchmarks/results/industrial-scale/runs
sha256sum industrial-scale-evidence.tar.gz | tee measurement-logs/bundle-sha256.txt
```

## 7. STOP conditions

STOP and attach evidence if ANY of:

* preflight fails;
* a frozen command exits nonzero;
* `SCALE VERIFIER` fails;
* any SUCCESS row lacks its correctness witness;
* you feel tempted to edit source, parameters, or artifacts.

Upload: `industrial-scale-evidence.tar.gz` + its SHA-256 + the final
`git status --porcelain` (must be empty) + the completion report from
the GPU issue you are executing.

## 8. Explicit non-goals

No README edits, no headline decisions, no commercial-tool
comparisons, no protocol/threshold/repeat changes, no benchmark source
edits, no v1.1/v2/P-054 mutation. Those are maintainer-owned. If code
must change: STOP — a source fix means a new commit, new run identity,
new dry run, new freeze, and re-running every GPU issue from zero.
