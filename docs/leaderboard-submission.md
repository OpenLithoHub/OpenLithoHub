# Leaderboard Submission Policy

OpenLithoHub's leaderboard is a **claim + verify-by-numbers** registry.
Contributors submit a small YAML file describing the result; the CI
validates it against a strict schema and opens a follow-up PR that
updates `data/leaderboard.json`. **No contributor-supplied code is
executed by CI.**

If you want a fully automated rerun-the-model leaderboard, see the
"Roadmap" section at the bottom — that requires sandboxed compute we do
not yet have.

## How to submit

1. Fork the repo.
2. Create a YAML file under `submissions/<your-handle>/<model>.yaml`
   following the schema below.
3. Open a PR. Add the `submission` label.
4. CI will validate the YAML and (on success) open a follow-up PR that
   appends your entry to `data/leaderboard.json`. A maintainer merges
   that PR. Your entry appears on the next docs build.

## Submission YAML schema

The schema mirrors `openlithohub.leaderboard.schema.BenchmarkResult`.
Required fields are marked with `*`.

```yaml
model_name: my-curvilinear-ilt          # *
dataset: lithobench                     # * one of: lithobench, lithosim, custom-<name>
process_node: 7nm                       # * one of: 45nm, 28nm, 7nm, 5nm-euv, 3nm-euv, 2nm-euv
mask_topology: curvilinear              # * manhattan | curvilinear
track: open                             # optional: open | hackathon-2026q3 (default open)

epe_mean_nm: 1.42                       # * mean EPE in nm, ≥ 0
epe_max_nm: 4.10                        # * max EPE in nm, ≥ 0

pvband_mean_nm: 6.8                     # optional, ≥ 0
pvband_max_nm: 12.1                     # optional, ≥ 0
mrc_violation_rate: 0.0                 # optional, in [0, 1]
drc_pass: true                          # optional
shot_count: 184232                      # optional, ≥ 0
stochastic_robustness: 0.92             # optional, in [0, 1]

paper_url: https://arxiv.org/abs/...    # strongly encouraged
code_url: https://github.com/.../ilt    # strongly encouraged
notes: |
  Trained on H100 for 2 days. Hyperparams in code_url/configs/ilt-7nm.yaml.
```

## What CI does

1. Checks out the base ref (never the PR's head — security boundary).
2. Pulls only files matching `submissions/**.y*ml` from the PR using
   `gh api ... contents` — no shell scripts, no notebooks, no model code.
3. Parses each YAML with `yaml.safe_load` and validates it against
   `BenchmarkResult.model_validate`.
4. Calls `submit_result` against `data/leaderboard.json`.
5. Opens a follow-up PR with the updated JSON.

If validation fails, CI annotates the failing file/line and the
workflow exits non-zero. Fix the YAML and push to the same PR.

## Reviewer responsibilities

A maintainer must confirm before merging the follow-up PR:

- The `paper_url` and `code_url` exist and describe the same model.
- The numbers are *plausible* given the dataset and node — gross
  outliers warrant asking for replication.
- The submitter is the model's author, or has explicit permission.

We will not police every claim — the leaderboard is a public scoreboard
of self-reported numbers. The trust model is "the submitter is staking
reputation on the link they provided," same as arXiv.

## Withdrawing or correcting a submission

Open a PR that edits `data/leaderboard.json` directly. Add the
`leaderboard-correction` label and explain the change in the PR body.

## Roadmap: sandboxed re-runs

A real "we re-run your model and verify the numbers" leaderboard
requires:

- A trusted compute environment (we are evaluating GitHub-hosted
  larger runners + nsjail, plus an opt-in self-hosted GPU pool).
- A standardized model packaging format (Docker image with a
  `predict.py` entrypoint conforming to `LithographyModel`).
- A frozen, public test set (in progress — see
  `docs/benchmarks.md`).

Until we have all three, the claim+verify model above is what we ship.
