# OpenLithoHub Mini-Hackathon — 2026-Q3

> **Status — 2026-Q2:** rules below are the *charter* the organisers will
> open the hackathon against. Registration, exact dataset checkpoint,
> and prize specifics will be finalised when the Discord channel
> launches. Watch the repo (or open an issue with the `community` label)
> to be notified when sign-ups go live.

## What

A short, focused competition for ML-for-OPC and ML-for-ILT models
against a frozen test split, using the OpenLithoHub benchmark stack
end-to-end.

The goal is **not** to crown a permanent SOTA. It is to:

1. Force the field to compare against the *same* numbers, on the
   *same* test split, with the *same* MRC gate.
2. Surface lightweight, reproducible models that don't require a
   100-GPU rig — bias-correction tricks, smart augmentation,
   architecture choices that pay off at small parameter budgets.
3. Stress-test the leaderboard CI (claim + verify-by-numbers) under
   real submission load.

## Track

`hackathon-2026q3` — a separate leaderboard track, scored independently
from the open ongoing leaderboard. Entries flow through the same
submission pipeline (`submissions/<handle>/<model>.yaml`) but with
`track: hackathon-2026q3` set in the YAML.

## Dataset

- **Training**: any combination of LithoBench, GAN-OPC, and ICCAD'16
  hotspot, plus optional synthetic data from
  `openlithohub.synth` (when shipped). Pretrained backbones are
  allowed; pretraining on *the held-out test split or any superset of
  it* disqualifies the entry.
- **Evaluation**: a frozen LithoBench-derived test split published as
  a tagged release of the repo when the hackathon opens. The exact
  commit SHA + tag will pin the test set so future-you can rerun.
  The contract — tag name, sample count, gates, target — lives in
  [`hackathon/2026q3.yaml`](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/hackathon/2026q3.yaml)
  and freezes at `status: open` time.

## Metrics & gating

| Metric                       | Computed by                                  | Gate                  |
| ---------------------------- | -------------------------------------------- | --------------------- |
| EPE mean (nm)                | `openlithohub.benchmark.compute_epe`         | Lower is better       |
| EPE max (nm)                 | `compute_epe`                                | Reported              |
| PV-Band mean (nm)            | `compute_pvband`                             | Reported              |
| MRC violation rate           | `check_mrc` / `check_curvilinear_mrc`        | **Must be 0.0**       |
| DRC pass                     | `check_drc`                                  | **Must be `true`**    |
| EUV stochastic robustness    | `compute_stochastic_robustness`              | Reported (EUV nodes)  |

**Hard gates: MRC violation > 0 or DRC fail → submission rejected.**
This is non-negotiable; "I got SOTA EPE but my mask wouldn't tape out"
is exactly what the project exists to prevent.

Ranking is by **EPE mean (nm), ascending**, ties broken by EPE max,
ties broken by PV-Band mean.

## Target

The organisers will publish a **target EPE mean** when the hackathon
opens (calibrated against the strongest open baseline at that time —
recipe pinned in [`hackathon/2026q3.yaml`](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/hackathon/2026q3.yaml)).
Beating the target unlocks the prize tier; not beating it still places
you on the ranked board if the hard gates pass.

## Timeline

| Phase                      | Duration                       |
| -------------------------- | ------------------------------ |
| Registration open          | 2 weeks                        |
| Submission window          | 4 weeks (overlapping)          |
| Result verification        | 1 week (CI re-runs every PR)   |
| Awards                     | Announced 1 week post-window   |

The competition runs once. There is no second round; we'd rather see
people contribute follow-up models to the open leaderboard than chase
hackathon variance.

## Prizes

Specifics will land before registration opens. Plan-of-record:

- Top-1: a sponsor compute credit + a project blog post.
- Top-3: shoutouts in the project release notes and a permanent
  leaderboard pin.
- All compliant entries: a co-authored short technical report posted on
  the docs site, with each contributor credited.

We are deliberately keeping monetary prizes out of v1 to avoid attracting
submission-mill behaviour. The leaderboard slot and the technical-report
co-authorship are the actual incentives we believe will pull serious
researchers.

## Eligibility

- Anyone, anywhere, individual or team.
- Code submitted with a permissive open-source licence (Apache-2.0,
  MIT, BSD). Closed-source submissions are not eligible — this is an
  open benchmarking project.
- Sponsors and core maintainers may submit but are scored separately
  ("hors concours") to keep the prize pool focused on the community.

## Anti-cheat

The leaderboard CI [auto-leaderboard.yml](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/.github/workflows/auto-leaderboard.yml)
re-evaluates each submission against the frozen test split. The model
class must be pip-installable from the `code_url` declared in the YAML;
opaque binaries are rejected. Numbers in the YAML must match the CI
re-run within tolerance, or the submission is closed.

If your training set provably contained the test split, the entry is
removed from the ranking. We rely on contributors to declare their
training data honestly; we will spot-check by sampling and comparing
against published dataset hashes.

## How to enter

1. Build your model so it conforms to the `LithographyModel` interface
   (see [colab_byom.ipynb](https://colab.research.google.com/github/OpenLithoHub/OpenLithoHub/blob/main/notebooks/colab_byom.ipynb)
   for the minimal example).
2. Run `openlithohub eval --dataset <hackathon-test-tag> --model
   <yours>` locally to get your numbers.
3. Open a PR adding `submissions/<your-handle>/<model>.yaml` with
   `track: hackathon-2026q3`.
4. Watch the CI verify the numbers. On success, your entry appears on
   the hackathon leaderboard within 24 hours.

## Questions

- Open a [GitHub Issue](https://github.com/OpenLithoHub/OpenLithoHub/issues/new?labels=hackathon)
  with the `hackathon` label.
- Once Discord launches, the `#hackathon` channel will be the live
  Q&A forum.
