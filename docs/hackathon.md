# Hackathon — Contributor Guide

> **The canonical hackathon page lives on the main site:**
> [openlithohub.com/hackathon](https://openlithohub.com/hackathon).
> That page holds the live status, contract values (test-set tag, sample
> count, target EPE), prize specifics, and "Notify me" sign-up.
>
> This page is for contributors who are *writing a submission* and need to
> know how the hackathon plugs into the OpenLithoHub toolchain.

## How the hackathon hooks into the toolchain

The hackathon reuses the open-leaderboard pipeline. Nothing special — same
CLI, same submission YAML, same CI verification. The only difference is a
single `track:` field that routes your entry to the hackathon ranking
instead of the open one.

```yaml
# submissions/<your-handle>/<model>.yaml
track: hackathon-2026q3   # routes to the hackathon board
model_name: ...
code_url: ...
# ... regular BenchmarkResult fields ...
```

## Frozen contract

The hackathon's test split, gates, and ranking rules are pinned in
[`hackathon/2026q3.yaml`](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/hackathon/2026q3.yaml).
Once that file ships with `status: open` in a tagged release, the contract
is immutable — adjustments after open invalidate prior entries.

The contract YAML is the source of truth for the live values rendered on
[openlithohub.com/hackathon](https://openlithohub.com/hackathon); read the
file directly if you want the exact tag, sample count, and target EPE.

## Hard gates

The hackathon shares OpenLithoHub's manufacturability gates — these are
the same gates documented in [Benchmarks](benchmarks.md):

- **MRC violation rate must be 0.0** — `check_mrc` /
  `check_curvilinear_mrc`
- **DRC must pass** — `check_drc`

If either gate fails, the submission is rejected before ranking. "I got
SOTA EPE but my mask wouldn't tape out" is exactly the failure mode the
project exists to prevent.

Ranking among gate-passing entries is by **EPE mean (nm) ascending**, with
EPE max and PV-Band mean as tiebreakers (in that order).

## Submission flow

1. Build your model so it conforms to the
   [`LithographyModel`](api/models.md) interface — the
   [BYOM Colab](https://colab.research.google.com/github/OpenLithoHub/OpenLithoHub/blob/main/notebooks/colab_byom.ipynb)
   has the minimal example.
2. Run the eval CLI locally against the frozen test tag (the tag value is
   on the [main hackathon page](https://openlithohub.com/hackathon) and in
   [`hackathon/2026q3.yaml`](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/hackathon/2026q3.yaml)):

   ```bash
   openlithohub eval --dataset <hackathon-test-tag> --model <yours>
   ```

3. Open a PR adding `submissions/<your-handle>/<model>.yaml` with
   `track: hackathon-2026q3`. The
   [Leaderboard Submissions](leaderboard-submission.md) guide covers the
   full schema and the `submission` PR label.
4. The
   [`auto-leaderboard.yml`](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/.github/workflows/auto-leaderboard.yml)
   workflow re-runs your model against the frozen test split. If your
   reported numbers match within tolerance and gates pass, the entry
   appears on the hackathon board within 24 hours.

## Anti-cheat — what the CI checks

- The model class must be **pip-installable** from `code_url`. Opaque
  binaries are rejected.
- The CI re-runs the eval against the frozen tag. Numbers in the YAML
  must match within tolerance; mismatches close the submission.
- If your training set provably contained the test split, the entry is
  removed. We rely on contributors to declare training data honestly and
  spot-check by sampling.

## Eligibility

- Anyone, anywhere, individual or team.
- Code submitted under a permissive open-source licence (Apache-2.0, MIT,
  BSD).
- Sponsors and core maintainers may submit, but are scored separately
  (*hors concours*) so the prize pool stays focused on the community.

## Questions

- [GitHub Issues](https://github.com/OpenLithoHub/OpenLithoHub/issues/new?labels=hackathon)
  with the `hackathon` label.
- `#hackathon` channel in Discord once the server launches — see the
  [Community](community.md) page.
