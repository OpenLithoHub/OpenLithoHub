# GPU Issue G3 — 1/2/3-GPU Scaling (Lane B)

## Purpose

Execute the frozen scaling protocol: same fixture (the Ibex scale
fixture, layer 66:44 — the manifest committed with the scale freeze;
prepared per runbook step 3), same tile geometry, same P1 forward,
same output semantics at 1, 2 and 3 GPUs.

## Prerequisite

```text
FROZEN_SCALE_COMMIT: <FULL_40_HEX_SHA>
G1: PASS
G2: PASS (single-GPU baseline from the same fixture)
```

## Execution

Runbook step 5 **Command B** — the `for GPUS in 1 2 3` loop, unmodified.

The harness records per-run wall/fwd metrics; T1 is the 1-GPU row of
THIS issue (never a reuse of another run's numbers).

Acceptance:

```text
each of the three runs: SUCCESS + correctness witness + verifier PASS
worker topology covers all tiles at every GPU count
per-GPU peaks recorded for 2/3-GPU runs
```

## Attach

The three run workspaces, three verifier PASS outputs, bundle SHA-256,
final clean-tree confirmation. The maintainer computes speedup /
efficiency from the artifacts — the operator reports raw evidence only.
