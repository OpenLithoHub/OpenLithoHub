# GPU Issue G2 — Large-Layout Single-GPU Scale (Lane A)

## Purpose

Execute the frozen Lane-A ladder (Ibex; optionally Microwatt bounded
ladder + full-die stress) on ONE GPU, fully streaming / out-of-core.

## Prerequisite

```text
FROZEN_SCALE_COMMIT: <FULL_40_HEX_SHA>
G1: PASS
```

## Execution

Runbook step 5 **Command A** (and Command C for the optional Microwatt
stress, only if G1 qualified it). Every declared ladder rung must
appear in the run artifacts as either SUCCESS or an explicit
`NOT_RUN_*` / `UNSUPPORTED` status with its reason — silent skips are
impossible by construction and must not be introduced by hand.

Acceptance per SUCCESS row:

```text
status = SUCCESS
correctness_witness_pass = true
aggregate_n == repeat_count
full-die rows use an out-of-core sink only
```

A slow result is valid evidence. Do NOT rerun with different parameters
to obtain a better number.

## Attach

Run workspaces, verifier PASS (`--require-formal`), bundle SHA-256,
final clean-tree confirmation, completion report (selective summary
forbidden — report every rung).
