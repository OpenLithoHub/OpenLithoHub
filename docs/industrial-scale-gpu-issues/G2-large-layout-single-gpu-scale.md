> **STATUS: SUPERSEDED — HISTORICAL 3×RTX 3080 TEMPLATE.**
> The hardware plan moved to 1×RTX 4090; the live contracts are the
> G*-4090 templates. This file is retained as frozen-protocol
> provenance. DO NOT EXECUTE from this template.

---

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

Runbook step 5 **Command A** (and Command C for the Microwatt stress,
only if G1 qualified it).

Frozen Microwatt stress fixture facts:

```text
PDB commit:      9e1e3399b1b707f26fee853bce1ff91ab466ce24
GDS sha256:      b0253af06f35d1a8b11c2a47f70aac89f33be53e8f28da844b35c2ad6cc92a6d
selected layer:  66:44
die size px:     3,020,000 × 3,610,000 @ 1 nm/px
dense float32 raster-equivalent: 43,608,800,000,000 bytes (derived — never materialized)
```

Every declared ladder rung must
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
