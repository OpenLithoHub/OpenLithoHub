# GPU Issue G4 — Optional Hopkins Compute Stress (P2, bounded)

## Purpose

Compute-heavy characterization ONLY: bounded-window Hopkins on the GPU
after the P1 profile is fully green. This is never full-die Hopkins and
never a first-acceptance requirement.

## Prerequisite

```text
FROZEN_SCALE_COMMIT: <FULL_40_HEX_SHA>
G1: PASS
G2: PASS
G3: PASS
```

## Execution

The P2 forward profile and its frozen command will be added to the
runbook by the maintainer ONLY after G3 closes. If this issue is opened
without that runbook section existing, STOP — the protocol is not
frozen yet.

Requirements already frozen by the charter:

```text
fixed Hopkins grid
fixed precomputed kernels
fixed optical parameters
same tile size / dtype as the P1 protocol
same correctness witness rules
no full-die Hopkins acceptance
```

## Attach

Standard evidence set (logs, workspaces, verifier PASS, bundle SHA-256,
clean-tree confirmation).
