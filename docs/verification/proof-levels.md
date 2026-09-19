# Proof Levels and the Capability Firewall

Theorem-facing statements are graded.  Two independent axes decide what a
result may claim, and both are enforced by code.

## Axis 1 — proof level (`verify/types.ProofLevel`)

| level | meaning |
|---|---|
| `HEURISTIC` | informal / sampling evidence |
| `NUMERICAL-DIAGNOSTIC` | deterministic numerical evidence, no enclosure theorem |
| `INTERVAL-CERTIFIED` | outward-rounded interval proof produced by a rigorous backend |
| `IMPORTED-QDM-CERTIFIED` | replayed from a frozen, hash-pinned certificate (e.g. P-054) |

## Axis 2 — backend capability (`verify/types.CertificationCapability`)

| capability | ceiling |
|---|---|
| `DIAGNOSTIC_ONLY` | `NUMERICAL-DIAGNOSTIC` |
| `RIGOROUS_INTERVAL` | `INTERVAL-CERTIFIED` |
| `IMPORTED_FROZEN_CERTIFICATE` | `IMPORTED-QDM-CERTIFIED` |

Backend **identity** and **capability** are separate: the canonical
`source_native_full` backend id may exist while
`OutwardRoundedCPUBackend` remains `DIAGNOSTIC_ONLY` — its `[0,1]`
placeholder enclosure is explicitly not a rigorous interval result.

## Enforcement points

1. **Assembler** (`verify/certifier.py`): a certifying-level dependency
   whose backend capability ceilings below the claimed level is a hard
   error — a corrupted proof package, not a downgrade choice.  A
   certifying dependency that names a backend without declaring its
   capability also fails closed.
2. **Backend gate** (`verify/source_native.assert_backend_supports`):
   requesting a proof level above the ceiling raises.
3. **Snapshot semantics** (`verify/source_snapshot`): v2 snapshots declare
   `SpectralRepresentation`; `SOCS_TRUNCATED` without a certified error
   bound is inadmissible for theorem-facing use.  (v1 snapshots answer
   `is_topk_truncated = None` — the schema cannot know.)
4. **Replay** (`verify/phase_diagram`): frozen phase diagrams replay only
   as `IMPORTED-QDM-CERTIFIED`; profiles claiming foundry calibration or
   continuous-Hopkins equivalence are rejected at load time.

## Terminology rules

- allowed: "certified event-free chamber", "continuous EPE upper bound",
  "frozen-certificate replay", explicit error budget, fail-closed
  verification;
- not allowed without external wafer/SEM validation: "foundry certified",
  "production qualified", "wafer accurate".
