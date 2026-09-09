# QDM / B04 Plugin Readiness Checklist

Status after RFC 0008 foundation + B04 Increment 14 integration
(2026-09-10). This checklist is the honest scoreboard required by the
B04 architecture brief (§23).

## Foundation statuses

```text
STREAMING FOUNDATION:            PASS
CERTIFIED-HALO FOUNDATION:       PASS
PLUGIN FOUNDATION:               PASS
SOURCE-NATIVE VERIFIER FOUNDATION: PASS (interfaces + reference shell only)
CONTINUOUS-METRIC FIREWALL:      PASS
B04 ACTUAL-MODEL READINESS:      READY
QDM READINESS:                   CONDITIONAL
```

## Evidence map

| Requirement | Evidence |
| --- | --- |
| A. big-layout streaming under budget | `tests/test_streaming/test_pipeline.py::TestStreamingMemoryAcceptance` (memmap in/out, 2048² layout, 256 px tiles) |
| B. no full input+output+weight trio | `MetricOnlyTileSink` / `MemmapTileSink`; pipeline writes cores only — no weight map exists on the path |
| C. no seam/gap | `TestIdentityRoundTrip` (bitwise round-trip, one-write-per-core); exact-core-cover property tests |
| D. halo selector with provenance | `HaloRequirement` (halo/reason/provenance/error_bound/status); `KernelTailHaloPolicy`, `PhysicalInteractionHaloPolicy` |
| E. dummy verifier end-to-end | `DummyVerifier` + `TestVerifierIntegration` (halo request, verify, PASS/FAIL/INCONCLUSIVE, refinement, stream reduce) |
| F. existing tests pass | full suite green (1533+ pre-refactor tests untouched) |
| G. old vs streaming benchmark | `benchmarks/benchmark_streaming.py` |
| H. source-native foundation | `verify/source_snapshot.py`, `verify/source_native.py` (opt-in `source_native_full` backend, no top-K dependency) |
| I. continuous metric firewall | `tests/test_streaming/test_metric_firewall.py`; `epe_max_nm` semantics unchanged |
| J. budget composition | `StreamingVerificationReducer.error_budget` + `test_error_budget_composition` |
| K. coverage semantics | `PARTIAL_COVER != PASS` pinned in reducer tests and `verify/spatial.reduce_coverage` |
| B04 actual model | `tests/test_verify/` replays the frozen Increment 11–14 artifacts (164 cells, 148 seeded, `ALL_COMPONENTS_COVERED`, Hausdorff uppers 0.976/1.056/2.032 nm) |

## Why QDM READINESS is CONDITIONAL

- No QDM mathematics is implemented (explicitly out of scope this round).
- `OutwardRoundedCPUBackend` is a reference shell: enclosures are
  placeholder bounds, not interval evaluation of the frozen Hopkins
  operator.
- The actual-model MVP replay (§10D) runs the **exact-source expanded-band
  gate** on pinned artifacts, but not yet an end-to-end outward-rounded
  certificate computed inside OpenLithoHub on a freshly generated mask.

## Remaining blockers to QDM READY

1. Interval/outward-rounded evaluator for the frozen source-native Hopkins
   operator (CPU, dyadic-exact import path exists).
2. Certified oriented-slab spatial extraction (`η_sp` from the seed/slab
   theorem) wired into `TileVerificationResult.spatial_extraction_error`.
3. `SourceNativeVerificationBackend` consumed by a real verifier plugin on
   a pinned mask + pinned forward model, machine-replayed in CI.
