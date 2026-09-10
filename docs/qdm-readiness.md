# B04 / QDM Readiness Scoreboard

**Current increment:** 29 (Realistic-density routed-block external validity: ExactVectorCropSource + Ibex gate)
**Current HEAD:** `d9d0f8a` (post-Inc28) + streaming architecture (RFC 0008) + Inc16–29 proof modules
**Last updated:** 2026-09-10

This is the canonical live scoreboard required by the B04 architecture brief (§18/§22).
Historical per-increment notes remain in `proof_artifacts/README_B04_PATCH.md`.

---

## Status summary

```text
EXACT-VECTOR PROOF PATH:            PASS
STREAMING FOUNDATION:               PASS
CERTIFIED-HALO FOUNDATION:          PARTIAL (brackets exist, tight closure open)
PLUGIN FOUNDATION:                  PASS
SOURCE-NATIVE VERIFIER FOUNDATION:  PASS (interfaces + reference shell)
CONTINUOUS CERTIFICATION:           PARTIAL (bridge characterized, interval eval OPEN)
PRE-FORWARD SCREENING:              PASS (fail-closed, certified decisions only)
VERIFICATION EXECUTION SEMANTICS:   R17 HARDENED (per-verifier sessions, true subdivision, dual ledgers)
REAL-DENSITY SCREENING VALIDITY:    MEASURED (Inc29 Ibex routed crops, MODERATE verdict)
FULL-CHIP DENSE ALLOCATION:         ABSENT (MetricOnlyTileSink path verified)
ACTIVE-WORK INSTRUMENTATION:        PASS (WorkAccounting driven by run_streaming)
LARGE-LAYOUT MEMORY SCALING:        PASS (memmap in/out, O(tile+batch) verified)
LARGE-LAYOUT SPEED CROSSOVER:       MEASURED (benchmark_b04_inc21/22/23/24/26/28)
REAL GDS/OAS SEMANTICS:             PASS (KLayout-gated semantic tests)
SIMULATOR INTEROPERABILITY:         PASS (backend contract + reference Hopkins)
PHYSICS DOMAIN VALIDATION:          OPEN (synthetic Hopkins, not foundry-calibrated)
INDUSTRIAL VALUE:                   PROMISING
QDM READINESS:                      CONDITIONAL
```

---

## What is proved (with evidence)

### Exact-vector proof path
`EXACT_VECTOR_PIXEL_CENTER_INDICATOR` is enforced by contract
(`ExactVectorMaskContract.__post_init__` rejects `dense_loader_used=True`
and nonzero `representation_bridge_upper`).  The dense/PIL loader is absent
from the proof path.  Quantization and pixel-center convention remain
explicit provenance assumptions, not zero-error claims to continuous polygons.

### Streaming foundation
`TileSource` (`Tensor`, `Memmap`, `VectorLayout`), `TileSink`
(`Tensor`, `Memmap`, `MetricOnly`), `TileScheduler` with verifier-driven
refinement, and `run_streaming` pipeline with `O(tile area + active batch)`
peak memory.  Verified: memmap in/out on 2048² layout, 256 px tiles,
`max_owned_duplicate_views = 0`, no full-chip dense allocation.

### Certified halo (bracket, not tight closure)
`verify/halo.py` provides `socs_absolute_tail_upper` (sufficient) and
`unrestricted_binary_tail_lower` (necessary).  On the 72×72 ArF K=24
snapshot, ε=2.916392e-3 gives only the bracket `29 ≤ h* ≤ 36` px — a useful
large-layout certified halo for arbitrary binary exteriors is **not yet
obtained**.  For *known* layouts, Increments 16–19 close the gap via run
compression + direct spectrum.

### Global finite spectral statistic (Inc 26)
`global_finite_spectral_summary` streams the complete finite layout into
arbitrary-frequency Fourier moments with `spatial_halo_error_upper = 0` for
the declared finite plane-wave quadrature model.  This eliminates the spatial
halo as an error source for that model — the proof obligation moves to
pupil/source quadrature.

### Continuous mask lift + fixed-source Arb pupil (Inc 27)
`continuous_square_mask.py` gives the exact lift
`M̂_c(f) = p² sinc_π(p fₓ) sinc_π(p f_y) M_point(f)` with
`representation_bridge_upper = 0` for the declared square-aperture semantics.
`arb_fixed_source_pupil.py` certifies the one-source-point coherent field as
a rigorous complex ball (python-flint `acb.integral`, hard pupil mapped to a
fixed rectangle).  Independent proof backend; source-plane quadrature and
partial-coherence enclosure remain open.

### Scale-first work avoidance (Inc 28)
`screening.py` adds a fail-closed pre-forward screen: tiles may be skipped
only with a certified `SCREENED_OUT` decision carrying an exact fill value,
and only when the screen explicitly certifies any attached verification
plugins.  `WorkAccounting` is driven by the actual `run_streaming` loop
(unique active area, screened-out area, read-window/forward/screen counters,
`accounted_pct` closure).  `verify_layout()` surfaces
`VerificationResult.work_accounting`.  The Inc28 scale-first ladder
(512²–16384², `dense_full`/`tiled_raster`/`b04_vector`/`b04_selective`)
measures wall time, peak RSS and crossover fields with zero dense-allocation
events on the selective path.

### Realistic-density routed-block validity (Inc 29)
On the pinned public PDB Sky130HD **Ibex** routed GDS (15515 cells, 0.579
utilization) via deterministic nested center crops, exact-empty-context
screening still discharges real work at every scale: active fraction
0.3125 (4096²) → 0.5391 (8192²) → 0.5117 (16384²), with a stable 0.570
screen-only diagnostic at 32768²; peak RSS stays flat (~1.4 GB) while the
dense baseline already needs 2.0 GB at 4096²; zero dense allocations on
the selective path; correctness witness `max_abs_error = 0.0`.  Verdict:
**MODERATE_ON_REALISTIC_DENSITY** — empty screening is retained, and the
next work-avoidance mainline adds nonempty-tile (sensitivity-collar)
screening.  Honest negative: **no wall-time crossover at real density**
in the tested range (selective is slower than dense/tiled per common-size
comparison); the B04 claim at realistic density is certified work
avoidance and bounded memory, not speed.

### Semantic hardening (Inc 22)
`PhysicalInstanceKey` distinguishes source repetition from read repetition.
`KLayoutAlignedRunSource` stamps `physical:<sha256>` owner ids.
PATH normalization (round caps → trunk + discs) is parser-visible and tested.
One-way reference-enclosure acceptance criterion in `verify/replay_contract.py`.

### Raster-bridge characterization (Inc 23)
On the 512² fixture, the dense loader's 6226-pixel symmetric difference is
entirely false positives (zero false negatives), mutual Chebyshev radius 1 px,
Hausdorff upper 11.31 nm @ 8 nm pixels.  The module does not assert
loader/exact-vector equivalence.

### Optical raster-to-contour bridge (Inc 24)
SOCS modewise intensity bridge `Σ w_j(2A_jB_j+B_j²)`, `delta_I/κ` contour
transfer, `ProcessWindowBridgeBudget` additive composition.  Periodic 128 px
Hopkins tile diagnostic: max intensity delta ~0.17, EPE ~24 nm.  Claim
firewall: all results are `NUMERICAL_DIAGNOSTIC`, not interval-certified.

### Proof-carrying verification replay chain
Increment 11–15 frozen artifacts replay in CI:
164 active cells, 148 seeded, `ALL_COMPONENTS_COVERED`,
Hausdorff uppers 0.976/1.056/2.032 nm.
`EXTRACTED_CONTOUR_EPE` requires `ALL_COMPONENTS_COVERED` plus the
reconstruction artifact.  `epe_max_nm` semantics untouched.

---

## Remaining blockers to QDM READY

1. **Pupil/source quadrature remainder**: interval-certified evaluation of
   the frozen Hopkins/SOCS operator on the compact optical frequency domain
   (Increment 26 defines the finite moment vector; Increment 27 certifies the
   single-fixed-source coherent field ball — composing it over a certified
   source-plane quadrature is the next proof obligation).
2. **Certified oriented-slab spatial extraction**: wire the seed/slab
   theorem's `η_sp` into `TileVerificationResult.spatial_extraction_error`.
3. **Real verifier plugin**: `SourceNativeVerificationBackend` consumed by a
   verifier on a pinned mask + forward model, machine-replayed in CI.
4. **Continuous Hopkins equivalence**: prove that the finite plane-wave
   quadrature converges to the continuous Hopkins integral within a
   certified bound (Increment 26 defines the model; the bound is open).
5. **Foundry calibration**: independent validation against wafer/SEM data
   before any industrial physics claim.

---

## What is explicitly NOT proved

- zero error to arbitrary continuous polygon/wafer models
- foundry-calibrated optical accuracy
- practical full-chip speedup over dense raster at production scale
- certified process-window on real layouts
- complete GDS/OASIS semantic coverage
