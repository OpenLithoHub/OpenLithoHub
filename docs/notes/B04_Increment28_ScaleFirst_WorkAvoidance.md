# B04 Increment 28 — Scale-First Work Avoidance and Crossover Gate

## Strategic cutover

The active B04 mainline is industry-first:

```text
large layout
+ no full-chip theorem raster
+ cheap theorem-safe screening
+ expensive work only on survivors
+ explicit work/memory accounting
+ continuous certificate kept separate from benchmark physics
```

Increment 27 closes a useful mathematical side gate — exact continuous
square-aperture lift plus fixed-source Arb hard-pupil integration — but the
public Sky130 layer contains 1605 canonical runs and correctly triggered the
Arb complexity guard.  That is a signal not to make high-precision global
quadrature the next production bottleneck.

Increment 28 therefore does not open new abstract quadrature theory.

## Repository obstruction exposed by the production path

Before Increment 28:

- `WorkAccounting` exists as a data structure;
- `verify_layout()` exists as a public audit API;
- `run_streaming()` still executes `read_window -> forward` for every tile;
- `WorkAccounting` is not integrated into the actual `run_streaming()` loop;
- there is no fail-closed pre-forward screening policy.

Therefore current repository instrumentation cannot yet answer the central
industrial question:

> How much expensive work was actually avoided?

Increment 28 closes that architecture gap.

## New fail-closed screening contract

A `TileScreeningPolicy` runs before `TileSource.read_window()` and before the
forward simulator.

A tile may be skipped only if the decision is:

```text
status = SCREENED_OUT
certified = true
fill_value = exact trusted-core output
```

If verification plugins are present, the screen may bypass them only when it
explicitly declares `certifies_verifiers=true` and supplies a certified upper
bound.  Otherwise the pipeline falls back to the ordinary active path.

The first built-in policy is intentionally narrow:

```text
ExactEmptyContextScreeningPolicy
```

It screens only when:

1. the exact-vector source exposes `iter_runs_for_bbox`;
2. the full current read region (core + halo) contains no physical runs;
3. the caller explicitly certifies that the chosen context/halo is sufficient;
4. the caller explicitly certifies the constant response of an all-zero read
   context.

This is not a universal empty-tile theorem for Hopkins diffraction.

## Integrated work accounting

`run_streaming()` now records:

- full-chip pixels;
- unique active core pixels;
- certified screened-out core pixels;
- active/screened fractions;
- `read_window` calls and pixels;
- forward-simulator calls and input pixels;
- screen queries;
- refinement attempts;
- dense-allocation events;
- avoided percentage.

Refinement no longer inflates unique `active_pixels`; repeated work is counted
separately.

The public `verify_layout()` result exposes the same accounting dictionary.

## Four-way benchmark

The scale benchmark compares the same deterministic zero-preserving
finite-support 9x9 forward model under four execution modes:

1. `dense_full`
   - materialize full exact-vector mask;
   - one whole-raster forward.

2. `tiled_raster`
   - materialize full mask;
   - ordinary tiled forward;
   - no pruning.

3. `b04_vector`
   - exact-vector window streaming;
   - no pruning;
   - no full-chip dense input.

4. `b04_selective`
   - exact-vector streaming;
   - certified empty-context screening;
   - forward only on surviving cores.

The halo is exactly the four-pixel support radius of this benchmark forward
model, so an empty read context gives an exact zero trusted core.

This is an architecture/work-avoidance benchmark, not calibrated lithography
physics.

## Size ladders

Synthetic sparse ladder:

```text
512^2
2048^2
4096^2
8192^2
```

Public-geometry ladder:

```text
4096^2
8192^2
16384^2
```

The public ladder places one real `sky130_fd_sc_hd__inv_1` cell hierarchically
inside an increasingly large finite top-level chip.  This exercises real
hierarchy and real mask geometry while intentionally creating a sparse-placement
scaling experiment.

The public ladder must therefore be labelled:

```text
REAL_SKY130_MOTIF_IN_SYNTHETIC_SPARSE_HIERARCHICAL_PLACEMENT
```

It is not a claim about realistic full-chip standard-cell density.

## Dense-allocation safety guard

Dense/full-raster modes run only up to a configurable maximum size, default
4096 pixels per side.  Larger levels still run both vector modes.

The benchmark must not intentionally allocate a dangerous full-chip raster
merely to demonstrate that the B04 architecture avoids one.

## Metrics

Every row reports:

- end-to-end wall time;
- peak process RSS;
- structural dense-input bytes;
- structural maximum streaming-window bytes;
- full-chip pixels;
- global run count;
- active verified pixels/fraction;
- screened-out pixels/fraction;
- forward calls;
- forward input pixels;
- read-window calls/pixels;
- dense allocation events;
- tile count;
- process points;
- rigorous-cell count.

Crossover fields are populated only when measured wall time actually supports
the claim.

## Correctness witness

A 512x512 synthetic case is run with full dense output and with selective
streaming output.  The maximum absolute discrepancy must stay below the
configured numerical tolerance before any performance row is accepted.

## Stop / go interpretation after the return

`PROMOTE` is not automatic even if the sparse ladders are impressive.

The next decision is:

### Continue toward PROMOTE if

- selective output correctness passes;
- no dense allocation occurs in the B04 selective path;
- active fraction decreases materially with layout size;
- forward calls fall with active area rather than full area;
- memory separates structurally and in measured RSS;
- a reproducible wall-time crossover appears;
- real Sky130 hierarchical sparse placement shows the same direction.

### Redirect if

- vector screening overhead consumes most of the saved forward work;
- crossover exists only because the synthetic forward is artificially costly;
- public real-geometry parsing dominates;
- advantage disappears when active fraction is not extremely sparse.

### Pause the industrial claim if

- no meaningful crossover appears before the dense safety limit;
- or memory/work-avoidance gains do not translate into operational throughput.

## Claim firewall

Even a strong Increment-28 result does **not** prove:

- foundry-calibrated physics;
- production full-chip OPC signoff speedup;
- continuous EPE/CD/process-window certification at 16k scale;
- realistic full-chip layout-density advantage.

It does establish, if the gate passes, a much stronger engineering fact than
previous toy benchmarks:

```text
the actual OpenLithoHub streaming pipeline can avoid certified-expensive
forward work before tensor materialization, and can account for that avoided
work end-to-end.
```
