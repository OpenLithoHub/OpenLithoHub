# B04 Increment 29 — Realistic-Density Routed-Block External Validity

Increment 28 demonstrated strong work avoidance on sparse layouts, including a
real Sky130 standard-cell motif placed sparsely inside synthetic large domains.
That is useful but insufficient for an industrial promotion decision.

Increment 29 attacks the exact failure mode named in the industry-value
protocol:

> scaling gain exists only on synthetic sparse layouts.

## Pinned public fixture

Dataset:

```text
SJTU-YONGFU-RESEARCH-GRP/PDB-Physical-Design-Database
commit 9e1e3399b1b707f26fee853bce1ff91ab466ce24
layout/sky130hd/ibex/ibex.gds
layout/sky130hd/ibex/ibex.json
```

The public metadata identifies:

```text
design: ibex
top cell: ibex_core
PDK: sky130hd
standard-cell count: 15515
core area: 300532 um^2
stdcell area: 173954 um^2
stdcell utilization: 0.578821
```

The GDS is an OpenROAD-generated routed physical design, not a single-cell
placement invented for B04.

## Deterministic crop rule

The benchmark uses nested center crops of the real routed block at 1 nm exact
vector sampling:

```text
4096^2
8192^2
16384^2
```

and a 32768^2 screen-only diagnostic.

The crop is exposed through `ExactVectorCropSource`, which translates local
window/run queries into the parent GDS coordinate system.  No parent or crop
full raster is materialized by the vector path.

## Primary question

Increment 29 is deliberately hostile to the current screening strategy:

```text
Does exact-empty-context screening still avoid enough forward work
inside a ~58% utilized real routed block?
```

This is more important than obtaining another favorable speed curve.

At the largest fully executed crop define

\[
a_{\rm real}
=
A_{\rm active}/A_{\rm crop}.
\]

The decision rule is:

```text
a_real <= 0.50     STRONG
0.50 < a_real <= .80  MODERATE
a_real > 0.80      WEAK
```

These thresholds are engineering stop/go thresholds, not mathematical
constants.

## If STRONG

Keep exact-empty screening and proceed to survivor-only continuous verifier
integration.

## If MODERATE

Retain empty screening but add a stronger theorem-safe screen for nonempty
tiles before promoting the industrial claim.

## If WEAK

Do not spend effort optimizing exact-empty screening.  Redirect the next
work-avoidance increment to nonempty-tile screening, for example:

```text
cheap nominal field / geometric margin
-> certified no-threshold-band condition
-> skip continuous contour machinery
```

The intended invariant is not "empty geometry"; it is "provably irrelevant to
the requested lithography metric".

## Baselines and timing

At 4096^2 the benchmark runs:

```text
dense_full
tiled_raster
b04_vector
b04_selective
```

At 8192^2:

```text
b04_vector
b04_selective
```

At 16384^2:

```text
b04_selective
```

The 32768^2 level performs screening only, to measure active-set discovery
without risking billions of forward pixels if the real block is dense.

Every worker reports both:

```text
parent_load_wall_s
execution_wall_s
end_to_end_wall_s
```

so a favorable post-parse result cannot hide an expensive real-GDS parser
cost.

## Claim firewall

This benchmark may support:

- work-avoidance external validity on a real routed block crop;
- real-GDS parser cost;
- no-full-raster scaling for the vector path;
- real-density active fraction;
- common-size wall-time comparisons.

It does not support:

- foundry-qualified wafer prediction;
- entire-Ibex full-chip verification;
- calibrated Hopkins speedup;
- continuous EPE/CD/process-window certification at these scales.

Its main scientific value is the stop/go decision on the *screening invariant*.
