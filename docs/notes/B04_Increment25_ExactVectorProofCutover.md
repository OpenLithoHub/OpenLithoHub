# B04 Increment 25 — Exact-Vector Proof-Facing Cutover

Increment 24 showed that the legacy dense GDS rasterization is not a small
representation perturbation on the tested fixture:

- max K24 aerial-intensity difference: about 0.177;
- threshold: 0.225;
- diagnostic `delta_I / kappa_grid`: about 133--191 nm;
- diagnostic printed-raster EPE maximum: 24 nm.

These numbers do not prove a lower bound on the best possible certified bridge,
but they make the dense-loader certification branch strategically
noncompetitive for a 2--3 nm theorem budget.

The proof-facing mask semantics are therefore frozen as:

```text
GDS/OASIS
-> KLayout parser only
-> exact pixel-center vector indicator
-> canonical horizontal runs
-> direct normalized run spectrum
-> Hopkins/SOCS coefficient chain
-> interval continuous geometry
```

The legacy dense loader remains supported for public/legacy benchmarking and
diagnostics, but it is no longer an admissible theorem dependency.

## Exact-vector spectrum theorem

Let a square theorem tile have side length `N`, and let its binary occupancy
indicator be partitioned exactly into disjoint row runs

\[
R=\{(y,[a,b))\}.
\]

The direct run accumulator computes

\[
c_m(k_x,k_y)
=
\frac{1}{N^2}
\sum_{(y,[a,b))\in R}
e^{-2\pi i k_y y/N}
\sum_{x=a}^{b-1}e^{-2\pi i k_xx/N},
\]

which is exactly

\[
\frac{1}{N^2}\operatorname{DFT}(m)(k_x,k_y).
\]

No dense GDS rasterizer appears in this identity.

Consequently, relative to the declared
`EXACT_VECTOR_PIXEL_CENTER_INDICATOR` semantics,

\[
E_{\rm representation}=0.
\]

This zero is **not** a statement about the approximation of arbitrary
continuous polygon boundaries by a pixel-center indicator; quantization remains
an explicit provenance assumption.

## Next acceptance gate

Use a pinned public Sky130 GDS cell/layer as real geometry, choose a
square exact-vector crop, and verify:

1. canonical runs -> normalized spectrum;
2. direct spectrum equals a materialized window of the **same exact-vector
   source** to floating replay tolerance;
3. direct spectrum + K24 kernel spectrum reproduces Hopkins aerial intensity;
4. no `load_layout()` / PIL rasterization is used anywhere in the proof path.

This is an algorithmic lithography benchmark under stated synthetic optics,
not foundry calibration.
