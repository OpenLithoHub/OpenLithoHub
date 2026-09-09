# B04 Increment 12 — OpenLithoHub MVP-1 integration patch

Apply these paths on top of pinned commit:

`348fa5d86d5355465af98e2c4ce3deac60081a4c`

The patch is intentionally additive under `src/openlithohub/verify/`, plus
RFC/test files. It does not alter the existing simulator, raster EPE, PV-band,
or leaderboard paths.

Run focused tests with:

```bash
pytest -q tests/test_verify
```

The included proof grid makes the Increment-11 replay regression executable in
this research pack. For a lightweight upstream PR, keep only the golden JSON
manifest and store the NPZ as a release/CI artifact addressed by SHA-256.


## Increment 13: actual boundary coverage

The exact source-native Fourier boundary replay now classifies every face of
all 164 expanded-band-active cells.  The frozen artifact reports:

- 476 unique geometric active edges;
- 148 unique edges with one certified transverse root bracket;
- 328 unique root-free edges;
- 148 SEEDED active cells;
- 16 CURVATURE_CERTIFIED_EMPTY active cells;
- 0 INCONCLUSIVE active cells;
- global coverage `ALL_COMPONENTS_COVERED`.

This closes component-discovery completeness.  It intentionally does **not**
upgrade `EXTRACTED_CONTOUR_EPE`: a certified oriented-slab reconstruction
error is still required for that target.


## Increment 14: certified nominal contour reconstruction

All 148 seeded cells are certified as one fixed-normal implicit graph. The
midpoints of their two boundary-root brackets form one 148-segment closed
polyline. The certified bounds are:

- nominal reconstruction Hausdorff upper: `1.056042021173 nm`;
- K=24 focus-window contour to reconstruction upper: `2.032478056570 nm`.

`EXTRACTED_CONTOUR_EPE` is now permitted only when this reconstruction
artifact and `ALL_COMPONENTS_COVERED` are both present. In this B04 schema
that target means distance to the certified nominal reconstruction, **not**
design-target EPE and not raster Sobel `epe_max_nm`.


## Increment 15: certified-halo architecture and unrestricted-exterior no-go

The theorem-facing halo logic is intentionally separate from
`workflow.halo.compute_halo_px`.  Existing OIR/receptive-field halo selection
remains a workflow heuristic; it is not relabeled as a proof.

For a frozen normalized SOCS kernel family, the verifier stores both:

- a sufficient absolute-tail upper bound
  `U(h) = Σ w_j (2 A_j T_j(h) + T_j(h)^2)`;
- a necessary unrestricted-binary lower bound
  `L(h) = max_j w_j (T_j(h)/π)^2`.

On the 72×72 ArF K=24 snapshot, epsilon `2.916392e-3` gives only the bracket
`29 <= h_* <= 36` pixels.  The upper proof becomes small only at the full
half-tile radius, so a useful large-layout certified halo is **not yet
obtained** for arbitrary binary exterior perturbations.

The next route must either certify an exterior regularity/interface class
(e.g. polygon/TV/edge-density control) or use a known-layout streaming tail
oracle.  No exponential spatial decay is assumed.
