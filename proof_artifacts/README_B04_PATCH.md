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


## Increment 16: known-layout run interface compression

A fixed binary raster exterior no longer has to be treated as adversarial
unknown pixels. It can be encoded as occupied horizontal runs and streamed
into a cyclic row-prefix oracle.

For the frozen 36x36 centered-square fixture:
- 1296 occupied pixels compress to 36 runs;
- 164 active centers and 24 coherent modes are enclosed by exact-dyadic
  prefix intervals;
- h=8 px uses about 31x fewer far run segments than far occupied pixels.

This supports the two-scale decomposition

    loaded near halo h
      + exact streamed known-layout interface h<r<=R
      + certified residual tail r>R.

If R covers the entire frozen kernel support, the residual tail is zero.
The current implementation certifies integer-center coherent values.
Continuous derivative run-prefix banks and a KLayout/OASIS/GDS RunSource
remain the next load-bearing integration steps.


## Increment 17: derivative run-prefix center jets

The known-layout interface oracle now carries the coherent jet
`value, dx, dy, dxx, dxy, dyy` of the unique periodic trigonometric
interpolant of the frozen complex64 coherent kernels.

All 164 active centers replay for loaded halos 8, 16, and 29 px.  The
near+interface corrected value/gradient/Hessian intervals intersect the
corresponding dense-layout intervals for every mode and every center.

This closes the **center-jet** interface correction.  It intentionally does
not claim a continuous-cell/slab theorem from the K24 interface jet alone:
the generic K24 kernel-L1 third-derivative majorant is noncoercive on an
8 nm cell.  A local higher-derivative envelope is the next proof gate.


## Increment 18: layout-conditioned alias-free cell envelope

Increment 17's center jet is extended across each 8 nm active cell without
using the noncoercive generic kernel-L1 third-derivative hierarchy.

The verifier first forms the exact known-layout coherent 72x72 run-grid
interval from the Increment-16 row-prefix artifact, lifts each coherent mode
to its unique 72-frequency trigonometric interpolant, evaluates the squared
modes on a 144x144 alias-free grid, and builds the total K24 intensity
Fourier ball before taking absolute derivative majorants.

Frozen fixture results:
- layout-conditioned Hessian Frobenius upper: about 1.21114e-3 nm^-2;
- layout-conditioned third-derivative tensor upper: about 5.10944e-5 nm^-3;
- continuous active-cell gradient lower: about 2.34025e-3 nm^-1.

The continuous-cell mathematical gate is therefore PASS.  The remaining
engineering gate is to build this layout-conditioned Fourier envelope
directly from streamed runs without first materializing the complete 72x72
corrected coherent grid.


The same Fourier ball is evaluated at the 164 active centers for the
Hessian. Using the layout-conditioned L3 only for local center-to-cell
inflation gives a worst cell gradient floor above 5.82e-3 nm^-1 and a
minimum curvature hidden-loop scale above 19.55 nm, larger than the 8 nm
cell diameter 11.31 nm. Thus 8 nm continuous transversality and
hidden-loop completeness are both recovered.

The new first failure is architectural: the present proof generator
materializes the full 72x72 corrected coherent grid before constructing its
alias-free Fourier envelope. A production streaming path should instead
accumulate the same layout-conditioned spectrum directly from row runs.


## Increment 19: direct streamed run spectrum

Horizontal runs are accumulated directly into the normalized layout Fourier
spectrum; no corrected coherent spatial grid is required.  Circular
convolution gives `c_E = N^2 * c_h * c_m`, after which the existing
144-grid alias-free intensity proof is reused.

Frozen results:
- coherent midpoint difference vs Increment 18: 4.163e-17;
- intensity midpoint difference vs Increment 18: 2.498e-16;
- Hessian upper: 1.209680990892e-03 nm^-2;
- L3 upper: 5.008705339056e-05 nm^-3;
- local cell gradient floor: 5.864247916084e-03 nm^-1;
- hidden-loop scale: 19.926830636 nm.

The remaining full-chip blocker is the input adapter: KLayout/GDS/OASIS
geometry must emit canonical horizontal runs before full-canvas rasterization.

## External large artifacts (not in git)

The snapshot NPZ files below exceed the repository's 500 KB large-file
gate. They are distributed as release/CI artifacts and must be placed in
this directory (untracked via `.gitignore`) for the artifact-dependent
tests to run; the tests skip when a file is absent.

| File | SHA-256 | Required by |
|------|---------|-------------|
| `B04_K24_HaloKernelSnapshot_2026-09-09.npz` | `b1c2d0c9f23ce6e6a8c931b05970f5a50ef0a16d5663a1ebbe463f191bdf14fd` | Increment 16 run-oracle test, `B04_HaloTail_Replay` |
| `B04_K24_RunPrefixIntervalSnapshot_2026-09-09.npz` | `7ff0322fa01075aa0be6495b50ee6ca7df357cf67672a878464fe1052988a050` | Increment 16 interval certificate replay |
| `B04_K24_DerivativeRunPrefixIntervalSnapshot_2026-09-10.npz` | `a5523c4c693be9380f1636511f38a26b711b18418a346a4ee867b42c20d89379` | Increment 17 derivative run-prefix tests |
| `B04_K24_RunInterface_AliasFreeIntensityFourier_2026-09-10.npz` | `b88741a354707cddbd58aadf8f7f063cb74ead665f14d1edef55b981a5a949d7` | Increment 18 cellwise Fourier tests |
| `B04_K24_DirectRunSpectrum_AliasFreeIntensity_2026-09-10.npz` | `9725e7a01bf6c425a9e626b7236bc943742c8be0bbd30f6acfe23270af0f7f7a` | Increment 19 direct run-spectrum tests |

Note: `B04_Increment15_HaloTail_Certificate_2026-09-09.json` pins the halo
kernel snapshot via `kernel_snapshot_sha256` = the first hash above.
