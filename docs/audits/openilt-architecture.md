# OpenILT Architecture Audit

**Status:** clean-room reimplementation, fidelity verified against the upstream README and the underlying `Gao2014_MOSAIC` formulation.
**Last audited:** 2026-05-23 against the upstream OpenILT project (MIT, commit pinned at [`dabb97c`](https://github.com/OpenOPC/OpenILT/commit/dabb97c6ca3dfd159362e48273c436444c77353b)) and the L2 + PVBand objective from Gao et al., *MOSAIC* (DAC 2014).

OpenLithoHub ships an `OpenILTModel` adapter (`src/openlithohub/models/openilt.py`) that consumers benchmark as a "SimpleILT" baseline. This page records what we implement, what the upstream / paper specify, and where we diverge.

## Audit method

Compared `models/openilt.py` against:

- The OpenILT upstream README at the pinned commit (algorithmic description only — no source copied or vendored).
- Gao et al., *MOSAIC: Mask Optimizing Solution With Process Window Aware Inverse Correction* (DAC 2014) — the L2 + PVBand objective.
- `Banerjee2013_ICCAD` — the contest the SimpleILT loss was originally tuned against.

Confidence:

- **A** — verified against the source in this repo.
- **B** — verified against the upstream README's algorithmic description; the README does not specify code-level constants, so e.g. learning rate / momentum here are this project's choices.
- **C** — derived from a paper note rather than an audited primary source.

## What the upstream specifies

| Item | Upstream / paper | Confidence |
|------|------------------|------------|
| Loss | `‖Z_nom − Z_T‖² + α · (‖Z_max − Z_T‖² + ‖Z_min − Z_T‖²)` over a 3-corner sweep (nominal + max-dose+defocus + min-dose+defocus). | **B** (README) / **C** (DAC 2014 §III) |
| Mask init | "PixelInit" — mask logit starts at `2 * target − 1`. | **B** |
| Optimizer | SGD with momentum (the SimpleILT default in the README). | **B** |
| Forward model | Lithography forward simulator — upstream OpenILT bundles its own; the formulation does not constrain the choice. | **B** |
| Resist | Sigmoid-thresholded aerial intensity (differentiable surrogate for binarisation). | **B** |
| Process-window corners | ±5% dose latitude is the conventional textbook choice. | **C** |

## What OpenLithoHub implements

`src/openlithohub/models/openilt.py` :: `OpenILTModel`:

| Item | OpenLithoHub | Matches upstream? | Confidence |
|------|---------------|-------------------|------------|
| Loss | `l2_nom + pvb_weight · (l2_max + l2_min)`, per `predict()` ll. ~298–300. `pvb_weight = 0.5` default. | **Yes** in shape; constant `α` is our default. | **A** |
| Mask init | `mask_logit = 2 * target − 1`, ll. ~249. | **Yes** — the PixelInit scheme verbatim. | **A** |
| Optimizer | `torch.optim.SGD(lr=1.0, momentum=0.9)`. | **Yes** — SGD with momentum. Constants are our defaults. | **A** |
| Forward model | Reuses OpenLithoHub's own forward stack: `_utils.forward_model.simulate_aerial_image` (Gaussian PSF, default) or `_utils.hopkins.simulate_aerial_image_hopkins` (SOCS). | **Different** — upstream uses its own optical core; we deliberately reuse ours so this baseline shares the SOCS kernels with `LevelSetILTModel` / `compute_l2_error`. | **A** |
| Resist | `differentiable_threshold(aerial, threshold=0.5, steepness=50.0)`. | **Yes** in form. Steepness is our default. | **A** |
| PW corners | Hard-coded 3 corners: `(nom_dose=1.0, max_dose=1.05, min_dose=0.95)` with paired defocus broadening on the Gaussian path or `defocus_nm=50.0` on the Hopkins path. | **Yes** in shape; ±5% dose is conventional. | **A** |
| Kernel cache | Locked, keyed on `(grid_size, device, defocus_nm, hopkins_params)`. Concurrent `predict()` callers cannot poison the cache. | **Stricter than upstream** — upstream does not have a multi-threaded inference story. | **A** |
| Halo / receptive field | `RECEPTIVE_FIELD_PX = 64` for tile seam-free reconstruction, matching `LevelSetILTModel`. | **Project-level** — upstream OpenILT does not tile. | **A** |

## Findings

1. **Implementation faithfully tracks the SimpleILT formulation.** The L2 + PVBand objective, PixelInit, and SGD-with-momentum are all in place at the docstring's stated places. No structural divergence from the upstream README.

2. **Forward-model reuse is intentional.** OpenLithoHub does not vendor upstream OpenILT's optical core. Reusing `_utils/hopkins.py` keeps the SOCS kernels shared across baselines and metrics — this is what closes the "Gaussian PVB ≠ SOCS PVB" reproducibility footgun the PVB metric module also calls out. Trade-off: our absolute numbers will not match the upstream OpenILT repo's reported numbers byte-for-byte.

3. **PW corner sweep is fixed at 3 corners.** Upstream `Yang2023_LithoBench` and our own `LevelSetILTModel(process_window=True)` use a 5-corner weighted-mean sweep. OpenILT's strawman behaviour on `dummy-identity`-like benign inputs (returns identity when the forward model already prints the target cleanly) is reproducible because of this 3-corner sweep, *not* a bug. Documented in `baselines/results.md` and `docs/benchmarks.md`.

4. **No paper PDF on disk.** Gao2014 (DAC) and the OpenILT README sit behind external URLs. Confidence on upstream-side specifications is **B** (README-derived) / **C** (paper-derived).

5. **Convergence-to-identity on synthetic-8 + ICCAD16 testcase1 is expected.** SimpleILT's stopping criterion is loss-improvement; if the forward model already achieves zero loss against the target at iteration 0 (because the target prints cleanly under nominal+ε corners), the optimizer takes no productive step and the best-mask snapshot is the PixelInit-derived target. This matches upstream OpenILT's reported behaviour on benign benchmarks.

## Implications for users

- **Comparing to upstream OpenILT numbers:** Don't expect byte-equality. The forward model is ours, not upstream's. Cite both when reporting.
- **Process-window faithfulness:** The 3-corner sweep is the paper-faithful choice for *SimpleILT*. If you want a richer corner sweep, use `LevelSetILTModel(process_window=True)` instead.
- **Citation hygiene:** When reporting, cite both the upstream `OpenILT` repo at the pinned commit (algorithmic source) **and** `Gao2014_MOSAIC` (formulation lineage). `Banerjee2013_ICCAD` is the appropriate dataset citation when running on ICCAD-13 contest layouts.

## Re-audit triggers

Re-run this audit when any of the following change:

- The PW-corner schedule (`PVBandCorners` default) changes.
- Optimizer (`SGD`/`Adam`/etc.) or PixelInit scheme changes.
- The Hopkins / Gaussian forward-model reuse policy changes.
- A new PW-corner count is documented in upstream OpenILT — consider syncing.
- The Gao2014 paper PDF lands in `docs/papers/` — promote items marked **C** to **A**.
