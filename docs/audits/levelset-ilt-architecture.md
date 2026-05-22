# LevelSet-ILT Architecture Audit

**Status:** continuous-mask gradient-descent ILT, faithful to the level-set / SimpleILT-style formulation surveyed in `Yang2023_LithoBench` §3.3.
**Last audited:** 2026-05-23 against the formulation in `Yang2023_LithoBench` (NeurIPS 2023, [openreview.net/forum?id=jWHU4b7Yk6](https://openreview.net/forum?id=jWHU4b7Yk6)) and the lineage papers cited in the model docstring.

OpenLithoHub ships a `LevelSetILTModel` adapter (`src/openlithohub/models/levelset_ilt.py`) used as the canonical ILT-via-gradient-descent baseline. This page records what we implement, what the surveyed formulation specifies, and where they diverge.

## Audit method

Compared `models/levelset_ilt.py` against:

- `Yang2023_LithoBench` §3.3 — survey of the SimpleILT-style L2 + total-variation formulation. The model docstring cites this as the open-access substitute for the paywalled Granik / Pang lineage.
- Pang, Liu & Abrams, *Inverse lithography technology principles in practice* (Proc. SPIE 5992, 2005) — original level-set / continuous-mask formulation.
- Poonawala & Milanfar, *Mask design for optical microlithography — an inverse imaging problem* (IEEE TIP 16(3), 2007) — gradient-based inversion of the imaging operator.

Confidence:

- **A** — verified against the source in this repo.
- **B** — verified against the LithoBench survey description; constants here (`lr=0.1`, `sigma_px=2.0`, `tv_weight=0.01`) are this project's defaults, **not literal paper values** — the model docstring already states this explicitly.
- **C** — derived from a paper note rather than an audited primary source.

## What the surveyed formulation specifies

| Item | Survey / paper specification | Confidence |
|------|------------------------------|------------|
| Loss | `‖Z − Z_T‖² + λ · TV(mask)` — L2 fidelity + total-variation regularisation. | **B** (LithoBench §3.3) |
| Mask representation | Continuous, sigmoid-bounded `[0, 1]`. Update via gradient descent on a logit. | **B** |
| Optimizer | Gradient descent — surveyed as "SGD/Adam"; the survey does not pin a choice. | **B** |
| Forward model | Convolution against an optical PSF (Gaussian) or a Hopkins/SOCS partial-coherent kernel. | **B** |
| Resist | Sigmoid-thresholded aerial intensity. | **B** |
| Process window | Multi-corner sweeps (dose × defocus) feed into a weighted-mean fidelity loss. | **C** (Pang2005 / Granik) |

## What OpenLithoHub implements

`src/openlithohub/models/levelset_ilt.py` :: `LevelSetILTModel`:

| Item | OpenLithoHub | Matches survey? | Confidence |
|------|---------------|-----------------|------------|
| Loss (default) | `mse_loss(resist_nom, target) + tv_weight · TV(mask)` per `predict()` ll. ~270–272. `tv_weight=0.01` default. | **Yes** in shape. Constants are our defaults. | **A** |
| Loss (PW mode) | `process_window=True` swaps the nominal-only fidelity term for `workflow.process_window.pw_fidelity_loss` over `DEFAULT_PW_CORNERS`. **Currently only on the `gaussian` forward model** — Hopkins multi-corner sweeps deferred (an explicit `ValueError` enforces this). | **Yes** in shape — multi-corner mean. Hopkins coverage is a known gap. | **A** |
| Mask init | `mask_logit = 4 · target − 2` per ll. ~213. After sigmoid, `≈ 0.12` at target=0 and `≈ 0.88` at target=1 — non-saturated. | **Different from `OpenILTModel`'s `2 · target − 1` PixelInit but the survey does not pin an init.** Our choice gives a slightly stronger initial signal. | **A** |
| Optimizer | `torch.optim.Adam(lr=0.1)`. | **Different from `OpenILTModel`'s SGD-with-momentum** — Adam is the project's choice for this baseline. Survey-compatible. | **A** |
| Forward model | `gaussian` (default) — `simulate_aerial_image(sigma_px=2.0)`. `hopkins` — `simulate_aerial_image_hopkins(...)` reusing `_utils/hopkins.py`. SOCS kernels are cached per `(grid_size, device)`. | **Yes** — both modes match the survey's optical-core options. | **A** |
| Resist | `differentiable_threshold(aerial, threshold=0.5, steepness=50.0)`. | **Yes.** Constants are our defaults. | **A** |
| Hopkins kernel cache | Cached per `(grid_size, device)`, invalidated when `hopkins_params` changes. **Not lock-guarded** — unlike `OpenILTModel`, this baseline is not thread-safe under concurrent `predict()`. | **Project-level** — survey is silent. **Limitation:** flag if exposing this baseline through a multithreaded API. | **A** |
| Compile path | `compile_forward=True` triggers `torch.compile(hopkins_fn, mode="reduce-overhead")` cached per `(shape, device, dtype, forward_model)`. Falls back to eager on torch < 2.0 / no-Triton platforms. | **Project-level optimisation** beyond the survey scope. | **A** |
| Checkpoint / resume | `checkpoint_dir` + `save_freq` + `resume_from`: writes `mask_logit` + Adam state every `save_freq` iterations; resume from any such file restarts the iteration counter and runs the remainder. | **Project-level** — survey is silent. | **A** |
| Halo / receptive field | `RECEPTIVE_FIELD_PX = 64`, matching `OpenILTModel`. | **Project-level** — survey does not tile. | **A** |

## Findings

1. **Implementation tracks the surveyed formulation faithfully.** L2 + TV with sigmoid-bounded mask logit, gradient descent, optical-core convolution, sigmoid-thresholded resist — all present.

2. **Two divergences from `OpenILTModel`** are deliberate and worth knowing:
   (a) Init is `4·target−2` here vs. `2·target−1` in `OpenILTModel`. Stronger initial signal.
   (b) Optimizer is Adam (lr=0.1) here vs. SGD+momentum in `OpenILTModel`. Adam converges faster on the L2+TV loss surface; SGD-with-momentum is the SimpleILT-paper-faithful choice for the L2+PVBand objective. **Both are survey-consistent — the survey does not pin the optimizer.**

3. **Process-window mode does not cover the Hopkins forward.** `predict(process_window=True, forward_model="hopkins")` raises `ValueError`. This is a *documented gap*, not a bug — multi-corner Hopkins sweeps would multiply SOCS kernel rebuilds (defocus changes invalidate kernels per `HopkinsSimulator._hparams_match`). Closing it requires either per-corner kernel caches (mirroring what `compute_pvband(simulator=…)` now does) or a PW-corner sweep that holds defocus constant.

4. **Hopkins kernel cache is not lock-guarded.** Concurrent `predict()` callers can race on the cache rebuild and leave it in a partially-populated state. `OpenILTModel` already addressed this with `threading.Lock`. **Open issue**: backport the lock here if `LevelSetILTModel` is exposed via FastAPI / multithread inference.

5. **`save_freq=N` semantics double-checked.** `it+1` is the count of *completed* steps; resume restarts at `it+1` and runs `iterations - (it+1)` more steps — total work equals an uninterrupted run. The inline comment at ll. ~283–286 is correct.

## Implications for users

- **Default vs. SimpleILT-paper-faithful choice:** if you want the SimpleILT formulation byte-for-byte, use `OpenILTModel`. `LevelSetILTModel` is the "level-set / continuous-mask gradient descent" baseline, more general than SimpleILT.
- **Process-window mode + Hopkins:** unsupported as of 2026-05-23. Use either `process_window=True, forward_model="gaussian"` or `process_window=False, forward_model="hopkins"`.
- **Multithread safety:** not guaranteed. Use a per-process model instance behind any concurrent inference path.
- **Citation hygiene:** when reporting numbers, cite `Yang2023_LithoBench` for the survey lineage. The Pang2005 / Poonawala2007 papers are the deeper lineage references — cite them only if you also vouch for the level-set numerical equivalence (this implementation is closer to "continuous-mask gradient descent" than to Pang's Hamilton-Jacobi level-set evolution).

## Re-audit triggers

Re-run this audit when any of the following change:

- Mask init scheme (`mask_logit = ...`) changes.
- Optimizer or learning-rate default changes.
- `process_window` gains Hopkins forward support.
- Hopkins kernel cache acquires lock guards (or grows multi-corner caches).
- The Pang2005 / Poonawala2007 paper PDFs land in `docs/papers/` — items marked **C** can be promoted to **A**.
