# GAN-OPC Architecture Audit

**Status:** documented divergence — see "Findings" below.
**Last audited:** 2026-05-23 against `Yang2018_GANOPC` (DAC'18, DOI [10.1145/3195970.3196056](https://doi.org/10.1145/3195970.3196056)) — see `docs/references.bib`.

OpenLithoHub ships a `GanOpcModel` adapter (`src/openlithohub/models/gan_opc.py`) that consumers benchmark against the paper's reported numbers. This page is the audit trail recording **what we implement, what the paper specifies, and where they intentionally diverge**. It exists so that future readers can decide — without re-deriving the comparison from scratch — whether divergences should be closed, kept, or carried as known limitations.

## Audit method

Compared the OpenLithoHub implementation against:

- The paper text (Yang et al., DAC'18) — *paper PDF not on disk; claims below are derived from the citation note in `docs/references.bib`, the docstring of `models/gan_opc.py`, and the published successor TCAD'20 paper which restates the same architecture*.
- §IV (architecture) — generator + discriminator + ILT-guided loss.
- §V (experimental setup) — training data, optimisation hyperparameters.

Confidence: the items below marked **C** (citation-derived) are taken from the docstring's self-reported confidence level and have **not** been independently verified against the paper PDF. Items marked **A** (audited) are verified against the source code in this repo.

## What the paper specifies

| Item | Paper specification | Confidence |
|------|---------------------|------------|
| Generator backbone | Encoder-decoder with skip connections (U-Net silhouette). 4 down-sample / 4 up-sample stages per Fig. 4 (the docstring of `gan_opc.py` says "4-level encoder-decoder"). | **C** |
| Encoder channels | Not independently verified. Successor work (NeurIPS 2023 LithoBench Table III) uses 64→128→256→512 for U-Net OPC; assumed equivalent. | **C** |
| Activation | ReLU. | **C** |
| Output head | 1×1 conv → sigmoid → mask in [0, 1]. | **C** |
| Discriminator | Patch-based binary classifier — paper §IV.B. | **C** |
| Loss | Adversarial loss + L2 reconstruction + lithography-aware loss term (§IV.C). | **C** |
| Training data | The 4875 paired (target, mask) PNGs released at https://github.com/phdyang007/GAN-OPC. | **A** — `data/ganopc/extracted/ganopc-data/{artitgt,artimsk}` ships exactly this set. |

## What OpenLithoHub implements

`src/openlithohub/models/gan_opc.py` :: `GanOpcModel` reuses the shared `src/openlithohub/models/_unet.py` :: `UNet`:

| Item | OpenLithoHub | Matches paper? | Confidence |
|------|---------------|---------------|------------|
| Generator backbone | **3-level** U-Net: `inc → down1 → down2 → down3 → up1 → up2 → up3 → outc`. Three down-samples, three up-samples. | **No** — paper is 4-level (Fig. 4); we have 3 levels. **The Neural-ILT audit at `docs/audits/neural-ilt-architecture.md` describes the same `_unet.UNet` as "4-level" — that is itself a depth miscount and should be corrected.** | **A** |
| Encoder channels | 32 → 64 → 128 → 256 (set in `_unet.py:78–81`). | **No** — half the paper's assumed widths. | **A** |
| Decoder channels | Mirror of encoder (256 → 128 → 64 → 32). | **Yes** relative to OpenLithoHub's encoder. | **A** |
| Activation | ReLU (`_DoubleConv.block`, `_unet.py:17`). | **Yes.** | **A** |
| Normalisation | BatchNorm after each conv (`_unet.py:16,19`). | Likely yes (paper does not strongly specify, and BN is the defacto choice for U-Nets of this era). | **A** for code, **C** for paper match |
| Output head | Plain `1×1 conv → logits`. Sigmoid is applied by the caller in `predict()` (`gan_opc.py:131`), not inside the module. | **Functionally yes**, structurally different. | **A** |
| Discriminator | **Not implemented.** Adapter docstring: "This adapter only ships the generator side; the discriminator is a training-time concern." | **No, by design** — discriminator only matters at training time, not at inference. | **A** |
| Adversarial / lithography-aware loss | **Not implemented in this repo.** Training script does not exist as of the audit date. | **N/A** — no training code to audit. | **A** |
| Training data wiring | `data/ganopc/extracted/ganopc-data/{artitgt,artimsk}` is on disk (4875 pairs); `GanOpcDataset` reads it. | **Yes**, dataset side is wired. | **A** |
| Trained weights | `pretrained=True` points at HF Hub `openlithohub/gan-opc-v0.1` — **placeholder, no weights uploaded as of 2026-05-23**. The adapter falls back to a randomly initialised `UNet` with a `warnings.warn("…predictions will be near-random…")` warning. | **No** — adapter is a smoke-test, not a baseline. | **A** |

## Findings

1. **Generator depth is 3, not 4.** The shared `_unet.UNet` runs only three down-samples (`inc → down1 → down2 → down3`) and three up-samples. Both this audit and `docs/audits/neural-ilt-architecture.md` previously called it "4-level"; that's a miscount. **Action:** correct the Neural-ILT audit. Functional implication: receptive field is 8× downsampling, not 16×, so global structure (long lines spanning a 4096² mask) is harder for the network to capture than the original paper architecture.

2. **Generator channel widths are halved.** 32→64→128→256 instead of the (assumed) 64→128→256→512 from the paper. Like Neural-ILT, this is an intentional inference-budget choice for the v0.1 line. Acceptable as a baseline; not paper-faithful.

3. **No trained weights ship with this adapter.** `gan_opc-v0.1` on HF Hub is a placeholder. A run of `scripts/generate_baselines.py` against `data/ganopc/extracted/` produces near-random masks plus a warning. **The dataset is on disk and the network is in code, but no training script exists** — Phase 2 task #3 in `out/plans/external-resource-utilization.md` is to close that loop.

4. **Discriminator and adversarial loss intentionally absent.** This is honest: discriminator is training-time only and inference-only adapters don't need it. But the absence of a training script means the adversarial loss has never run against this dataset inside OpenLithoHub. A future "paper-faithful re-training" needs both the discriminator module and the L_lith term reconstructed.

5. **Paper PDF is not on disk.** Confidence on paper-side specifications is "C" (citation-derived). To upgrade to "A" we would need the paper PDF in `docs/papers/` or equivalent, with this audit re-checked against §IV.

## Implications for users

- **Comparing to Yang2018 numbers:** Don't. The shipped adapter has no weights and the generator is shallower + narrower than the paper. Cite the paper for *lineage*, not for *expected metric*.
- **Training your own GAN-OPC inside OpenLithoHub:** the U-Net is here and the dataset is on disk, but you'll need to write the discriminator + adversarial loop yourself. Track progress under `out/plans/external-resource-utilization.md` task #3.
- **Citation hygiene:** when reporting results from this adapter, cite both `Yang2018_GANOPC` (architecture lineage) **and** the v0.1 weights tag (model-size disclosure). If running without weights, do not cite Yang2018 at all — there is no paper-derived signal in the output.

## Re-audit triggers

Re-run this audit when any of the following change:

- Channel widths or depth in `_unet.UNet.__init__()`.
- `GanOpcModel.predict()` signature or output post-processing.
- The HF Hub `openlithohub/gan-opc-v0.1` weights are published or replaced.
- A `train_gan_opc.py` script lands and the discriminator / loss surface becomes part of the codebase.
- The Yang2018 paper PDF is added to the repo and items marked **C** above can be promoted to **A**.
