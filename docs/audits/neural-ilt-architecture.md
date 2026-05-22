# Neural-ILT Architecture Audit

**Status:** documented divergence — see "Findings" below.
**Last audited:** 2026-05-22 against `Jiang2020_NeuralILT` (ICCAD'20, DOI [10.1145/3400302.3415649](https://doi.org/10.1145/3400302.3415649)) — see `docs/references.bib`.

OpenLithoHub ships a `NeuralILTModel` adapter (`src/openlithohub/models/neural_ilt.py`) that consumers benchmark against the paper's reported numbers. This page is the audit trail recording **what we implement, what the paper specifies, and where they intentionally diverge**. It exists so that future readers can decide — without re-deriving the comparison from scratch — whether divergences should be closed, kept, or carried as known limitations.

## Audit method

Compared the OpenLithoHub implementation against:

- The paper text (Jiang et al., ICCAD'20).
- Figure 4 of the paper (the U-Net architecture diagram).
- §3.2 (loss function), §3.3 (ILT correction layer), §4 (experimental setup).

## What the paper specifies

| Item | Paper specification |
|------|---------------------|
| Backbone | 4-level U-Net (encoder + decoder + skip connections), Fig. 4. |
| Encoder channels | 64 → 128 → 256 → 512 at the four levels (Fig. 4 channel labels). |
| Decoder channels | Mirror of encoder, with skip-concat at each level. |
| Activation | ReLU. |
| Output head | 1×1 conv → sigmoid → mask logits in [0, 1]. |
| ILT correction layer | A **differentiable forward-litho block appended to the U-Net** so the loss can flow back through the simulator. §3.3, Fig. 4 right-hand block. |
| Loss | L2 wafer error + complexity regulariser. §3.2, Eq. (4). |

## What OpenLithoHub implements

`src/openlithohub/models/_unet.py` :: `UNet`:

| Item | OpenLithoHub | Matches paper? |
|------|---------------|---------------|
| Backbone | 4-level U-Net (encoder + decoder + skip) | **Yes.** |
| Encoder channels | 32 → 64 → 128 → 256 | **No** — half the paper's widths. |
| Decoder channels | Mirror of encoder | **Yes** (relative to OpenLithoHub's encoder). |
| Activation | ReLU | **Yes.** |
| Normalisation | BatchNorm after each conv | Paper uses BN per Fig. 4 caption. **Yes.** |
| Output head | 1×1 conv → sigmoid (`predict()`) / sigmoid via `_NeuralILTExportWrapper` (export) | **Yes** for the U-Net output; sigmoid is applied by the caller, not inside the module. |
| ILT correction layer | **Not implemented.** The model treats mask prediction as a direct forward pass; loss-through-simulator training happens in user code. | **No** — paper-faithful re-training would need this block. |
| Loss | Not part of the model adapter — the leaderboard scores L2 + PVB on a separately-run forward simulator (`Yang2023_LithoBench`-style, see [citations.md](../citations.md)). | **Different but consistent**: scoring is done by the benchmark surface, not by the model. |

## Findings

1. **Channel widths are halved.** This is an intentional inference-budget choice — the deployed weights (`openlithohub/neural-ilt-v0.1`) were trained at this size to fit on commodity GPUs with `RECEPTIVE_FIELD_PX = 64` tile inputs. Not a bug. Documented here so users comparing to Jiang2020's reported metrics know the OpenLithoHub baseline is a smaller model. If we publish a paper-faithful re-training, it should land as `neural-ilt-v0.2` (full widths) to keep the v0.1 weights reproducible.

2. **ILT correction layer is missing.** The paper's headline contribution is end-to-end training through a differentiable simulator. OpenLithoHub's `NeuralILTModel` only provides the U-Net half. This is a **functional gap** — users who want the paper's training behaviour cannot get it from this adapter alone. The leaderboard's forward-sim gate (`tracker.py::_require_forward_simulation`) compensates at *eval* time but not at training time.

   Tracking issue: open follow-up RFC ("Neural-ILT correction-layer training adapter") if there is demand for paper-faithful re-training inside OpenLithoHub.

3. **Output sigmoid is applied at the caller.** Our `_unet.UNet.forward()` returns logits; `predict()` and the export wrapper apply `sigmoid`. The paper diagram shows sigmoid inside the network. Functionally equivalent, but worth noting for anyone diffing the `state_dict` keys.

## Implications for users

- **Comparing to Jiang2020 numbers:** OpenLithoHub's v0.1 baseline is a smaller model. Don't read a ~5–10% L2 gap as a methodology defect — it's a model-size gap.
- **Training your own Neural-ILT inside OpenLithoHub:** the U-Net is here, but the correction layer isn't. Train against `openlithohub.simulators.HopkinsSim` in your own loop, or wait for the v0.2 adapter.
- **Citation hygiene:** when reporting results that use this adapter, cite both `Jiang2020_NeuralILT` (architecture lineage) **and** the v0.1 weights tag (model-size disclosure).

## Re-audit triggers

Re-run this audit when any of the following change:

- Channel widths in `_unet.UNet.__init__()`.
- `NeuralILTModel.predict()` signature or output post-processing.
- The bundled weight repository (`openlithohub/neural-ilt-v0.1`) is retrained or replaced.
- A second Neural-ILT adapter is added with the correction layer.
