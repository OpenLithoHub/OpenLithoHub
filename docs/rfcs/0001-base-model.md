# RFC 0001 — Layout-MAE Base Model

| | |
|-|-|
| Status | Draft |
| Author | OpenLithoHub maintainers |
| Created | 2026-05-19 |
| Targets | v0.2 |
| Related | RFC 0002 (Layout Tokens), `openlithohub.synth.DiffusionLayoutGenerator` |

## Summary

A self-supervised masked-autoencoder (MAE) pretrained on rasterised PDK
layouts. The pretrained backbone serves three downstream consumers:

1. **Initialisation for the diffusion-based synthetic layout generator**
   (`openlithohub.synth.DiffusionLayoutGenerator`).
2. **Backbone for OPC/ILT models** that take a target layout and predict
   a corrected mask.
3. **Embedding source for retrieval / clustering** of layouts with
   similar manufacturability profiles.

This RFC defines *only* the design. No weights ship in v0.1; a curated
training corpus and GPU budget are prerequisites that we do not yet
have.

## Why a base model

Computational lithography is data-poor relative to natural images:
- Public PDK layouts are scarce; vendor PDKs cannot be redistributed.
- Each fab/PDK shifts the manufacturable distribution.
- Downstream tasks (OPC, ILT, hotspot detection) have *very* small
  labelled sets — typically tens of thousands of patches at most.

Pretraining a backbone on the *rasterised* layout distribution lets us
trade GPU time once for sample-efficient fine-tuning across all
downstream tasks. MAE specifically is appropriate because layout patches
are spatially redundant and the masking-and-reconstruct objective forces
the model to learn the local feature topology (line ends, T-junctions,
via stacks) that downstream tasks care about.

## Non-goals

- A foundation model competitive with vendor-internal models trained on
  proprietary data. We are explicitly building an *open baseline*.
- Solving OPC/ILT directly — those are downstream applications.
- Generative quality on its own. The diffusion path (RFC 0002 territory
  via the layout-token route) is a separate consumer of these weights.

## Architecture

ViT-S backbone (~22M params) as the v0.2 default. Choices:

- **Patch size**: 16×16 px, on 256×256 input layouts.
- **Embed dim**: 384, depth 12, heads 6.
- **Decoder**: 4-layer ViT, 256 dim — discarded after pretraining.
- **Loss**: per-pixel L1 over masked patches only. Binary masks make L1
  numerically equivalent to a bit-reconstruction objective; L1 is simpler
  to reason about than BCE here because of class imbalance.

We deliberately stay small. Layout patches do not need a 1B-param model,
and a small backbone is cheaper to fine-tune for downstream OPC users on
a single-GPU budget.

### Why ViT and not U-Net

- MAE's masked patch prediction is naturally a transformer pattern.
- Fine-tune use cases benefit from a single backbone + lightweight task
  heads, not from a fully convolutional encoder-decoder.
- The diffusion sampler (RFC 0002) wants sequence-style outputs anyway.

## Pretraining data

Three tiers, in order of preference:

1. **Procedurally generated** layouts from
   `openlithohub.synth.generate_synthetic_batch` — unlimited, MRC-clean
   by construction, but distribution is narrow (3 pattern families × 2
   PDKs).
2. **Public academic releases** — `phdyang007/damo-via-rs`,
   `phdyang007/damo-mb`, ICCAD-style benchmark suites. Each is small
   (≤10k images) and licence-permissive.
3. **User-contributed PDK rasterisations** under a
   contributor-licence-agreement model. v0.2 ships a contribution
   workflow but no curated corpus.

Mixture target for v0.2: 70% synthetic / 30% public academic, with
synthetic acting as a regulariser. This is a *baseline*, expected to be
revised once we measure transfer.

Augmentations: 8-way dihedral group (rotations × flips), random
translation, dose/threshold-style intensity jitter applied *after*
binarisation has been undone (i.e., on the analog raster used as input).

## Training protocol

- **Steps**: 200k at batch 256 on a single A100 — ~36 GPU-hours.
  Reproducible from `make pretrain` once the data tier is set up.
- **Mask ratio**: 75%, MAE default.
- **Optimiser**: AdamW, lr 1.5e-4 (linear warmup 10k steps, cosine to
  1e-6), weight decay 0.05, β=(0.9, 0.95).
- **Mixed precision**: bf16 autocast; gradient clip 1.0.
- **Checkpointing**: every 10k steps; keep last 5; promote best on
  reconstruction L1 over a held-out synthetic eval batch.

## Evaluation protocol

Pretraining is "good enough to ship" iff *all* of:

1. **Reconstruction quality**: held-out L1 ≤ 0.02 on synthetic + public
   eval splits at mask ratio 0.75.
2. **Linear probe — hotspot detection**: linear classifier on frozen
   features beats a from-scratch ResNet-18 on
   `phdyang007/damo-mb` patches (target: ≥+5 F1 absolute).
3. **Fine-tune transfer — OPC**: backbone + small UNet decoder, fine-tuned
   for 5k steps on FreePDK45 OPC pairs, beats the same architecture
   trained from scratch (target: ≥10% lower mean EPE).
4. **MRC-respecting reconstructions**: reconstructed masks pass
   `check_mrc` at the corresponding PDK rules with violation rate ≤ 1%.

Failing any of (2)–(4) means the backbone is *not yet* worth shipping —
the pretraining objective is correlated with reconstruction but not
guaranteed to transfer.

## Public API

```python
from openlithohub.models import LayoutMAE

# Pretrained checkpoint download (HuggingFace Hub).
model = LayoutMAE.from_pretrained("openlithohub/layout-mae-vit-s-256")

# Frozen feature extraction.
features = model.encode(mask_tensor)  # (B, N_tokens, embed_dim)

# Fine-tune hook — replace decoder with a task head.
model.set_decoder(my_opc_decoder)
loss = model(mask_tensor, target=corrected_mask)
```

## Open questions

- **Tokenisation vs raster**: this RFC commits to rasters. RFC 0002
  argues for layout-as-sequence. We expect to ship both — the MAE on
  rasters, the autoregressive generator on tokens — and use the MAE
  features as a *conditioning signal* for the token model, not as a
  competitor.
- **Per-PDK heads vs PDK-conditioned single model**: open. v0.2 ships
  a single PDK-conditioned model; per-PDK specialisation is a v0.3
  question.
- **Licensing**: weights to be released under CC-BY-4.0, training code
  under the repo licence (MIT). The training corpus inherits its
  source licences and is *not* redistributed — we ship a recipe, not a
  tarball.

## Rollout

- v0.2 alpha: pretraining recipe + first checkpoint, no fine-tune
  examples.
- v0.2 beta: hotspot-detection linear probe + FreePDK45 OPC fine-tune
  example.
- v0.2 GA: HuggingFace Hub release, blog post with reproducibility
  numbers, downstream benchmark line on the leaderboard.

## Alternatives considered

- **DINO / self-distillation** — more expensive to train, no clear
  advantage on binary rasters where the masked-reconstruction signal is
  unambiguous.
- **Pure supervised pretraining on hotspot labels** — labels are too
  small in volume and the transfer to OPC/ILT is weaker than the
  unsupervised route.
- **Frozen ImageNet ViT** — domain gap is large enough that we expect
  worse transfer than even a tiny in-domain MAE; still, we will report
  it as a baseline.
