# Layout-MAE Architecture Audit

**Status:** ViT-S MAE prototype — RFC 0001 recipe, no pretrained weights yet.
**Last audited:** 2026-05-23 against `docs/rfcs/0001-base-model.md` and the canonical MAE reference (He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022).

OpenLithoHub ships a `LayoutMAE` module (`src/openlithohub/models/layout_mae.py`) implementing a ViT-S masked-autoencoder over rasterised layout patches, intended as a self-supervised pretraining base for downstream OPC / hotspot tasks. This page records what we implement, what RFC 0001 specifies, and where they diverge.

## Audit method

Compared `models/layout_mae.py` against:

- `docs/rfcs/0001-base-model.md` — the project RFC that pins the recipe.
- He et al., *Masked Autoencoders Are Scalable Vision Learners* (CVPR 2022, [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)) — canonical MAE reference. **Not in `docs/references.bib` as of 2026-05-23 — see Findings.**

Confidence:

- **A** — verified against the source in this repo.
- **B** — verified against RFC 0001's stated values.
- **C** — derived from the canonical MAE reference (paper PDF not on disk).

## What RFC 0001 specifies

| Item | RFC 0001 specification | Confidence |
|------|------------------------|------------|
| Encoder | ViT-S — `embed_dim=384`, `depth=12`, `num_heads=6`. | **B** |
| Decoder | Lightweight ViT — `decoder_embed_dim=256`, `decoder_depth=4`, `decoder_num_heads=8`. | **B** |
| Patch | 16×16 patches over 256×256 inputs (16² = 256 patches). | **B** |
| Masking | Random 75% mask ratio. | **B** |
| Loss | L1 reconstruction over masked patches only. | **B** |
| Position embedding | 2D sin-cos (non-learned). | **B** |
| Pretraining target | 200k steps on A100 — v0.2 deliverable, not in this prototype. | **B** |

## What OpenLithoHub implements

`src/openlithohub/models/layout_mae.py` :: `LayoutMAE`:

| Item | OpenLithoHub | Matches RFC / paper? | Confidence |
|------|---------------|----------------------|------------|
| Encoder | ViT-S — `embed_dim=384`, `depth=12`, `num_heads=6` (`LayoutMAEConfig` defaults, ll. ~36–38). | **Yes.** | **A** |
| Decoder | `decoder_embed_dim=256`, `decoder_depth=4`, `decoder_num_heads=8` (ll. ~39–41). | **Yes.** | **A** |
| Patch | `patch_size=16`, `image_size=256`, `in_channels=1` (single-channel rasterised layout). | **Yes** for spatial dims; canonical MAE uses 3-channel ImageNet. Single-channel is the project's domain choice. | **A** |
| Patch embedding | `nn.Conv2d` with kernel=stride=patch_size (ll. ~95–97). | **Yes** — standard ViT patchify-via-conv. | **A** |
| Position embedding | 2D sin-cos via `_sincos_pos_embed`, non-learned (`requires_grad=False`, ll. ~98–100). | **Yes** — matches MAE paper §A.1. | **A** / **C** |
| Random masking | Per-batch sample-the-noise, argsort-shuffle, keep first `n*(1-mask_ratio)` indices (ll. ~141–154). `mask_ratio=0.75` default. | **Yes** — the canonical MAE shuffle algorithm. | **A** / **C** |
| Encoder block | `LayerNorm → MultiheadAttention → residual → LayerNorm → MLP(GELU) → residual`, pre-norm (`_Block`, ll. ~70–84). | **Yes** — standard pre-norm Transformer. | **A** |
| Decoder | Linear projection from encoder dim to `decoder_embed_dim`, mask-token expansion + restore-via-gather, decoder-side pos-embed addition, `decoder_depth` blocks, final linear → flat patch values (ll. ~178–189). | **Yes** — matches MAE paper §3.4. | **A** / **C** |
| Mask token | Single learned embedding of shape `(1, 1, decoder_embed_dim)`, init `N(0, 0.02)` (ll. ~107–108). | **Yes** — matches MAE paper. | **A** / **C** |
| Reconstruction loss | L1 over masked patches only — `(pred − target).abs().mean(dim=-1)` weighted by `mask`, summed and normalised (ll. ~192–199). | **Yes** in form. **The MAE paper uses MSE (per-patch L2), not L1.** RFC 0001 explicitly chooses L1; this is a documented divergence. | **A** |
| Patch normalization | **Not implemented.** MAE paper §3.4 normalizes target patches by per-patch mean/std before computing loss. | **No** — RFC 0001 does not mention this. Possible quality gap on natural-image-like layouts; less likely to matter on binary rasterised layouts (mean/std are nearly constant per patch). | **A** |
| `train_step` | Single step: forward → loss → backward → optimizer.step. Returns scalar loss. | **Yes** — minimal training-loop primitive. | **A** |
| `set_decoder` / fine-tune adapter API | **Not implemented.** RFC 0001 marks this as a v0.2 follow-up. | **N/A** — by design for the v0.1 prototype. | **A** |
| Pretrained weights | **None.** The 200k-step A100 pretrain is a v0.2 deliverable. | **N/A** — by design. | **A** |
| HF Hub `from_pretrained` | **Not wired** for this model. | **N/A** — by design until weights exist. | **A** |

## Findings

1. **Architecture matches the canonical MAE recipe.** ViT-S encoder, lightweight ViT decoder, 75% mask ratio, per-batch shuffle masking, sin-cos pos-embeds, mask-token-and-gather decoder input — all in place.

2. **L1 reconstruction loss diverges from the MAE paper.** The paper uses MSE; we use L1 per RFC 0001 §Architecture. **Implication:** absolute pretrain loss numbers are not comparable to the published MAE / SimMIM values. L1 is more robust to the binary-edge structure of rasterised layouts (where MSE would be dominated by edge pixels), so the divergence is well-motivated.

3. **No per-patch target normalization.** The MAE paper normalizes each target patch by its own mean/std before L1/L2. We do not. **For binary rasterised layouts this is unlikely to matter** — patches are mostly in `{0, 1}`, so per-patch normalization collapses to a near-identity. Worth revisiting if the input ever becomes anti-aliased greyscale.

4. **No pretrained weights.** The v0.1 prototype is the **recipe**, not a pretrained model. `LayoutMAE()` constructed today returns randomly-initialised weights — useful only for the `train_step` smoke test or as a starting point for project-internal pretraining. The v0.2 deliverable in RFC 0001 calls for 200k pretraining steps on an A100; that work is not part of this audit's scope.

5. **No fine-tune adapter API.** RFC 0001 explicitly defers this to v0.2. The `encode()` method is the documented frozen-feature path for any consumer that wants to pretrain elsewhere and consume features here.

6. **`Yang2022_MAE` (or equivalent) is not in `docs/references.bib`.** Confidence on paper-side claims is **C**. Adding the He et al. 2022 entry would let us promote the items marked **C** to **A** without the audit needing the paper PDF on disk.

## Implications for users

- **Treat this as a recipe, not a model.** Random-init `LayoutMAE()` will not give meaningful features. The work to pretrain on a layout corpus is downstream of this audit.
- **Don't compare reconstruction loss values to published MAE numbers.** L1 vs. MSE + no patch-normalization → different absolute scale.
- **Citation hygiene:** if you publish using this module, cite the canonical MAE paper (He et al., CVPR 2022) for architecture lineage, and `docs/rfcs/0001-base-model.md` for the project-level recipe choices (L1, single-channel, 256×256, 16×16 patches). Once the He et al. entry lands in `references.bib`, use the canonical key (suggested `He2022_MAE`).

## Re-audit triggers

Re-run this audit when any of the following change:

- `LayoutMAEConfig` defaults change (encoder/decoder dims, depth, mask_ratio).
- `reconstruction_loss` switches from L1 to MSE, or gains target patch-normalization.
- `set_decoder` / fine-tune adapter API lands.
- A `from_pretrained` path lands and v0.2 pretrained weights are published.
- The He et al. MAE paper PDF lands in `docs/papers/` (or its bib entry lands in `docs/references.bib`) — promote items marked **C** to **A**.
