# RFC 0002 — Layout Tokens

| | |
|-|-|
| Status | Draft |
| Author | OpenLithoHub maintainers |
| Created | 2026-05-19 |
| Targets | v0.2 |
| Related | RFC 0001 (Layout-MAE Base Model), `openlithohub.synth.DiffusionLayoutGenerator` |

## Summary

Define a polygon-level tokenisation of layouts so a transformer can
generate layouts as *sequences of tokens* instead of rasters. This
unlocks:

1. Autoregressive layout generation that respects PDK rules at the
   tokeniser level — no post-hoc DRC repair pass.
2. Conditioning on tokenised PDK rules (so a single model can serve
   multiple PDKs).
3. Length-prefixed compression: 256×256 raster → ~1k tokens for typical
   patches.

This RFC defines *only* the tokeniser format and the public interface.
The transformer that consumes these tokens is out of scope; the
intended downstream consumer is `DiffusionLayoutGenerator` (which we
expect to replace with an autoregressive token model in v0.2).

## Why tokens

- **PDK awareness is built in**: every coordinate is snapped to the
  manufacturing grid and every emitted polygon respects min-width and
  min-spacing by construction (a token sequence that violates these is
  unrepresentable, not just discouraged).
- **Lossless**: round-trip layout → tokens → layout matches at the
  pixel level, unlike rasterisation at finite pixel pitch.
- **Variable-resolution**: the same token stream can render at any
  pixel size; useful when downstream tools expect different grid
  resolutions.
- **Cheaper attention**: typical 256×256 layouts have <2k polygon
  vertices but ~65k pixels.

## Non-goals

- Replacing rasters everywhere. RFC 0001's MAE backbone stays raster-
  based; tokens are an *additional* representation, not a replacement.
- Hierarchical / cell-level tokenisation. v0.2 is flat polygon tokens.
- Net-level / connectivity tokens. We tokenise *geometry only*; netlist
  awareness is a v0.3 problem.

## Token vocabulary

Five token classes:

| Class | Encoding | Notes |
|-------|----------|-------|
| `<bos>` | reserved id 0 | Start of layout. |
| `<eos>` | reserved id 1 | End of layout. |
| `<polygon>` | reserved id 2 | Marks the start of a polygon. |
| `<vertex>` | (x, y) tuple, quantised to PDK grid | 2 ints per vertex. |
| `<close>` | reserved id 3 | Closes the current polygon. |

Coordinate quantisation: each (x, y) is snapped to a multiple of the
PDK manufacturing grid (typically 1nm or 0.5nm). The vocabulary size
is then ~2 × (canvas_size / grid). For a 256nm canvas at 1nm grid:
~512 distinct coordinate ids.

For full PDK-rule awareness we additionally support a **rule-prefix**
sequence — a small, fixed-format header that encodes (pdk_name,
pixel_size_nm, min_width_nm, min_spacing_nm). This lets a single model
condition on the target PDK at inference time.

## Sequence format

```
<bos> <pdk-rules> <polygon> (x, y)+ <close> [<polygon> (x, y)+ <close>]* <eos>
```

Polygons are emitted in scan order (top-to-bottom, left-to-right of
their bounding-box minimum), and vertices within each polygon are
emitted clockwise starting from the top-left vertex of the polygon's
bounding box. This canonicalisation is required — without it the same
layout has many valid token sequences and the model wastes capacity
modelling that ambiguity.

## Public API

```python
from openlithohub.tokens import LayoutTokenizer

tokenizer = LayoutTokenizer.from_pdk("freepdk45")

# Mask tensor → token ids.
ids = tokenizer.encode(mask_tensor)  # (T,) int64

# Token ids → mask tensor.
mask = tokenizer.decode(ids, size=256)

# Round-trip should be exact.
assert torch.equal(mask, tokenizer.decode(tokenizer.encode(mask), size=mask.shape[-1]))
```

The tokenizer is implemented as a pure Python class on top of
`shapely` — no Torch ops needed for tokenisation itself.

## Encoder pipeline

1. **Vectorise**: extract polygon outlines from the binary mask via
   marching-squares + Douglas-Peucker simplification.
2. **Snap**: round each vertex to the PDK grid; drop polygons whose
   simplification yields fewer than 3 vertices or whose area is below
   `min_area_nm2`.
3. **Canonicalise**: order polygons + vertices as defined above.
4. **Emit**: write `<bos> <pdk-rules> <polygon> ... <eos>`.

## Decoder pipeline

1. **Parse**: split sequence at `<polygon>` / `<close>` boundaries.
2. **Reject**: drop sequences where any polygon has self-intersection
   or area below `min_area_nm2`. (Reject silently is fine — sampling
   re-tries.)
3. **Rasterise**: use `shapely` polygon → raster at the requested
   pixel size.

## Prior art

- **PolyGen** (Nash et al., 2020): autoregressive polygon meshes for
  3D geometry. We borrow the (vertex, face) factoring and the
  canonical-ordering trick.
- **VectorMask** internal/proprietary work at major fabs is known to
  use polygon tokenisation for ML-OPC; details are not public but the
  shape of the problem is the same.
- **DeepSDF / CodeOPC** academic work uses signed-distance fields
  rather than polygons — orthogonal approach, not pursued here.

## Evaluation

Tokeniser-only evaluation (no model in the loop):

1. **Round-trip exactness**: encode-then-decode on synthetic + public
   layouts gives identical pixel masks. Target: 100% on synthetic,
   ≥99% on public (the ≤1% loss is from sub-grid features that don't
   survive snapping — those are MRC violations anyway).
2. **Compression ratio**: tokens-per-pixel for representative patches.
   Target: <5% (256×256 = 65k pixels → <3.3k tokens).
3. **Vocabulary size**: <4k tokens including all coordinate ids for
   the 256nm canvas. (Larger canvases use windowed encoding.)

## Open questions

- **Variable-coordinate vocabulary**: emitting (x, y) as two ints
  doubles sequence length; emitting them as a single packed id grows
  the vocab quadratically. We pick two-int emission for v0.2 and
  revisit if attention cost dominates.
- **Multi-layer layouts**: Manhattan layouts often have multiple metal
  layers; v0.2 tokenises one layer at a time. Layer-stack
  tokenisation is a v0.3 question.
- **Holes**: polygons with holes (e.g., ring shapes) are represented
  as outer + inner contours separated by a `<hole>` token. Spec'd in
  v0.2; not the common case.

## Rollout

- v0.2 alpha: tokeniser implementation, round-trip tests, no model.
- v0.2 beta: small autoregressive transformer trained on synthetic
  layouts, replacing the diffusion stub.
- v0.2 GA: documented public API + a HuggingFace dataset of pre-
  tokenised public layouts.

## Alternatives considered

- **Pixel-level autoregression** (PixelCNN-style on rasters): does not
  scale beyond ~64×64 patches and offers no PDK-rule guarantees.
- **Bezier-curve tokens**: nice for analog cells but 95% of digital
  layout is Manhattan polygons, so the added vocab/complexity is not
  worth it for v0.2.
- **DSL-based sequences** (gates, routes): captures intent better but
  is fab-specific. We tokenise *post-place-and-route* geometry, not
  intent.
