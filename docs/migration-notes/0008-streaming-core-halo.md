# Migration Note — Streaming Core/Halo (RFC 0008)

**Audience:** callers of `workflow.tiling`, `workflow.full_chip_tiling`,
`workflow.halo.compute_halo_px`, and the CLI optimize path.

## What changed

Nothing existing breaks. RFC 0008 lands an **additive** `streaming`
package. All legacy entry points keep their exact behaviour:

| Legacy API                                   | Status      | Streaming counterpart                        |
| -------------------------------------------- | ----------- | -------------------------------------------- |
| `tile_layout(mask, tile_size, overlap)`      | unchanged   | `plan_tile_requests(shape, core, halo)`      |
| `stitch_tiles(pairs, output_shape)`          | unchanged   | `TensorTileSink` / `MemmapTileSink`          |
| `compute_halo_px(node, model, pixel, tile)`  | unchanged   | `PhysicalInteractionHaloPolicy`              |
| fixed 128 px halo                            | unchanged   | `LegacyFixedHaloPolicy(128)`                 |
| dense full-raster output                     | unchanged   | `MetricOnlyTileSink` (raster-free mode)      |

## Semantic differences to know about

1. **Ownership, not overlap.** `overlap` blurred "read margin" and
   "committed area". Core/halo separates them: the core is the only
   committed region; the halo is context. A legacy
   `tile_size=T, overlap=O` corresponds roughly to
   `core_size=T-O, halo=O` with the same read extent but *no blending*
   (cores partition the output).
2. **No weight map.** Streaming writes each output pixel once, so the
   historical linear-blend weight map does not exist on the new path.
   Edge seams are handled by giving the forward model real halo content,
   not by averaging two half-trusted results.
3. **Halo answers carry provenance.** `HaloRequirement.certified=False`
   means "engineering estimate". Never quote it as a certified bound;
   `CERTIFIED_SUFFICIENT` requires an analytic tail / QDM bound.

## Regression pins

- `tests/test_streaming/test_pipeline.py::TestIdentityRoundTrip` —
  bitwise tile→core/halo→sink round-trip and one-write-per-core.
- `tests/test_streaming/test_pipeline.py::TestFullVsTiled` — core error
  vs full-context reference decreases as halo grows.
- `tests/test_streaming/test_metric_firewall.py` — continuous
  certificate fields never shadow raster `epe_*` metric names.
