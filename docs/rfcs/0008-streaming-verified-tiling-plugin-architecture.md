# RFC 0008 — Streaming Core/Halo Tiling and Verification Plugin Architecture

- Status: Accepted (foundation implemented)
- Related: RFC 0004 (multi-GPU tile pipeline), RFC 0005 (process-node halo
  sizing), RFC 0007 (QDM continuous process-window verification)

## 1. Why a full dense raster does not scale

Every current full-chip path materialises the whole layout as a dense host
tensor before tiling, and stitching rebuilds a dense output plus a weight
map. Peak memory is therefore `O(full-chip raster area)` on **three**
tensors (input, output, weights). Layout growth is capped by host RAM long
before tile count becomes the limiting factor. For future QDM continuous
verification, where only certificates and worst-case bounds need to be
kept, this is structurally wrong.

Target invariant: **layout growth increases tile count, not per-run memory.**
Peak memory must be `O(tile area + active batch)`.

## 2. Core vs halo ownership

`streaming/geometry.BoundingBox` + `HaloSpec` + `TileRequest` split each
tile read region into:

- **Trusted core** — the only region whose result may be committed;
- **Context halo** — real neighbourhood content for the forward model.

Planning (`plan_tile_requests`) produces an *exact* core cover of the
global domain: no gap, no double-owned pixel. Read regions may overlap.
Because cores partition the output, stitching needs **no weight map and
no blend pass** — each output pixel is decided exactly once.

`tile_layout(..., overlap=...)` is retained unchanged as the
backward-compatible API (see `docs/migration-notes/0008-streaming-core-halo.md`).

## 3. Streaming source / sink

- `TileSource` — `read_window(bbox)` yields only the requested window.
  Shipped: `TensorTileSource` (compat), `MemmapTensorTileSource`
  (out-of-core raster via `np.load(mmap_mode=...)` / raw `np.memmap`),
  `VectorLayoutTileSource` + `SpatialLayoutIndex` (rasterize only objects
  intersecting the window). Remote/database sources need only these three
  members.
- `TileSink` — `write_core(tile_id, bbox, tensor, metadata)` +
  `finalize()`. Shipped: `TensorTileSink` (old behaviour),
  `MemmapTileSink` (on-disk output), `MetricOnlyTileSink`
  (no raster at all — certificates/aggregates only; the QDM-critical mode).

## 4. Halo policy

`HaloPolicy.required_halo(HaloContext) -> HaloRequirement` replaces hard
coded sizing. `HaloRequirement` carries `halo_px`, `provenance`,
`error_bound`, `certified`, `reason`, and a `HaloStatus` of:

```text
FIXED | PHYSICS_ESTIMATED | EMPIRICALLY_STABLE | CERTIFIED_SUFFICIENT | INCONCLUSIVE
```

Shipped policies: `LegacyFixedHaloPolicy` (RFC-0004-era fixed halo),
`PhysicalInteractionHaloPolicy` (RFC-0005 `max(OIR, RF)` logic migrated),
`KernelTailHaloPolicy` (smallest `h` with kernel tail mass
`T(h) <= epsilon`, certified only when the kernel sample is trusted).
`combine_requirements` max-composes plugin requests; any non-certified or
INCONCLUSIVE participant demotes the combination.
`estimate_minimum_halo` runs the adaptive candidate sweep and returns
`EMPIRICALLY_STABLE` — explicitly *not* a certificate.

## 5. Certified vs empirical halo

`CERTIFIED_SUFFICIENT` requires a rigorous bound (analytic kernel tail,
future QDM bound). Empirical stability — core discrepancy under tolerance
across enlargements — is reported as `EMPIRICALLY_STABLE` with
`certified=False`. The firewall is enforced downstream: a reducer may not
treat an empirical halo as a theorem-facing error bound of zero.

## 6. Plugin API

`streaming.verification.VerificationPlugin` (Protocol):
`required_halo / prepare / verify_tile / reduce / finalize` plus optional
`refine`. `VerificationRegistry` (`register/get/list`) ships with a
`dummy` verifier; entry-point discovery is deferred to a later phase.
`StreamingVerificationReducer` folds `TileVerificationResult`s one at a
time (latest verdict per tile wins after refinement) and composes the
error budget:

```text
E_total <= E_process + E_spatial + E_halo + E_raster_bridge + E_other
```

Global PASS is impossible with any FAIL, any INCONCLUSIVE tile,
`PARTIAL_COVER` coverage, or an error budget above tolerance.

## 7. QDM future integration

QDM plugs in as a `VerificationPlugin` + optional
`SourceNativeVerificationBackend` (`verify/source_native.py`) over a
frozen `SourceSnapshot` (`verify/source_snapshot.py`). No QDM mathematics
is implemented here; nothing QDM-specific leaks into the scheduler. The
B04 continuous certificate schema (`verify/types.py`) is independent of
the raster benchmark schema — `epe_max_nm` semantics are untouched.

## 8. Backward compatibility

- `tile_layout`, `stitch_tiles`, `schwarz_tiled_ilt` unchanged;
- CLI defaults unchanged;
- raster EPE/PV-band/leaderboard semantics unchanged (firewall test);
- old small-tensor workflows keep working via `TensorTileSource` +
  `TensorTileSink`, which reproduce the legacy behaviour exactly.

## 9. Memory complexity

Per-tile peak: `O((C + 2h)^2)` for read + forward + core result, times the
active batch. The scheduler holds only request metadata. `MetricOnlyTileSink`
uses O(1) aggregate state. Whole-chip memory appears only if the caller
chooses `TensorTileSource`/`TensorTileSink`.

## 10. Migration plan

1. Phase 1 (this RFC): addititive `streaming/` package + policies +
   plugin API + benchmarks; no existing API changes.
2. Phase 2: CLI `optimize` gains `--streaming` using the new pipeline with
   `LegacyFixedHaloPolicy` as default, `--halo-policy` selector.
3. Phase 3: klayout-Region-backed `SpatialLayoutIndex`; entry-point
   verifier discovery; verifier-driven subdivide scheduling.
4. Phase 4: QDM plugin + source-native interval backend on real models.
