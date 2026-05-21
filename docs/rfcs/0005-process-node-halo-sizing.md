# RFC 0005 — Process-Node-Aware Tile Halo Sizing

| | |
|-|-|
| Status | Implemented in v0.3 |
| Author | OpenLithoHub maintainers |
| Created | 2026-05-20 |
| Targets | v0.3 |
| Related | `openlithohub.workflow.tiling`, `openlithohub.workflow.halo`, `openlithohub.workflow.process_node`, `openlithohub.models.base`, RFC 0004 |

## Summary

`openlithohub optimize run` tiles a layout, runs a forward lithography model
on each tile in isolation, and stitches the results back together with a
**halo** (overlap region) that lets each tile see real neighbouring layout
instead of zero-padded artefacts at the boundaries.

Until v0.3, the halo size was a **single hard-coded constant** (`overlap=128`
pixels) regardless of the process node, the imaging wavelength, or the
model's receptive field. This is physically wrong:

- **EUV (λ ≈ 13.5 nm)** has a tiny optical interaction radius — a 128 px
  halo at 1 nm/px is wildly oversized, wasting compute on guard band that
  contributes nothing.
- **DUV ArF (λ ≈ 193 nm)** at older nodes has a large kernel — a 128 px
  halo at 1 nm/px is **too small**, and tile-edge resist contours diverge
  from the full-chip reference.
- A **deep neural model** (e.g. a 4-level UNet) propagates information
  through ~64 px of receptive field; a halo smaller than this means
  boundary tiles see padded zeros inside the model's effective kernel.

This RFC defines how OpenLithoHub picks the halo size automatically from
the process node and model, while preserving back-compat for scripts that
pass a fixed integer.

## Background — what halo actually buys you

The Hopkins / SOCS forward model is a 2-D convolution: each output pixel
is the partial-coherence sum of source weights times kernel(neighbours).
Outside a radius of roughly `~10 × λ / (2 × NA)` the kernel weight is
numerically zero. That radius — call it the **optical interaction radius
(OIR)** — is the smallest halo that makes a tile's interior pixels
indistinguishable from the full-chip reference.

For convolutional ML models, the analogous quantity is the **receptive
field** — how many input pixels can influence one output pixel through
the stack of convolutions. The halo must be at least the receptive field
or the boundary of the tile sees zero-padding inside the network's
effective kernel.

The correct halo is therefore:

```
halo_px = max(ceil(OIR_nm / pixel_nm), receptive_field_px)
```

rounded up to a stride-friendly multiple (8) and clamped to `tile_size - 1`
(`tile_layout` rejects halos ≥ tile size).

## Current state (factual, verified 2026-05-20)

- **Tile geometry** in `workflow/tiling.py:11–181` already implements
  halo + ramp-blended stitching. The math is right; only the **default
  size** is wrong.
- **`ProcessNodeConfig`** in `workflow/process_node.py` carries
  `wavelength_nm`, `numerical_aperture`, `pixel_size_nm`, etc. — but no
  `optical_radius_nm`.
- **`LithographyModel`** in `models/base.py` carries `NAME`,
  `SUPPORTS_CURVILINEAR`, `setup/predict/teardown` — but no
  `RECEPTIVE_FIELD_PX`.
- **`optimize_cmd.py`** had a single `--overlap` flag defaulting to
  `128`. No node- or model-awareness.

## Design

### 1 · Carry OIR on the process node

`ProcessNodeConfig` gains:

```python
optical_radius_nm: float = 1500.0
```

Per-node values (chosen from `~10 × λ / (2 × NA)` rounded to the nearest
typical industrial halo):

| Node | λ (nm) | NA | OIR (nm) |
|------|--------|-----|----------|
| 2nm-euv | 13.5 | 0.55 | 250 |
| 3nm-euv | 13.5 | 0.33 | 250 |
| 5nm-euv | 13.5 | 0.33 | 400 |
| 7nm | 193 | 1.35 | 400 |
| 28nm | 193 | 1.35 | 1500 |
| 45nm | 193 | 1.35 | 1500 |

These are **defaults** — tape-out teams with their own kernel
characterization data should set the field explicitly when constructing
a custom node.

### 2 · Carry receptive field on the model

`LithographyModel` gains a class-level hint:

```python
class LithographyModel:
    RECEPTIVE_FIELD_PX: ClassVar[int] = 0

    @property
    def receptive_field_px(self) -> int:
        return type(self).RECEPTIVE_FIELD_PX
```

`0` means "the model contributes no receptive-field constraint" — it's
either iterative (`levelset-ilt`), pixel-local (`dummy-identity`), or
purely rule-based with a small structuring element (`rule-based-opc`,
RF=16 to cover the largest morph kernel).

In-tree models:

| Model | RF (px) | Reason |
|-------|---------|--------|
| `dummy-identity` | 0 | Pure identity |
| `dummy-failing` | 0 | Test fixture |
| `rule-based-opc` | 16 | SE radius for jog smoothing |
| `levelset-ilt` | 0 | Iterative, full-tile gradient |
| `openilt` | 0 | Iterative, full-tile gradient (SimpleILT) |
| `neural-ilt` | 64 | 4-level UNet, 3×3 convs, 3 maxpools |

### 3 · Centralise the math in `workflow/halo.py`

```python
def compute_halo_px(
    node: ProcessNodeConfig | None,
    model: LithographyModel | None,
    pixel_nm: float,
    tile_size: int,
) -> int:
    if pixel_nm <= 0: raise ValueError(...)
    if tile_size <= 1: raise ValueError(...)
    if node is None and model is None:
        return min(DEFAULT_HALO_PX, tile_size - 1)  # 128, pre-RFC default
    oir_px = ceil(node.optical_radius_nm / pixel_nm) if node else 0
    rf_px = model.receptive_field_px if model else 0
    raw = max(oir_px, rf_px)
    return max(0, min(_round_up(raw, 8), tile_size - 1))


def describe_halo(halo_px, node, model, pixel_nm) -> str:
    """One-line provenance string for CLI logging."""
```

`compute_halo_px` is pure (no I/O, no state) — trivially testable.

### 4 · CLI surface

`optimize_cmd.py` exposes:

- `--halo auto` (default) — compute via `compute_halo_px`.
- `--halo N` — explicit pixel count.
- `--overlap N` — **legacy**, kept for back-compat with pre-RFC scripts.
- `--halo` and `--overlap` are mutually exclusive when both are
  explicit. Detection: `halo != "auto" and overlap is not None`.

The resolved halo and its provenance are logged:

```
Halo: 256 px (≈256 nm at 1.0 nm/px) — auto from 3nm-euv (OIR=250 nm) + dummy-identity (RF=0 px)
```

so users can audit what was chosen and why.

## Hard constraints

1. **No silent behaviour change for pre-RFC scripts.** A pipeline that
   passes `--overlap 128` keeps producing bit-identical output.
2. **Default for new users picks a sensible value.** `--halo auto` with
   `--node 3nm-euv` gives 256 px — physically motivated, not 128 px by
   coincidence.
3. **Pure function.** `compute_halo_px` does no logging, no state, no
   global lookups. The CLI is the *only* place that prints provenance.
4. **No new runtime dependency.** Stdlib `math.ceil` only.
5. **Halo math fits inside `tile_size - 1`.** `tile_layout` rejects
   `overlap >= tile_size`; `compute_halo_px` clamps to honour that.

## Verification

- `tests/test_workflow/test_halo.py` covers:
  - Pure-math: default fallback, OIR-dominates, RF-dominates,
    `max(OIR, RF)`, pixel_nm scaling, tile_size clamping, error paths.
  - Provenance: `describe_halo` mentions node, OIR, halo px.
  - Coverage: every node has positive `optical_radius_nm`; EUV < DUV.
  - Coverage: every registered model exposes a non-negative
    `RECEPTIVE_FIELD_PX`.
  - CLI: auto default, explicit int, legacy `--overlap`, conflict,
    invalid string, negative integer.
- Existing `test_workflow/test_workflow.py` and
  `test_cli/test_commands.py` regress nothing; `--overlap 128` still
  produces today's output bit-for-bit.

## Out of scope

- **Tile-aware compute saving.** Once OIR is small (EUV at sub-nm/px),
  the halo can be tiny and the *useful* tile area is a much larger
  fraction. We do not yet exploit that to bump default tile sizes.
- **Per-tile adaptive halo.** All tiles in a run share one halo — the
  geometry plumbing in `tile_layout` would need invasive changes for
  per-tile values.
- **Auto-derivation from `wavelength_nm` + `NA`.** We carry
  `optical_radius_nm` directly so node authors can override the formula
  with measured kernel data; we do not auto-compute it from NA/λ.
- **Receptive-field auto-discovery.** Models declare RF as a class
  attribute. We do not introspect the `nn.Module` graph to compute it.

## Implementation

- `src/openlithohub/workflow/halo.py` (new): `compute_halo_px`,
  `describe_halo`, `DEFAULT_HALO_PX`.
- `src/openlithohub/workflow/process_node.py`: `optical_radius_nm` field
  on `ProcessNodeConfig`, populated for all built-in nodes.
- `src/openlithohub/models/base.py`: `RECEPTIVE_FIELD_PX` class attribute
  + `receptive_field_px` property.
- `src/openlithohub/cli/optimize_cmd.py`: `--halo` option,
  `_resolve_halo()` helper, mutual-exclusion check.
- `tests/test_workflow/test_halo.py`: 17 tests covering math + CLI.
- Public exports in `workflow/__init__.py`: `compute_halo_px`,
  `describe_halo`, `DEFAULT_HALO_PX`.
