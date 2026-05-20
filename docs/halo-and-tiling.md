# Halo and Tiling

OpenLithoHub processes large layouts by **tiling**: cut the chip into
overlapping tiles, run the forward / OPC model on each, and stitch the
results back together with a ramped blend across the overlap. The
overlap region is called the **halo**.

This page explains what the halo does, how OpenLithoHub picks its size,
and the CLI knobs you have to override it.

## What the halo does

Each tile's interior pixels need to "see" real layout context for the
forward model to compute the same value as it would on the full chip.
Two phenomena set the lower bound:

- **Optical interaction radius (OIR).** The lithography kernel is a
  2-D convolution; pixels outside the kernel's effective radius
  contribute zero. EUV (λ ≈ 13.5 nm) has a small radius; DUV ArF
  (λ ≈ 193 nm) has a much larger one.
- **Model receptive field (RF).** A convolutional ML model also
  propagates information across pixels. A halo smaller than the model's
  receptive field means the boundary tiles see zero-padding inside the
  network's effective kernel.

The right halo is therefore `max(OIR_px, RF_px)`, rounded up to a
stride-friendly multiple, clamped to `tile_size - 1`.

## Auto-sizing (default)

`openlithohub optimize run` defaults to `--halo auto`. The CLI prints
the resolved value and its provenance, e.g.:

```
Halo: 256 px (≈256 nm at 1.0 nm/px) — auto from 3nm-euv (OIR=250 nm) + neural-ilt (RF=64 px)
```

The math:

| Source | Value | Used as |
|--------|-------|---------|
| `ProcessNodeConfig.optical_radius_nm` | per-node table below | `ceil(OIR_nm / pixel_nm)` |
| `LithographyModel.RECEPTIVE_FIELD_PX` | per-model table below | direct px |
| Both `None` | n/a | falls back to `DEFAULT_HALO_PX = 128` |

Built-in node OIR values:

| Node | OIR (nm) |
|------|----------|
| `2nm-euv` | 250 |
| `3nm-euv` | 250 |
| `5nm-euv` | 400 |
| `7nm` | 400 |
| `28nm` | 1500 |
| `45nm` | 1500 |

Built-in model receptive fields:

| Model | RF (px) |
|-------|---------|
| `dummy-identity` | 0 |
| `rule-based-opc` | 16 |
| `levelset-ilt` | 0 |
| `neural-ilt` | 64 |

## Overrides

| CLI flag | When to use |
|----------|-------------|
| `--halo auto` | Default. Pick `max(OIR_px, RF_px)`. |
| `--halo N` | Force a fixed pixel count, computed from your own kernel data. |
| `--overlap N` | **Legacy** — kept for scripts that pre-date RFC 0005. |

`--halo` and `--overlap` are **mutually exclusive** when both are
explicit. Prefer `--halo`.

## When to override

- **You measured your kernel.** If you have empirical fall-off data for
  your node, a tighter halo than the table default may be safe.
- **You're benchmarking halo sensitivity.** Sweep `--halo` to check
  edge-stitch artefacts.
- **You're pinning behaviour for a regression suite.** Use `--halo N`
  (or `--overlap N`) to lock the value across CLI versions.

## Programmatic API

```python
from openlithohub.workflow import compute_halo_px, describe_halo, get_node
from openlithohub.models.registry import registry, register_builtin_models

register_builtin_models()
node = get_node("3nm-euv")
model = registry.get("neural-ilt")
halo = compute_halo_px(node=node, model=model, pixel_nm=1.0, tile_size=2048)
print(describe_halo(halo, node, model, pixel_nm=1.0))
```

## See also

- [RFC 0005 — Process-Node-Aware Tile Halo Sizing](rfcs/0005-process-node-halo-sizing.md)
- [Architecture — Workflow Engine](architecture.md)
- [CLI Reference — `optimize run`](cli-reference.md)
