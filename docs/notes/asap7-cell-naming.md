# ASAP7 cell-name format

Verification snapshot for the ASAP7 7.5-track standard-cell library, used as the
reference for [`openlithohub.data.asap7.resolve_cell_name`][openlithohub.data.asap7.resolve_cell_name].

**Source:** [`asap7sc7p5t_28/LEF/asap7sc7p5t_28_R_1x_220121a.lef`](https://github.com/The-OpenROAD-Project/asap7sc7p5t_28/blob/main/LEF/asap7sc7p5t_28_R_1x_220121a.lef)
**Verified:** 2026-05-22 via direct download (212 MACRO definitions, 373 KB).

## Canonical pattern (7.5-track library)

```
<FUNC><DRIVE>_ASAP7_75t_<FLAVOR>
```

- `FUNC` — functional name in caps: `INV`, `NAND2`, `NOR2`, `AOI21`, `AO22`,
  `OAI332`, `XOR2`, `DFFHQNx`, `SDFHx`, `BUFx`, `MAJx`, `CKINVDC`, `ICG`, ...
- `DRIVE` — drive strength suffix: `x1`, `x2`, `x4`, `x10`, ..., or fractional
  `xp33`, `xp5`, `xp67`, `x1p5`, `x6p67`, `x9p33`. Fractional drives use `p` for
  the decimal separator.
- `FLAVOR` — Vt / corner variant: `R` (regular), `L` (low-Vt — faster, leakier),
  `SL` (super-low-Vt), `SRAM` (SRAM-bitcell-adjacent variant).

## Examples (verified, not guessed)

| Function   | Cell name                       |
|------------|---------------------------------|
| Inverter   | `INVx1_ASAP7_75t_R`             |
| 2-NAND     | `NAND2x1_ASAP7_75t_R`           |
| 2-NOR      | `NOR2x1_ASAP7_75t_R`            |
| AOI21      | `AOI22xp5_ASAP7_75t_R`          |
| Buffer     | `BUFx2_ASAP7_75t_R`             |
| DFF (Q,~Q) | `DFFHQNx1_ASAP7_75t_R`          |
| DFF (Q)    | `DFFHQx4_ASAP7_75t_R`           |
| Scan-DFF   | `SDFHx2_ASAP7_75t_R`            |
| Clock-INV  | `CKINVDCx20_ASAP7_75t_R`        |
| ICG        | `ICGx6p67DC_ASAP7_75t_R`        |
| Filler     | `FILLERxp5_ASAP7_75t_R`         |

## Five LEF files in `LEF/` of the 7.5T submodule

| File                                       | Purpose                                  |
|--------------------------------------------|------------------------------------------|
| `asap7sc7p5t_28_R_1x_220121a.lef`          | Regular-Vt cells (212 MACROs)            |
| `asap7sc7p5t_28_L_1x_220121a.lef`          | Low-Vt cells                             |
| `asap7sc7p5t_28_SL_1x_220121a.lef`         | Super-low-Vt cells                       |
| `asap7sc7p5t_28_SRAM_1x_220121a.lef`       | SRAM-bitcell-adjacent cells              |
| `IO_cell/`                                 | I/O ring / pad cells                     |
| `scaled/`                                  | 4× shrunk variants (the file-size-tagged 1x is the un-shrunk reference) |

The `28` in `asap7sc7p5t_28_*` refers to the 2.8 nm-pitch fin grid (28 fin
half-pitches). The 6T sibling library uses the analogous pattern:
`asap7sc6t_26_*` (26 = 2.6 nm-pitch, 6-track height).

## How `Asap7Dataset` consumes this

When the user passes a plain function name (`"INV"`, `"NAND2"`, `"DFFHQ"`), the
dataset adapter resolves it transparently:

1. Default `flavor="R"`, `drive="x1"`, `track="75"` (smallest regular-Vt drive,
   7.5-track library).
2. Compose `f"{func}{drive}_ASAP7_{track}t_{flavor}"`.
3. Look up in the GDS by that exact string.

Set `Asap7Dataset(..., resolve_shorthand=False)` to opt out and require
exact-match canonical names. The 6T variant uses `_ASAP7_65t_` instead of
`_ASAP7_75t_`; pass `track="6"` to `resolve_cell_name` to target it.

## Out-of-scope here

Generating GDS cutouts of individual standard cells (vs. routed blocks) would
require either (a) the cell-library GDS submodule download flow, or (b) an
OpenROAD-driven flow that builds a tiny test design instantiating just those
cells. The current adapter loads the cells directly from the
`asap7sc7p5t_27/GDS/` submodule. Cutting the cells from a routed design is a
future enhancement.
