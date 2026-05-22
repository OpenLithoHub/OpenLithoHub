# Cell rasterization gallery — `openlithohub data show`

The two PDK adapters added for issue #4 — `Asap7Dataset` and
`FreePdk45SramDataset` — are surfaced through the CLI as
`openlithohub data list` / `openlithohub data show`. This page documents
the inspection workflow.

## Why no real-cell gallery is bundled

[`DATA-LICENSES.md`](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/DATA-LICENSES.md)
states the repo does not redistribute PDK bytes. A rasterized PNG of an
ASAP7 standard cell is a derivative of upstream GDS data; while ASAP7's
BSD-3-Clause license would permit it, our self-imposed policy is
stricter. So this page does not embed real-cell PNGs — it documents how
to regenerate them locally against the upstream PDK install.

## Generating the gallery locally

### ASAP7 (requires a local clone, BSD-3-Clause attribution)

```bash
# One-time: clone ASAP7 with the asap7sc7p5t_27 submodule.
python -c 'from openlithohub.data.asap7 import Asap7Dataset; \
    Asap7Dataset.fetch("/path/to/asap7", accept_license=True)'

# Render every canonical cell to ./asap7-cells/<cell>.png
openlithohub data show asap7 \
    --all \
    --data-root /path/to/asap7 \
    --accept-license \
    --out ./asap7-cells/
```

Default cells (`CANONICAL_CELLS` in `openlithohub.data.asap7`):

- `INVx1_ASAP7_75t_R`
- `NAND2x1_ASAP7_75t_R`
- `NOR2x1_ASAP7_75t_R`
- `DFFHQNx1_ASAP7_75t_R`

To render a single cell with shorthand → canonical resolution:

```bash
openlithohub data show asap7 \
    --cell INV \
    --data-root /path/to/asap7 \
    --accept-license \
    --out inv.png
# cell=INVx1_ASAP7_75t_R shape=(8, 64) layer=[10, 0] pixel_nm=1.0 license=BSD-3-Clause -> inv.png
```

The shorthand expands via `resolve_cell_name(...)` — pass
`--drive`, `--flavor`, `--track` to override the defaults
(`x1`, `R`, `75`) for SL / SRAM / 6-track flavors.

### FreePDK45 SRAM bundle (requires `pip install openram`)

`FreePdk45SramDataset` reads the GDS bundle that ships inside the
`openram` pip wheel — no separate clone needed. Once installed, the
adapter auto-locates it via `importlib.resources`.

```bash
pip install 'openlithohub[freepdk45-sram]'   # adds openram

openlithohub data show freepdk45-sram \
    --all \
    --out ./freepdk45-sram-cells/
```

Default cells (`CANONICAL_CELLS` in `openlithohub.data.freepdk45_sram`):

- `cell_1rw` (6T 1RW bitcell)
- `cell_2rw` (8T 2RW bitcell)
- `dff`
- `sense_amp`
- `write_driver`
- `tri_gate`
- `replica_cell_1rw`, `replica_cell_2rw`
- `dummy_cell_1rw`, `dummy_cell_2rw`

## Quick reference

| Flag | Default | Notes |
|---|---|---|
| `--cell NAME` | required (unless `--all`) | ASAP7 accepts shorthand; FreePDK45-SRAM takes the GDS file stem |
| `--all` | off | Render every cell in `CANONICAL_CELLS`. Mutually exclusive with `--cell`. |
| `--out PATH` | `<cell>.png` (single) / `<dataset>-cells/` (`--all`) | Single-cell mode treats it as a file; `--all` mode treats it as a directory |
| `--data-root` | required for ASAP7 only | Path to a local ASAP7 clone |
| `--accept-license` | required for ASAP7 only | Acknowledges BSD-3-Clause attribution |
| `--design-layer L/D` | `10/0` (ASAP7) / `11/0` (FreePDK45-SRAM) | Override the default rasterized layer |
| `--pixel-nm` | `1.0` | Raster pixel size in nm |
| `--drive` / `--flavor` / `--track` | `x1` / `R` / `75` | ASAP7 shorthand-resolver overrides |

## When to use this vs. `openlithohub eval run`

| Use case | Tool |
|---|---|
| Eyeball a single cell to debug an adapter | `data show --cell` |
| Populate a slide deck with cell rasterizations | `data show --all` |
| Run a benchmark and produce a `BenchmarkResult` | `eval run --dataset <X> --pdk <Y>` |
| Inspect what cells an adapter exposes | `data list <X>` |

The `data` subcommand is intentionally narrow — it's a research-flow
helper, not the eval path. For benchmark numbers and leaderboard
submissions use `eval run`.
