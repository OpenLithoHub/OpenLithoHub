# ORFS → OpenLithoHub Integration

This guide walks through the end-to-end pipeline from an OpenROAD-flow-scripts (ORFS) routed design to a lithographic manufacturability report.

## Prerequisites

- ORFS cloned and built: `git clone https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts`
- OpenLithoHub installed: `pip install -e .`
- KLayout installed (for GDS rasterization): `pip install klayout`

## Quick Start

### 1. Produce a routed GDS with ORFS

```bash
cd OpenROAD-flow-scripts
make DESIGN_CONFIG=designs/asap7/mock-alu/config.mk
# Output: results/asap7/mock-alu/base/6_final.gds
```

### 2. Run the flow pipeline

```bash
openlithohub flow run results/asap7/mock-alu/base/6_final.gds \
    --pdk orfs_asap7 \
    --layer metal1 \
    --tile-nm 2000 \
    --node 45nm \
    --output report.json
```

### 3. Using a custom PDK layermap

Create a JSON file with your PDK's layer numbers:

```json
{
    "metal1": [20, 0],
    "metal2": [30, 0],
    "via1": [21, 0]
}
```

```bash
openlithohub flow run design.gds --pdk my_custom_layermap.json --layer metal1
```

## Supported PDKs

| PDK | Key | metal1 | metal2 | via1 |
|-----|-----|--------|--------|------|
| ASAP7 (cell lib) | `asap7` | (10, 0) | (11, 0) | (12, 0) |
| FreePDK45 | `freepdk45` | (11, 0) | (13, 0) | (12, 0) |
| ORFS-routed ASAP7 | `orfs_asap7` | (20, 0) | (30, 0) | (21, 0) |
| SkyWater 130nm | `sky130` | (67, 20) | (69, 20) | (68, 44) |

## CLI Reference

```
openlithohub flow run [OPTIONS] INPUT_PATH

Arguments:
  INPUT_PATH  GDS/OAS/DEF file or ORFS results directory

Options:
  --pdk TEXT              PDK layer mapping name or path to custom JSON
  --layer TEXT            Layer name from the PDK layermap [default: metal1]
  --pixel-nm FLOAT        Pixel size in nm [default: 1.0]
  --tile-nm FLOAT         Tile edge length in nm [default: 2000.0]
  --node TEXT             Process node for litho params [default: 45nm]
  --resist-diffusion-nm   Acid diffusion length in nm [default: 0.0]
  --quencher FLOAT        Quencher concentration [default: 0.0]
  --drc / --no-drc        Run DRC compliance [default: on]
  --mrc / --no-mrc        Run MRC compliance [default: on]
  --output PATH           Save JSON report to file
  --deterministic         Force bit-reproducible backends
```

## Architecture

The flow command chains these existing OpenLithoHub components:

1. **Layer resolution** — `_layers.py` → JSON layermap → `(layer, datatype)` tuple
2. **GDS rasterization** — `data/io.py` via KLayout
3. **Tiling** — `data/orfs.py` `tile_design_tensor()` at 2µm/5µm windows
4. **Hopkins forward** — `simulators/hopkins_sim.py` with SOCS kernels
5. **Resist modeling** — optional acid diffusion via `--resist-diffusion-nm`
6. **Metrics** — EPE, PV Band, DRC, MRC per tile
7. **Aggregation** — mean across tiles → JSON report

## License Compliance

The flow CLI only reads user-provided files. It never downloads, bundles, or
redistributes PDK data. Users must obtain ORFS and PDK artifacts independently
and accept the relevant licenses (ASAP7 BSD-3-Clause, sky130 Apache-2.0, etc.).
