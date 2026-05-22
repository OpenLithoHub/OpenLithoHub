# OpenRAM → FreePDK45 path

Verification snapshot for the OpenRAM-bundled FreePDK45 GDS files consumed by
[`openlithohub.data.freepdk45_sram.FreePdk45SramDataset`][openlithohub.data.freepdk45_sram.FreePdk45SramDataset].

**Verified:** 2026-05-22 in a Python 3.12 virtualenv with `openram` 1.2.48.

## Install

```
pip install 'openlithohub[freepdk45-sram]'   # pulls openram>=1.2.48
# or, equivalently:
pip install openram
```

OpenRAM ships on PyPI under BSD-3-Clause. No system dependency is required for
the read path below — `klayout` (already in OpenLithoHub's `[workflow]` extras)
covers the rasterizer side.

## What ships in the wheel

`<site-packages>/openram/`:

```
technology/
  freepdk45/    ← BSD-3 PDK + bitcell GDS bundle (THIS is what we want)
  gf180mcu/
  scn3me_subm/
  scn4m_subm/
  sky130/
sram_compiler.py
rom_compiler.py
common.py
```

`technology/freepdk45/gds_lib/` (10 files, all included in the wheel):

```
cell_1rw.gds         ← 6T 1-port SRAM bitcell
cell_2rw.gds         ← 8T dual-port bitcell
dff.gds              ← D-flip-flop
dummy_cell_1rw.gds   ← row/col edge dummy
dummy_cell_2rw.gds
replica_cell_1rw.gds ← timing-replica column
replica_cell_2rw.gds
sense_amp.gds
tri_gate.gds
write_driver.gds
```

## Spot-check: read `cell_1rw.gds` via klayout

```python
import klayout.db as kdb
ly = kdb.Layout()
ly.read(".../technology/freepdk45/gds_lib/cell_1rw.gds")
# Result:
#   cell: cell_1rw bbox(um): (-0.095, -0.1; 0.8, 1.465)
#   layers: [(1,0), (2,0), (3,0), (4,0), (5,0), (6,0),
#            (9,0), (10,0), (11,0), (12,0), (13,0), (239,0)]
#   dbu_nm: 0.5
```

Bbox ≈ 0.9 µm × 1.6 µm — a believable 45 nm 6T bitcell footprint. `dbu = 0.5
nm` matches FreePDK45's published precision.

## Why we don't run the compiler

`sram_compiler.py` runs from `cwd = openram_pkg_dir` (the script does
`from common import *` at top level, so it must be invoked as a script, not
imported as a library):

```bash
cd <site-packages>/openram
python sram_compiler.py /tmp/tiny_sram_config.py
```

with config:

```python
word_size = 4
num_words = 16
num_banks = 1
tech_name = 'freepdk45'
nominal_corner_only = True
route_supplies = False
check_lvsdrc = False
output_path = '/tmp/openram_smoke_out'
output_name = 'tiny_4x16'
```

Result: progresses through bank / decoder / sense-amp synthesis, placement, then
crashes in `route_layout` → `get_bbox` with:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
  at compiler/base/vector.py:29: self.x = float(x)
```

This is a known **numpy 2.x scalar-conversion regression** in OpenRAM 1.2.48 —
`boundary[0][0]` is returned as a 1-d numpy slice instead of a scalar by recent
numpy. Pinning `numpy<2` sidesteps it; that's an OpenRAM-upstream issue, not
ours.

## What this confirms for `FreePdk45SramDataset`

1. **Wheel-installable, BSD-3.** `pip install openram` is a single-line
   dependency we opt into via the `[freepdk45-sram]` extra, no submodule pin.
2. **GDS lib ships in the wheel.** No external download / git submodule
   required — the adapter locates the files via
   `importlib.resources.files("openram") / "technology" / "freepdk45" / "gds_lib"`.
3. **klayout reads them cleanly.** The shared `rasterize_cell_layer` helper
   from `openlithohub.data.asap7` works on these files without modification.
4. **The numpy crash blocks running the compiler** — but for the
   bitcell-rasterization use case, we **don't need to run the compiler**. The
   pre-shipped GDS files are themselves the canonical artifact. The adapter
   rasterizes `cell_1rw.gds` directly with no compile step.

## Coverage scope

The adapter ships **Option A**: read the bundled GDS files only. This is
zero-compile-step, works in CI without numpy pinning, and covers exactly the
"well-known SRAM cells" surface (1RW / 2RW bitcells, sense amp, write driver,
DFF, replica, dummy).

**Option B** — exposing the full SRAM compiler so users can request
"16×4 SRAM, FreePDK45" — is a future enhancement, blocked on OpenRAM's
numpy 2.x compatibility landing upstream.

## License

OpenRAM is BSD-3-Clause (see `LICENSE` inside the installed wheel). FreePDK45
is BSD-3-Clause as published by NCSU / OKState. OpenLithoHub does not
redistribute the bytes — the adapter consumes them from the user's
pip-installed `openram` wheel at runtime.
