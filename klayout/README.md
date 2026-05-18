# OpenLithoHub KLayout Macro

A reference KLayout macro that runs `openlithohub optimize` on the currently
selected shapes.

This is a **v0 reference integration** — the goal is to demonstrate the
plumbing end-to-end, not to ship a polished plugin. PRs that turn it into a
proper Salt package, add background-thread inference, or integrate with
KLayout's progress UI are very welcome.

## How it works

The macro **shells out** to the `openlithohub` CLI in a subprocess instead
of importing the model in-process. KLayout ships its own bundled Python
without `pip`, so injecting `torch` + `openlithohub` into that interpreter is
a packaging mess. Subprocess uses the user's regular Python where
`pip install openlithohub` already works.

```
KLayout (pya)                     User's Python env
┌──────────────────┐              ┌────────────────────────┐
│ select shapes    │              │                        │
│ write tile.oas   │  subprocess  │ openlithohub optimize  │
│ ─────────────────┼─────────────▶│   --input tile.oas     │
│ read result.oas  │◀─────────────┤   --output result.oas  │
│ insert as cell   │              │                        │
└──────────────────┘              └────────────────────────┘
```

## Install

1. Install OpenLithoHub in any Python environment that's on your `PATH`:

   ```bash
   pip install openlithohub
   which openlithohub  # confirm it's on PATH
   ```

2. Copy this folder into your KLayout user macro directory:

   - **macOS / Linux**: `~/.klayout/salt/openlithohub_macro/`
   - **Windows**: `%HOMEPATH%\KLayout\salt\openlithohub_macro\`

   ```bash
   cp -R klayout/openlithohub_macro ~/.klayout/salt/
   ```

3. Restart KLayout.

## Use

1. Open a layout (`File → Open`).
2. Select the shapes you want to optimize. (If nothing is selected, the macro
   falls back to the current viewport bounding box.)
3. `Tools → OpenLithoHub: Optimize Selection`.
4. Watch the log panel — it shows the subprocess stdout. On success a new
   top-level cell named `OPC_<timestamp>` is created with the optimized
   geometry, and the view jumps to it.

## Limitations (v0)

- **Synchronous**: the GUI blocks while the subprocess runs. For small tiles
  this is sub-second; for large tiles plan accordingly.
- **No progress bar in the GUI**: only the log panel.
- **Hardcoded model**: rule-based OPC. Edit `openlithohub_optimize.py:113`
  to use `levelset-ilt`, `neural-ilt`, etc.
- **Not on the KLayout Salt registry yet**: install manually for now.

## Manual smoke test

There is no automated test (CI doesn't have a KLayout GUI). To verify after
install:

1. Open a small `.oas` file in KLayout.
2. Select a single shape.
3. Run the menu item.
4. Confirm a new `OPC_*` cell appears and is non-empty.
