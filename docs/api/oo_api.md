# Object-oriented API: `Mask`, `LitheEngine`, `Report`

The OO façade is a thin wrapper over the existing functional API. It
exists so fab-/EDA-shaped callers can think in **masks** and **engines**
without going through the tensor + registry plumbing directly.

The functional API is unchanged — `compute_epe`, `models.registry`, and
`workflow.tiling.tile_layout` all work exactly as before. Use whichever
shape fits the task.

## Quick start

```python
from openlithohub import Mask, LitheEngine

mask     = Mask.from_oasis("design.oas", layer="1:0", pixel_size_nm=1.0)
engine   = LitheEngine(model="neural-ilt", node="3nm-euv")
optimized = engine.optimize(mask)
report    = engine.evaluate(optimized, target=mask)

print(report.epe_mean_nm, report.pvband_mean_nm, report.drc_violations)
optimized.to_oasis("optimized.oas")
```

## What goes where

| Concept                          | Class           | Wraps                                                  |
| -------------------------------- | --------------- | ------------------------------------------------------ |
| `(tensor, pixel_size_nm, layer)` | `Mask`          | `data.io.load_layout`, `workflow.export.export_oasis`, `workflow.export.export_gds` |
| Run a model on a layout          | `LitheEngine`   | `models.registry`, `workflow.tiling`, `workflow.halo`  |
| Aggregate metric & compliance    | `Report`        | `benchmark.metrics.*`, `benchmark.compliance.*`        |

## Constructors

`Mask` offers explicit and suffix-sniffing constructors:

```python
Mask.from_tensor(t, pixel_size_nm=0.5, layer="1:0")
Mask.from_pt("design.pt")
Mask.from_npy("design.npy")
Mask.from_oasis("design.oas", layer="1:0")
Mask.from_gds("design.gds", layer="1:0")
Mask.from_def("routed.def", layer="1:0", lef_files=["stdcells.lef"])
Mask.load("design.oas", layer="1:0")    # dispatches by file suffix
```

`Mask` is a frozen dataclass — once constructed, the `(tensor, pixel_size_nm,
layer)` triplet cannot be mutated. Build a new `Mask` if you need a
different pitch.

## Backward compatibility

`engine.optimize(...)` and `engine.evaluate(...)` accept either a `Mask`
or a raw `torch.Tensor`. Existing tensor-first callers do not need to
change.

## Reference

::: openlithohub.api.mask
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.api.engine
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.api.report
    options:
      show_root_heading: true
      members_order: source
