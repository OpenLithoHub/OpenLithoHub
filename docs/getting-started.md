# Getting Started

## Installation

=== "Core (metrics + CLI)"

    ```bash
    pip install openlithohub
    ```

=== "With datasets"

    ```bash
    pip install openlithohub[data]
    ```

=== "Full workflow"

    ```bash
    pip install openlithohub[workflow]
    ```

=== "Everything"

    ```bash
    pip install openlithohub[all]
    ```

### From source (development)

```bash
git clone https://github.com/OpenLithoHub/OpenLithoHub.git
cd OpenLithoHub
pip install -e ".[dev]"
```

## Your First Evaluation

Run the built-in dummy model against synthetic data:

```bash
openlithohub eval run \
  --model dummy-identity \
  --dataset lithobench \
  --data-root ./data/lithobench \
  --format table
```

Output:

```
┌──────────────────┬────────────────┐
│ Metric           │ Value          │
├──────────────────┼────────────────┤
│ epe_mean_nm      │ 0.0000         │
│ epe_max_nm       │ 0.0000         │
│ mrc_violation_rate│ 0.0000        │
│ mrc_passed       │ 1.0000         │
└──────────────────┴────────────────┘
```

## Using as a Python Library

```python
import torch
from openlithohub.benchmark.metrics import compute_epe, compute_pvband
from openlithohub.benchmark.compliance import check_mrc, check_drc

predicted = torch.rand(1, 1, 256, 256) > 0.5
target = torch.rand(1, 1, 256, 256) > 0.5

# Edge Placement Error
epe = compute_epe(predicted.float(), target.float(), pixel_size_nm=1.0)
print(f"EPE mean: {epe['epe_mean_nm']:.2f} nm")

# Process Variation Band
pvb = compute_pvband(predicted.float(), defocus_range_nm=20.0)
print(f"PV Band: {pvb['pvband_mean_nm']:.2f} nm")

# Manufacturing compliance
mrc = check_mrc(predicted.float(), min_width_nm=40.0, min_spacing_nm=40.0)
print(f"MRC passed: {mrc.passed}")
```

## Registering a Custom Model

```python
import torch
from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry

@registry.register
class MyOPCModel(LithographyModel):
    @property
    def name(self) -> str:
        return "my-opc"

    @property
    def supports_curvilinear(self) -> bool:
        return True

    def predict(self, design: torch.Tensor, **kwargs) -> PredictionResult:
        # Your optimization algorithm here
        mask = design  # placeholder
        return PredictionResult(mask=mask)
```

Once registered, your model is available via the CLI:

```bash
openlithohub eval run --model my-opc --dataset lithobench --data-root ./data
```

To evaluate against a real PDK instead of LithoBench, swap `--dataset` for
`asap7`, `freepdk45`, or `orfs` and pass `--accept-license` to acknowledge
the upstream PDK terms. See [Benchmarks](benchmarks.md) for the full
ORFS-routed RISC-V mock-alu walkthrough.

## Try It Without a Dataset

If you don't have LithoBench/LithoSim handy (or you're running on Colab), you
can still exercise the full pipeline using the bundled dummy layout
generator — it's deterministic, DRC-clean, and depends only on NumPy and
PyTorch:

```python
import torch
from openlithohub.data import generate_dummy_layout
from openlithohub.benchmark import compute_epe, compute_pvband
from openlithohub.vis import plot_contours

target = generate_dummy_layout(size=256, seed=0)
predicted = generate_dummy_layout(size=256, seed=1)

print(compute_epe(predicted, target, pixel_size_nm=1.0))
print(compute_pvband(predicted))

# Vector PDF, IEEE column-width, colorblind-safe palette
plot_contours(target, predicted, save_path="result.pdf", style="ieee")
```

For a click-to-run version, open
[`notebooks/quickstart.ipynb`](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/notebooks/quickstart.ipynb)
in Google Colab.

## Bridging to Commercial EDA

After you `export_oasis(...)`, you can emit minimal Calibre / IC Validator
rule decks alongside the OASIS file so layout engineers can sanity-check it
in their existing toolchain:

```python
from openlithohub.workflow import (
    BridgeRules,
    emit_bridge_bundle,
    export_oasis,
)

export_oasis(mask, "optimized.oas", mode="curvilinear")
emit_bridge_bundle(
    "optimized.oas",
    BridgeRules(min_width_nm=40.0, min_spacing_nm=40.0),
)
# → optimized.svrf, optimized.rs, optimized.bridge.md
```

## Next Steps

- Read the [Architecture](architecture.md) guide to understand the system design
- Browse the [CLI Reference](cli-reference.md) for all available commands
- Check [API Reference](api/data.md) for detailed module documentation
