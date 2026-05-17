# Architecture

OpenLithoHub uses a layered architecture designed for extensibility and separation of concerns.

## Layer Overview

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **Data** | `openlithohub.data` | Dataset loading, format conversion, resolution alignment |
| **Benchmark** | `openlithohub.benchmark` | Metrics computation, compliance checking, reporting |
| **Models** | `openlithohub.models` | Abstract model interface, registry, example implementations |
| **Workflow** | `openlithohub.workflow` | Layout parsing, tiling, contour extraction, OASIS export |
| **CLI** | `openlithohub.cli` | User-facing commands via Typer |
| **Leaderboard** | `openlithohub.leaderboard` | SOTA tracking, submission, querying |

## Data Flow

```text
Dataset (LithoBench/LithoSim)
        │
        ▼
┌─────────────────┐
│  DataAdapter    │  Normalize to (B, C, H, W) tensors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LithographyModel│  Predict optimized mask
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Benchmark     │  Compute EPE, PV Band, MRC/DRC
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Workflow      │  Tile → Contour → B-spline → OASIS
└────────┬────────┘
         │
         ▼
   Fab-ready mask (.oas)
```

## Data Layer

The data layer provides unified access to lithography datasets through the `DatasetAdapter` interface:

- **LithoBenchAdapter** — loads NumPy `.npy` files from LithoBench (NeurIPS'23)
- **LithoSimAdapter** — loads HuggingFace Parquet datasets from LithoSim (NeurIPS'25)

All adapters produce `(design, target)` tensor pairs with consistent shape `(B, 1, H, W)`.

## Benchmark Layer

### Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| EPE | `compute_epe()` | Edge Placement Error between predicted and target contours |
| PV Band | `compute_pvband()` | Process variation band width across dose/focus window |
| Shot Count | `compute_shot_count()` | Mask write time proxy for MBMW/VSB writers |
| Stochastic | `compute_stochastic_robustness()` | Monte Carlo photon noise bridge/break probability |

### Compliance

| Check | Function | Description |
|-------|----------|-------------|
| MRC | `check_mrc()` | Minimum width/spacing rule check (hard-fail) |
| DRC | `check_drc()` | Full design rule check: area, notch, width, spacing |

## Model Layer

Models implement the `LithographyModel` abstract class:

```python
class LithographyModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def supports_curvilinear(self) -> bool: ...

    @abstractmethod
    def predict(self, design: Tensor, **kwargs) -> PredictionResult: ...
```

The `ModelRegistry` provides decorator-based registration and lookup by name.

## Workflow Layer

The workflow layer converts tensor masks to fab-ready OASIS files:

1. **Tiling** — split large layouts into manageable tiles with configurable overlap
2. **Contour Extraction** — convert binary masks to polygon boundaries (manhattan or curvilinear)
3. **B-spline Fitting** — smooth curvilinear contours with configurable control point density
4. **OASIS Export** — write polygons to industry-standard OASIS format

## Leaderboard

The leaderboard system uses a JSON-backed store with Pydantic validation:

- **Schema** — `BenchmarkResult` model with typed fields for all metrics
- **Tracker** — file-backed store with atomic writes and query filtering
- **CLI integration** — `openlithohub leaderboard view/submit/export` commands
