# Architecture

OpenLithoHub uses a layered architecture designed for extensibility and separation of concerns.

## Layer Overview

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **API façade** | `openlithohub.api` | Object-oriented entry points (`Mask`, `LitheEngine`, `Report`) re-exported at the package root for fab-/EDA-shaped callers |
| **Data** | `openlithohub.data` | Dataset loading, format conversion, resolution alignment, dummy layout generation |
| **Benchmark** | `openlithohub.benchmark` | Metrics computation, compliance checking, reporting |
| **Models** | `openlithohub.models` | Abstract model interface, registry, example implementations, model hub |
| **Workflow** | `openlithohub.workflow` | Layout parsing, tiling, contour extraction, OASIS export, EDA bridge templates |
| **Vis** | `openlithohub.vis` | Paper-publication matplotlib helpers (IEEE / SPIE styles, contour overlays, PV-band plots) |
| **Jupyter** | `openlithohub.jupyter` | IPython display helpers and `%load_ext` magics |
| **CLI** | `openlithohub.cli` | User-facing commands via Typer |
| **Leaderboard** | `openlithohub.leaderboard` | SOTA tracking, submission, querying |
| **Forward models** | `openlithohub._utils` | Differentiable Hopkins SOCS imaging, resist simulation, morphology |

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

- **LithoBenchDataset** — loads NumPy `.npy` files from LithoBench (NeurIPS'23)
- **LithoSimDataset** — loads HuggingFace Parquet datasets from LithoSim (NeurIPS'25)
- **GanOpcDataset** — loads ~4875 paired-PNG (target, OPC mask) samples from
  GAN-OPC (TCAD'20) for AI-OPC training
- **Iccad16Dataset** — rasterizes the ICCAD'16 Problem C OASIS layouts via
  klayout and exposes per-case hotspot annotations (no reference mask;
  `LithoSample.mask` is `None`)
- **Dummy generator** — `generate_dummy_layout` / `generate_dummy_pair` produce
  deterministic, DRC-clean synthetic layouts with only NumPy and PyTorch, for
  CI and Colab use.

All adapters produce `LithoSample` records. Mask-optimization adapters yield
paired `(design, mask)` tensors; hotspot-detection adapters set `mask=None` and
expose annotations through `metadata`.

## Benchmark Layer

### Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| EPE | `compute_epe()` | Edge Placement Error between predicted and target contours |
| L2 wafer error | `compute_l2_error()` | Neural-ILT canonical pixel-area error between simulated wafer image and target |
| PV Band | `compute_pvband()` | Process variation band width across dose/focus window |
| Shot Count | `estimate_shot_count()` | Mask write time proxy for MBMW/VSB writers |
| Stochastic | `compute_stochastic_robustness()` | Monte Carlo photon noise bridge/break probability |
| Stochastic defect classes | `compute_stochastic_defect_classes()` | imec-style per-class failure rates (microbridge / break / missing / merging contact) in failures/cm² |
| Hotspot Detection | `compute_hotspot_detection()` | Distance-tolerant point matching → recall / precision / F1 |

### Compliance

| Check | Function | Description |
|-------|----------|-------------|
| MRC | `check_mrc()` | Minimum width/spacing rule check (hard-fail). Reports `actual_nm` as the local distance-transform maximum within the violating component (feature spine), matching what foundry MRC docks print. |
| Curvilinear MRC | `check_curvilinear_mrc()` | Min curvature radius and feature area for post-ILT curvilinear shapes |
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

1. **Layout parsing** (`parse_layout`) — read `.oas` / `.gds` into tensors
2. **Tiling** (`tile_layout`, `stitch_tiles`) — split large layouts into
   manageable tiles with configurable overlap (the **halo**) and stitch
   them back with a ramp-blended overlap. The default halo is computed
   per-run by `workflow.halo.compute_halo_px` from
   `max(ProcessNodeConfig.optical_radius_nm / pixel_nm, model.RECEPTIVE_FIELD_PX)`,
   rounded up to a multiple of 8 — physically motivated rather than a
   single hard-coded constant (RFC 0005). `--halo N` and `--overlap N`
   override the auto value.
   `optimize run --num-gpus N` (`N>1`) shards the tile loop across `N`
   spawn-context worker processes via `workflow.parallel.parallel_tile_inference`;
   the parent process owns the canonical tile geometry, workers each
   instantiate the model from the registry, and results return over an
   `mp.Queue`. The model layer stays untouched so ONNX/TorchScript export
   is unaffected (RFC 0004).
3. **Contour Extraction** — convert binary masks to polygon boundaries
   (manhattan or curvilinear)
4. **B-spline Fitting** — smooth curvilinear contours with configurable
   control point density
5. **OASIS Export** (`export_oasis`) — write polygons to OASIS
6. **Process node presets** (`ProcessNodeConfig`, `get_node`, `list_nodes`) —
   physical parameters for `45nm`, `7nm`, `5nm-euv`, `3nm-euv`, `2nm-euv`
7. **EDA bridge** (`BridgeRules`, `emit_calibre_svrf`, `emit_icv_runset`,
   `emit_bridge_bundle`) — emit minimal Calibre nmDRC and Synopsys IC
   Validator runsets next to an exported OASIS file

## Visualization & Jupyter

`openlithohub.vis` ships paper-publication matplotlib helpers — `plot_contours`
and `plot_pv_band` produce single-panel IEEE / SPIE column-width figures with
a colorblind-safe palette and Type-42 vector PDF defaults via the `paper_style`
context manager.

`openlithohub.jupyter` exposes `display_mask` and `display_comparison` for rich
inline display, plus `%load_ext openlithohub.jupyter` magics for the CLI.

## Forward Models

`openlithohub._utils` contains the differentiable forward models that power
the ILT loops and PV-band metric:

- **Hopkins SOCS** (`simulate_aerial_image_hopkins`, `HopkinsParams`,
  `compute_socs_kernels`) — partial-coherent imaging with circular / annular
  / dipole / quasar illumination and per-(params, grid) kernel caching.
- **Resist simulation** (`simulate_resist`, `simulate_resist_soft`,
  `differentiable_threshold`) — chemically-amplified resist with acid
  diffusion and a sigmoid-based soft threshold.
- **Morphology** (`binary_dilation`, `binary_erosion`, `distance_transform`)
  — GPU-friendly binary primitives shared by metrics and the dummy generator.

### Resist Model

The resist path used by the leaderboard's EPE/PVB scoring is a **constant
threshold resist (CTR) without diffusion**: a fixed sigmoid threshold
applied to the aerial image, no Mack-style acid diffusion, no per-node
calibration. The default cutoff is `threshold = 0.225` (Yang2023_LithoBench
§3.2 / ICCAD16 reference). The same value flows through `openlithohub eval`,
`openlithohub optimize` (`--threshold` flag, default `0.225`), and the
stochastic-defect metrics (`resist_threshold` kwarg) so all three see one
calibration. The leaderboard pins `0.225` — overriding it produces
non-comparable numbers.

**Opt-in acid diffusion (CTR with diffusion)** is available as an opt-in
feature for users who need more physically accurate resist modeling:

- CLI flags: `--resist-diffusion-nm` and `--quencher` (both default 0)
- When enabled, a Gaussian blur (acid diffusion) and quencher subtraction
  are applied before the hard threshold
- Default 0.0 produces bit-identical results to the legacy CTR model
- ILT optimizers accept the same parameters for differentiable diffusion
  via `apply_differentiable_resist()`
- **Incompatible with leaderboard submission** — non-zero diffusion is
  rejected at submit time

**Calibration**: `ResistCalibration.fit()` in `_utils/resist_model.py`
provides a grid-search least-squares calibration from SEM CD anchors. Users
provide `(aerial_intensity, measured_cd_nm)` pairs and get back
`(threshold, resist_diffusion_nm, quencher)`.

This is a deliberate scope decision:

- **Per-node CTR parameters are foundry-confidential.** Real numbers come
  from wafer SEM measurements on a specific resist + track + bake recipe.
  They are not published and cannot ship in an open-source repo regardless
  of effort spent.
- **For benchmark-relative comparison the default CTR is sufficient.** All
  models on the leaderboard are scored against the same CTR, so the ordering
  is meaningful even though the absolute EPE numbers are not predictive of
  wafer print at any specific fab.
- **For absolute wafer prediction, users must self-calibrate.** A user who
  needs to trust an OPC mask through a real fab flow should calibrate
  diffusion parameters for their target node. Relative rankings remain
  meaningful; absolute predictions require user-supplied parameters.

The differentiable variants `simulate_resist_soft` and
`differentiable_threshold` exist for ILT training (the hard step is not
backprop-friendly); they share the same simplification.

## Performance

The Hopkins forward model and the CLI both expose opt-in performance flags.
Default behaviour is unchanged (CPU, fp32, eager) so existing scripts remain
bit-identical.

| Flag | CLI | API | Effect |
|------|-----|-----|--------|
| Device | `--device cuda` | `predict(..., device=...)` | Run kernels and forward on a non-CPU device. |
| Dtype | `--dtype bf16` | `simulate_aerial_image_hopkins(..., dtype=torch.bfloat16)` | Cast the aerial image to bf16. The internal FFT stays in `complex64` because PyTorch's `fft2` does not support complex-bf16, so memory savings come from the squared-magnitude accumulator and the mask copy, not from the FFT itself. |
| Compile | `--compile` | `torch.compile(simulate_aerial_image_hopkins, mode="reduce-overhead")` | Wrap the K-kernel forward in a TorchInductor graph after the SOCS kernels are pre-computed. SOCS computation itself is **not** compiled (Python control flow + SVD is a poor fit). |

The SOCS kernel cache key includes the requested dtype, so flipping between
fp32 and bf16 in a long-running service does not corrupt cached tensors.
Cache keys also include the device, so the cache is safe across mixed CPU /
CUDA workloads.

## Leaderboard

The leaderboard system uses a JSON-backed store with Pydantic validation:

- **Schema** — `BenchmarkResult` model with typed fields for all metrics
- **Tracker** — file-backed store with atomic writes and query filtering
- **CLI integration** — `openlithohub leaderboard view/submit/export` commands

## Optional Physics Plugins

OpenLithoHub supports optional physics plugins via `pip install` extras and
lazy imports. Plugins are **opt-in** — the core package installs and tests
with zero plugins.

### Installation

```bash
pip install openlithohub                    # core only (lightweight)
pip install openlithohub[diffnano]          # + DiffNano EM / resist
pip install openlithohub[diffcfd]           # + DiffCFD spin coating / litho
pip install openlithohub[plugins]           # + all plugins
```

### Plugin Architecture

| Component | Location | Role |
|-----------|----------|------|
| Plugin manifest | `openlithohub.plugins` | Describes known plugins, extras, modules |
| `optional_import()` | `openlithohub.plugins` | Lazy import with actionable error messages |
| `LithoPlugin` protocol | `openlithohub.plugins` | `register()` method all plugins must implement |
| Adapter modules | `openlithohub.plugins.*` | Wrap external solvers as `BaseSimulator` subclasses |
| Registry integration | `simulators.registry` | Lazy-loads plugin backends on first `get_simulator()` call |

### Available Plugins

| Plugin | Extra | Capabilities | Status |
|--------|-------|-------------|--------|
| **DiffNano** | `[diffnano]` | High-precision resist (PEB diffusion + calibration), RCWA / FDTD / FDFD EM solvers for 3D mask effects | Research, not third-party verified |
| **DiffCFD** | `[diffcfd]` | Dill/Mack lithography, Meyerhofer spin coating, joint spin-litho optimization | Research, not third-party verified |

### Resist Backend Selection

`SimulatorConfig.resist_backend` selects the resist model:

- `"ctr"` (default) — built-in constant-threshold resist, bit-identical to legacy
- `"diffnano"` — DiffNano `DifferentiableResistModel` with PEB diffusion and
  node-specific calibration

Plugin backends raise `OptionalPluginError` with a `pip install` hint when not
installed.

### Verification & Reproducibility

Both DiffNano and DiffCFD are early-stage research projects that have **not**
been independently verified by third parties. Plugin-based backends:

- Are **disabled by default** and must be explicitly selected
- Are **incompatible with leaderboard submission** — non-CTR backends are
  rejected at submit time
- May change numerical results — users must document which backend they use

### Simulator Backends Added by Plugins

| Backend name | Plugin | Class |
|-------------|--------|-------|
| `diffnano_rcwa` | DiffNano | `DiffNanoRCWA` |
| `diffnano_fdtd2d` | DiffNano | `DiffNanoFDTD2D` |
| `diffnano_fdfd2d` | DiffNano | `DiffNanoFDFD2D` |
| `diffcfd_litho` | DiffCFD | `DiffCFDLithoSimulator` |
| `diffcfd_spin_coat` | DiffCFD | `DiffCFDSpinCoatSimulator` |
