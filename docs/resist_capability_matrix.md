# Resist / Lithography Capability Matrix

This document maps every resist and lithography forward-model implementation across the
OpenLithoHub ecosystem, clarifying fidelity levels, parameter surfaces, and intended use
cases so that contributors can choose the right backend and avoid accidental duplication.

| Implementation | Location | Fidelity | Key Parameters | Use Case |
|---|---|---|---|---|
| `apply_differentiable_resist` | `openlithohub/_utils/resist_model.py` | Light | `threshold`, `steepness`, `resist_diffusion_nm`, `quencher` | Built-in default for ILT/OPC optimization loops. Dispatches to sigmoid threshold or soft CAR model depending on whether diffusion/quencher are set. |
| `HopkinsSimulator` | `openlithohub/simulators/hopkins_sim.py` | Medium | Hopkins SOCS kernels, partial coherence (`sigma`, `sigma_inner`), illumination shape, `dose`, `threshold`, `resist_diffusion_nm`, `quencher`, `resist_backend` | Core lithography forward model. Bundled reference backend with full SOCS decomposition. Supports opt-in acid diffusion and DiffNano resist backend. |
| `CalibreSimulator` | `openlithohub/simulators/calibre.py` | External | `config.extra["calibre_home"]`, `config.extra["runset"]`, `config.extra["mock_mode"]`, `dose`, `threshold` | Siemens EDA Calibre nmOPC adapter. Requires commercial toolchain + license; mock mode for testing. |
| `TachyonSimulator` | `openlithohub/simulators/tachyon.py` | External | `config.extra["tachyon_home"]`, `config.extra["recipe"]`, `config.extra["mock_mode"]`, `dose`, `threshold` | ASML Brion Tachyon adapter. Requires commercial toolchain + license; mock mode for testing. |
| `DiffCFDLithoSimulator` (Dill + Mack) | `openlithohub/plugins/diffcfd_process.py` | High | `dill_A`, `dill_B`, `dill_C`, `r_max`, `r_min`, `mack_n`, `mack_a`, `gamma_solvent`, `thickness_m`, `residual_solvent`, `dev_time_s` | High-fidelity exposure + development chain via `diffcfd` plugin. Requires `[diffcfd]` extra. |
| `DiffNanoResistAdapter` (CAR/PEB) | `openlithohub/plugins/diffnano_resist.py` | High | `acid_diffusion_length_nm`, `development_contrast`, `threshold_dose`, `peb_diffusion_nm`, `pixel_size_nm` | High-fidelity chemically-amplified resist with acid diffusion + PEB via `diffnano` plugin. Requires `[diffnano]` extra. |
| `HopkinsLithoModel` | `diffnano/solvers/litho.py` | Medium | `wavelength_nm`, `na`, `sigma_source`, `n_kernels`, `pixel_size_nm`, `resist_threshold`, `resist_beta` | Hopkins PSF model in DiffNano. Functionally equivalent to OpenLithoHub's `HopkinsSimulator` for the PSF convolution + sigmoid resist path. See dedup note below. |
| `LithoSolver` (Dill + Mack) | `diffcfd/solvers/litho.py` | High | `dill_A`, `dill_B`, `dill_C`, `r_max`, `r_min`, `mack_n`, `mack_a`, `gamma_solvent` | Dill exposure + Mack development solver in DiffCFD. Wrapped by `DiffCFDLithoSimulator` in OpenLithoHub. |
| `DifferentiableResistModel` | `diffnano/solvers/resist.py` | High | `grid_shape`, `dl`, `acid_diffusion_length_nm`, `development_contrast`, `threshold_dose`, `peb_diffusion_nm` | Full CAR/PEB analytical model in DiffNano. Takes `(H, W)` grid shape and grid spacing `dl` (nm). Wrapped by `DiffNanoResistAdapter` in OpenLithoHub, which adds `pixel_size_nm` for nm-to-pixel conversion. |

## Which Model Should I Use?

| Scenario | Recommended Model | Rationale |
|---|---|---|
| ILT gradient descent on 64x64 synthetic tiles | `apply_differentiable_resist` (Light) | Fast, differentiable, no external dependencies. The scored leaderboard path uses this. |
| Mask optimization with realistic aerial images | `HopkinsSimulator` (Medium) | Full SOCS decomposition with partial coherence. Still uses simple threshold resist. |
| Validation against commercial tool | `CalibreSimulator` or `TachyonSimulator` (External) | Uses vendor forward simulator for ground-truth comparison. Requires commercial toolchain + license; `mock_mode` for CI testing. |
| Process-window analysis, CD prediction | `DiffCFDLithoSimulator` via `[diffcfd]` (High) | Physics-based Dill exposure + Mack development. Models PAC bleaching, dissolution rate, residual solvent. Requires calibrated Dill/Mack parameters from the target process. |
| Chemically-amplified resist modeling | `DiffNanoResistAdapter` via `[diffnano]` (High) | CAR acid diffusion + post-exposure bake + development contrast. Suitable for EUV CAR resists where acid diffusion dominates print fidelity. Parameters calibratable to SEM CD data. |
| Joint spin-coating + lithography co-optimization | `DiffCFDLithoSimulator` + DiffCFD spin-coating solver | The only path that couples film thickness variation (from spin coating) into the exposure/development chain. |
| DiffNano-internal metalens with litho DFM | `HopkinsLithoModel` in DiffNano | Uses simplified Gaussian PSF rather than full SOCS. Retained for DiffNano standalone workflows; prefer OpenLithoHub's `HopkinsSimulator` for ILT/OPC work. |

**Important:** Plugin backends (DiffCFD, DiffNano) produce **non-comparable** metric values. The scored leaderboard default remains `HopkinsSimulator` + CTR (threshold `0.225`) without diffusion.

## Fidelity Levels

- **Light** -- Sigmoid threshold with optional Gaussian acid diffusion. No physics-based
  exposure or dissolution model. Suitable for ILT gradient descent where speed matters
  more than absolute CD accuracy.
- **Medium** -- Hopkins/SOCS aerial image with physically correct partial coherence and
  dose scaling. Resist model is still a simple threshold or sigmoid. Good for
  mask-optimization loops that need realistic aerial images.
- **High** -- Full physics-based models (Dill exposure, Mack development, CAR acid
  diffusion + PEB). Suitable for process-window analysis, CD prediction, and joint
  process optimization.
- **External** -- Vendor simulator (Calibre nmOPC, ASML Tachyon). Highest fidelity but
  requires commercial toolchain + license. Mock mode available for API testing without
  the vendor binary.

## Deduplication Notes

### Hopkins / PSF Convolution

`HopkinsLithoModel` in DiffNano (`diffnano/solvers/litho.py`) implements the same PSF
convolution + sigmoid resist pipeline as OpenLithoHub's `HopkinsSimulator` +
`apply_differentiable_resist`. The DiffNano version uses a simplified Gaussian PSF
approximation rather than full SOCS decomposition. For new ILT/OPC work, prefer
OpenLithoHub's `HopkinsSimulator` which supports multiple illumination types and proper
SOCS kernel caching.

### Dill / Mack Solvers

`LithoSolver` in DiffCFD is the canonical implementation. OpenLithoHub wraps it via
`DiffCFDLithoSimulator` (the adapter) and should not reimplement the physics.

### CAR / PEB Resist Models

`DifferentiableResistModel` in DiffNano is the canonical implementation.
OpenLithoHub wraps it via `DiffNanoResistAdapter` and should not reimplement the physics.
