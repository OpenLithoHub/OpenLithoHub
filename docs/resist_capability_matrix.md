# Resist / Lithography Capability Matrix

This document maps every resist and lithography forward-model implementation across the
OpenLithoHub ecosystem, clarifying fidelity levels, parameter surfaces, and intended use
cases so that contributors can choose the right backend and avoid accidental duplication.

| Implementation | Location | Fidelity | Key Parameters | Use Case |
|---|---|---|---|---|
| `apply_differentiable_resist` | `openlithohub/_utils/resist_model.py` | Light | `threshold`, `steepness`, `resist_diffusion_nm`, `quencher` | Built-in default for ILT/OPC optimization loops. Dispatches to sigmoid threshold or soft CAR model depending on whether diffusion/quencher are set. |
| `HopkinsSimulator` | `openlithohub/simulators/hopkins_sim.py` | Medium | Hopkins SOCS kernels, partial coherence (`sigma`, `sigma_inner`), illumination shape, `dose`, `threshold` | Core lithography forward model. Bundled reference backend with full SOCS decomposition. |
| `DiffCFDLithoSimulator` (Dill + Mack) | `openlithohub/plugins/diffcfd_process.py` | High | `dill_A`, `dill_B`, `dill_C`, `r_max`, `r_min`, `mack_n`, `mack_a`, `gamma_solvent`, `thickness_m`, `residual_solvent`, `dev_time_s` | High-fidelity exposure + development chain via `diffcfd` plugin. Requires `[diffcfd]` extra. |
| `DiffNanoResistAdapter` (CAR/PEB) | `openlithohub/plugins/diffnano_resist.py` | High | `acid_diffusion_length_nm`, `development_contrast`, `threshold_dose`, `peb_diffusion_nm`, `pixel_size_nm` | High-fidelity chemically-amplified resist with acid diffusion + PEB via `diffnano` plugin. Requires `[diffnano]` extra. |
| `HopkinsLithoModel` | `diffnano/solvers/litho.py` | Medium | `wavelength_nm`, `na`, `sigma_source`, `n_kernels`, `pixel_size_nm`, `resist_threshold`, `resist_beta` | Hopkins PSF model in DiffNano. Functionally equivalent to OpenLithoHub's `HopkinsSimulator` for the PSF convolution + sigmoid resist path. See dedup note below. |
| `LithoSolver` (Dill + Mack) | `diffcfd/solvers/litho.py` | High | `dill_A`, `dill_B`, `dill_C`, `r_max`, `r_min`, `mack_n`, `mack_a`, `gamma_solvent` | Dill exposure + Mack development solver in DiffCFD. Wrapped by `DiffCFDLithoSimulator` in OpenLithoHub. |
| `DifferentiableResistModel` | `diffnano/solvers/resist.py` | High | `acid_diffusion_length_nm`, `development_contrast`, `threshold_dose`, `peb_diffusion_nm` | Full CAR/PEB analytical model in DiffNano. Wrapped by `DiffNanoResistAdapter` in OpenLithoHub. |

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
