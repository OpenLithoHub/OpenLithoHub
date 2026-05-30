# OpenLithoHub Ecosystem Capability Matrix

This document provides a unified view of what each repository in the OpenLithoHub
ecosystem provides, which physics solvers are available, how the cross-domain interfaces
work, and -- critically -- what is **not** provided.

## Repositories

| Repository | Role | Language | Maturity |
|---|---|---|---|
| **OpenLithoHub** | Computational lithography benchmarking and workflow toolkit | Python (PyTorch) | Alpha (`0.1.0a2`) |
| **DiffCFD** | Differentiable steady-state CFD for inverse design and RL | Python + Rust (PyO3) | Pre-release, no public push |
| **DiffNano** | Differentiable nanophotonics inverse design | Python (PyTorch) | Pre-release, no public push |
| **diff-surrogate** | Shared surrogate framework (MLP, CNN, ensemble) | Python (PyTorch) | Pre-release, no public push |

## Physics Solvers

### OpenLithoHub (built-in)

| Solver | Module | Physics | Differentiable |
|---|---|---|---|
| Gaussian PSF forward model | `openlithohub._utils.forward_model` | Single-Gaussian convolution | Yes (PyTorch autograd) |
| Hopkins SOCS forward model | `openlithohub._utils.hopkins` | Partial-coherent imaging via SVD-truncated SOCS | Yes (PyTorch autograd) |
| CTR resist | `openlithohub._utils.resist_model` | Sigmoid threshold with optional Gaussian acid diffusion | Yes |
| Calibre nmOPC adapter | `openlithohub.simulators.calibre` | Siemens EDA Calibre forward sim (requires toolchain + license; mock mode available) | No |
| Tachyon adapter | `openlithohub.simulators.tachyon` | ASML Brion Tachyon forward sim (requires toolchain + license; mock mode available) | No |
| LevelSet ILT | `openlithohub.models.levelset_ilt` | Gradient-descent mask optimization | Yes |
| Neural-ILT (U-Net) | `openlithohub.models.neural_ilt` | Learned mask optimization | Yes |
| OpenILT (MOSAIC) | `openlithohub.models.openilt` | L2 + PVB SGD | Yes |
| Rule-based OPC | `openlithohub.models.rule_based_opc` | Bias-rule mask adjustment | No |
| GAN-OPC | `openlithohub.models.gan_opc` | Generative adversarial mask synthesis | Yes |
| Surrogate ILT | `openlithohub.models.surrogate_ilt` | CNN surrogate replacing Hopkins forward | Yes |

### DiffCFD (plugin, `[diffcfd]` extra)

| Solver | Module | Physics | Differentiable |
|---|---|---|---|
| NavierStokes2D | `diffcfd.solvers.navier_stokes_2d` | 2D incompressible NS, SIMPLE, steady-state | Yes (implicit diff via GMRES) |
| HeatTransfer2D | `diffcfd.solvers.heat_transfer` | Conjugate heat transfer coupled with NS | Yes |
| LithoSolver | `diffcfd.solvers.litho` | Dill exposure + Mack development | Yes |
| MeyerhoferSolver | `diffcfd.solvers.spin_coating` | Meyerhofer spin-coating analytical model | Yes |
| RadialThinFilmSolver | `diffcfd.solvers.spin_coating` | Radial PDE thin-film evolution | Yes |
| Frozen eddy viscosity | `diffcfd.solvers.turbulence` | Frozen mu_t from external RANS | Partial (mu_t not differentiable) |

### DiffNano (plugin, `[diffnano]` extra)

| Solver | Module | Physics | Differentiable |
|---|---|---|---|
| RCWASolver | `diffnano.solvers.rcwa` | Rigorous coupled-wave analysis for periodic structures | Yes |
| FDTDSolver2D | `diffnano.solvers.fdtd2d` | 2D FDTD with CPML, TM/TE | Yes (gradient checkpointing) |
| FDTDSolver3D | `diffnano.solvers.fdtd3d` | 3D FDTD with CPML | Yes (gradient checkpointing) |
| FDFDSolver2D | `diffnano.solvers.fdfd2d` | 2D frequency-domain, dense solve | Yes |
| HopkinsLithoModel | `diffnano.solvers.litho` | Gaussian PSF + sigmoid resist | Yes |
| DifferentiableResistModel | `diffnano.solvers.resist` | CAR acid diffusion + PEB + development | Yes |
| LearnedFabModel | `diffnano.solvers.fab_model` | U-Net learned fabrication transfer | Yes |
| NeuralSurrogate | `diffnano.solvers.surrogate` | CNN-accelerated RCWA | Yes |

### diff-surrogate (shared library)

| Component | Module | Purpose |
|---|---|---|
| MLPSurrogate | `diff_surrogate.mlp` | Scalar property prediction (T,P -> density, etc.) |
| CNNSurrogate | `diff_surrogate.cnn` | 2D field prediction (mask -> velocity field) |
| EnsembleSurrogate | `diff_surrogate.ensemble` | K-member ensemble with uncertainty |
| CorrectionPolicy | `diff_surrogate.base` | When to call truth solver |
| AdaptiveCorrectionPolicy | `diff_surrogate.base` | Adaptive correction interval |
| ConvergenceMonitor | `diff_surrogate.convergence` | Hybrid z-score convergence detection |
| TrainingBudget | `diff_surrogate.budget` | Allocate solver calls across regions |
| optimize_multifidelity | `diff_surrogate.multifidelity` | Surrogate + truth alternating optimization |
| robust_design_step | `diff_surrogate.robust_design` | Mask + antithetic + multi-corner |
| geometry (bspline, SDF, projection) | `diff_surrogate.geometry` | Shared geometry operators used by DiffCFD and DiffNano |

## Cross-Domain Interfaces

The **only** cross-domain interface between repos is the **OpenLithoHub plugin system**.
There are no direct imports between DiffCFD and DiffNano.

```
OpenLithoHub
    |
    +--[diffcfd]--> openlithohub.plugins.diffcfd_process
    |                    |-> DiffCFDLithoSimulator  (wraps diffcfd.solvers.litho.LithoSolver)
    |                    |-> DiffCFDSpinCoatSimulator (wraps diffcfd.solvers.spin_coating)
    |
    +--[diffnano]--> openlithohub.plugins.diffnano_em
    |                    |-> DiffNanoRCWA  (wraps diffnano.solvers.rcwa.RCWASolver)
    |                    |-> DiffNanoFDTD2D (wraps diffnano.solvers.fdtd2d.FDTDSolver2D)
    |                    |-> DiffNanoFDFD2D (wraps diffnano.solvers.fdfd2d.FDFDSolver2D)
    |
    +--[diffnano]--> openlithohub.plugins.diffnano_resist
                         |-> DiffNanoResistAdapter (wraps diffnano.solvers.resist.DifferentiableResistModel)

diff-surrogate is a shared dependency, not a plugin:
    DiffCFD, DiffNano, OpenLithoHub all import from diff_surrogate directly.
```

**Key constraint:** Plugin backends produce **non-comparable** metric values. The scored
leaderboard default remains built-in Hopkins + CTR (threshold `0.225`) without diffusion.

## Metrics (OpenLithoHub)

| Metric | Function | Description |
|---|---|---|
| EPE | `compute_epe()` | Edge Placement Error (mask-vs-mask or wafer-level via forward sim) |
| L2 wafer error | `compute_l2_error()` | Neural-ILT canonical pixel-area error |
| PV Band | `compute_pvband()` | Process variation band across dose/focus window |
| Shot Count | `estimate_shot_count()` | Mask write time proxy for MBMW/VSB writers |
| Stochastic robustness | `compute_stochastic_robustness()` | Monte Carlo photon noise bridge/break probability |
| Stochastic defect classes | `compute_stochastic_defect_classes()` | Per-class failure rates (microbridge, break, missing contact, merging) |
| Hotspot detection | `compute_hotspot_detection()` | Distance-tolerant point matching (recall / precision / F1) |
| MRC | `check_mrc()` | Minimum width/spacing rule check |
| Curvilinear MRC | `check_curvilinear_mrc()` | Min curvature radius + feature area |
| DRC | `check_drc()` | Area, notch, width, spacing |

## What Is NOT Provided

### General limitations

- **No industrial validation.** All four repos are early-stage personal research projects
  with no external users and no third-party validation. Do not use for production
  decisions.
- **No foundry-confidential parameters.** Per-node CTR/Dill/Mack/PEB parameters are
  foundry-confidential and cannot ship in open-source repos. Users must self-calibrate.
- **No commercial EDA tool integration in core.** Calibre (`CalibreSimulator`) and Tachyon
  (`TachyonSimulator`) adapters exist as config-validated stubs with `mock_mode` for
  testing. Real simulations require the respective vendor toolchain on `PATH` and a
  license. OpenLithoHub does not ship these tools.

### OpenLithoHub

- No 3D mask effects (only 2D Hopkins imaging with optional EUV 3D-mask shadow proxy)
- No compressible or turbulent flow simulation (that is DiffCFD's domain)
- No electromagnetic simulation beyond Hopkins/SOCS (RCWA/FDTD/FDFD are DiffNano plugin)
- No spin-coating simulation (DiffCFD plugin)
- No Dill/Mack exposure+development solver in core (DiffCFD plugin)
- Neural-ILT and GAN-OPC models are trained on synthetic 64x64 tiles; performance on
  real industrial layouts is unvalidated
- Absolute wafer prediction requires user-calibrated, foundry-confidential parameters

### DiffCFD

- 2D only; 3D out of scope for v0.x
- Incompressible flow only; compressible (shocks, transonic) not supported
- No unstructured meshes (structured Cartesian + Brinkman IB only)
- No turbulence model beyond frozen eddy viscosity (no RANS, no LES, no DNS)
- No multi-physics beyond NS + heat (no radiation, no combustion, no multiphase)
- Tuned for optimization loops at 32x32--64x64, not production-scale simulation
- Rust-accelerated forward only; backward (implicit diff) is pure PyTorch

### DiffNano

- FDTD does not match MEEP or Tidy3D in feature completeness (PML variants, dispersive
  materials, subpixel smoothing)
- RCWA eigendecomposition backward uses broadening-based stabilization (similar to
  published TORCWA approach); not the primary differentiator
- 3D FDTD is CPU-only and slow; no multi-GPU support (FDTDX occupies that niche)
- Metalens workflow uses simplified PSF approximation, not full SOCS
- GDSII export is via gdstk; no direct interface to commercial mask writers

### diff-surrogate

- Provides surrogate models and convergence utilities only; no physics solvers
- No GPU-specific optimizations beyond standard PyTorch CUDA
- Ensemble uncertainty is epistemic (model disagreement), not calibrated Bayesian
