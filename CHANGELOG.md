# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Real PDK rollout (issue #4)** — three new dataset adapters that bring OpenLithoHub onto industrial layouts:
    - **`Asap7Dataset`** (`openlithohub.data.asap7`) — loads the BSD-3-Clause [ASAP7 7nm predictive PDK](https://github.com/The-OpenROAD-Project/asap7), exposes a canonical 4-cell smoke set (`INVx1`, `NAND2x1`, `NOR2x1`, `DFFHQNx1`), gated by `--accept-license`. Adds a klayout-based GDS rasterizer reused by the FreePDK45 and ORFS adapters.
    - **`FreePdk45Dataset`** (`openlithohub.data.freepdk45`) — loads FreePDK45 + NanGate Open Cell Library from the [mflowgen mirror](https://github.com/mflowgen/freepdk-45nm); exposes the canonical 4-cell smoke set (`INV_X1`, `NAND2_X1`, `NOR2_X1`, `DFF_X1`); stacked-license disclosure since the mirror ships no LICENSE file.
    - **`OrfsArtifactDataset`** (`openlithohub.data.orfs`) — loads ASAP7-routed RTL→GDSII outputs from [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts), cuts the routed block into 2 µm × 2 µm and 5 µm × 5 µm tiles (canonical AI-OPC inference windows), defaults to ORFS metal1 layer 20/0.
- **`build-asap7-mock-alu` GitHub Actions workflow** (`.github/workflows/build-asap7-mock-alu.yml`) — runs ORFS in the `openroad/orfs` container against pinned commit `74b5f96` and uploads the routed GDS as a workflow artifact (~25 min for `mock-alu`). Companion `scripts/build_riscv_alu.sh` for local Linux runs.
- **CLI `--dataset {asap7,freepdk45,orfs}`** + `--accept-license` and `--tile-nm` flags on `openlithohub eval run`. The CLI now supports five datasets total (LithoBench, LithoSim, ASAP7, FreePDK45, ORFS).
- **Phase-3 baseline (`baselines/orfs-mock-alu-{2um,5um}.json`)** — first numbers against a real ASAP7-routed RISC-V mock-alu. PVB mean 15.07 nm (729 × 2 µm tiles) / 14.98 nm (121 × 5 µm tiles) at `pixel_nm=4.0`.
- **Before/after PNG** at `docs/assets/orfs-mock-alu-tile.png` (design / rule-OPC mask / resist contour) embedded in `docs/benchmarks.md`.
- **RFC 0001 — Layout-MAE base model** (`docs/rfcs/0001-base-model.md`) and **RFC 0002 — Layout Tokens** (`docs/rfcs/0002-layout-tokens.md`) lock in the v0.2 path: a small ViT-S MAE pretrained on rasterised PDK layouts as the open backbone, and a polygon-vertex tokeniser that round-trips losslessly and replaces the diffusion stub with an autoregressive sequence model.
- **Rule-based synthetic layout generator (`openlithohub.synth`)** — PDK-aware patterns (FreePDK45, ASAP7) for SRAM, contact arrays, and randomly routed metal that pass MRC by construction, plus `openlithohub synth` CLI for batch export and a `DiffusionLayoutGenerator` stub pinned to RFC 0001 + 0002.
- **EUV 3D-mask shadow proxy + Monte Carlo failure metric** (`openlithohub.benchmark.metrics.euv_3d`, `openlithohub.benchmark.metrics.monte_carlo`) — first-order anisotropic shadowing operator parameterised by absorber thickness and chief-ray azimuth, plus a higher-fidelity Monte Carlo failure path that runs against any registered simulator backend.
- **Vendor-neutral simulator hook API (`openlithohub.simulators`)** — `BaseSimulator` ABC with a Hopkins reference adapter (`hopkins_sim`) shipping in-tree and config-validated stubs for Calibre nmOPC and Tachyon, exposed via `openlithohub simulate` CLI.
- **Mini-Hackathon (2026-Q3) charter + leaderboard track** (`docs/hackathon.md`) — frozen test split, hard MRC/DRC gate, separate `track` field on leaderboard submissions.
- **Auto-Leaderboard CI** (`.github/workflows/auto-leaderboard.yml`) — claim-and-verify-by-numbers workflow that validates `submissions/*.yaml` against the BenchmarkResult schema. Submission template at `submissions/_template/example-model.yaml`; full guide at `docs/leaderboard-submission.md` (now also documents the optional `track` field).
- **Community charter** (`docs/community.md`) — Discord-only (English-first), launching 2026-Q3. Channel layout, etiquette, moderator policy, onboarding flow.
- **v0.1 launch announcement** (`docs/announcements/2026-05-launch.md`) — paste-ready copy for X / LinkedIn / 知乎 / HuggingFace Forum.
- **AI-engineer terminology guide** (`docs/lithography-for-ai-engineers.md`) — bridges ML vocabulary and lithography terminology for newcomers.
- **Multi-stage KLayout Docker build** — slimmer image, separate build/runtime stages.
- **OpenLithoHub logo** in README and MkDocs (light + dark variants).
- **Paper-ready visualization (`openlithohub.vis`)** — `plot_contours`, `plot_pv_band`, and the `paper_style` context manager (with `IEEE_STYLE` and `SPIE_STYLE` presets) emit IEEE / SPIE column-width figures with a colorblind-safe palette, vector PDF defaults, and Type-42 fonts.
- **Hermetic dummy layout generator** — `openlithohub.data.generate_dummy_layout`, `generate_dummy_pair`, and `DummyLayoutSpec` produce deterministic, DRC-clean synthetic layouts with only NumPy and PyTorch — usable in CI and Colab without the `[workflow]` extras.
- **EDA bridge templates (`openlithohub.workflow.eda_bridge`)** — `BridgeRules`, `emit_calibre_svrf`, `emit_icv_runset`, and `emit_bridge_bundle` write minimal Calibre nmDRC and Synopsys IC Validator runsets next to an exported OASIS file.
- **Colab quickstart** — `notebooks/quickstart.ipynb` runs install → dummy layout → metrics → paper figure end-to-end on Colab's stock runtime.
- **Spaces leaderboard tab** — `spaces/app.py` now ships a third tab that renders the JSON leaderboard with a refresh button.
- **Rule-based OPC model** — analytic per-edge bias OPC baseline registered as `rule-based-opc`.
- **Differentiable Hopkins forward model** — partial-coherent imaging via SVD-truncated SOCS (`openlithohub._utils.hopkins`), supporting circular / annular / dipole / quasar illumination, defocus, and per-(params, grid) kernel caching. End-to-end auto-differentiable so it can drop into AI-OPC training and ILT loops.
- **`LevelSetILTModel.forward_model="hopkins"`** — opt-in switch from the default Gaussian PSF to the new Hopkins SOCS model, with optional `HopkinsParams` override.
- **`differentiable_threshold`** — standalone sigmoid-based resist threshold helper exposed from `openlithohub._utils`.
- **Baseline reference numbers** — `scripts/generate_baselines.py` runs `dummy-identity`, `rule-based-opc`, `levelset-ilt`, and `neural-ilt` against eight synthetic 64×64 layouts (or LithoBench when `--data-root` is supplied) and writes `baselines/results.json` + `baselines/results.md`.
- **ICCAD'16 Problem C hotspot dataset (`openlithohub.data.Iccad16Dataset`)** — klayout-based OASIS rasterizer for the ICCAD 2016 EUV hotspot benchmark. Returns `LithoSample(design, mask=None, ...)` with hotspot annotations and clip-site bboxes in `metadata`.
- **GAN-OPC paired-mask dataset (`openlithohub.data.GanOpcDataset`)** — loader for the ~4875 paired `(target, OPC mask)` 2048×2048 PNGs from Yang et al. *GAN-OPC* (TCAD'20), suitable for AI-OPC training.
- **Hotspot detection metric (`compute_hotspot_detection`)** — distance-tolerant greedy point matching → recall / precision / F1, configurable via `match_radius_nm`.
- **Hotspot baseline pipeline (`scripts/run_hotspot_baseline.py`)** — end-to-end wiring of `Iccad16Dataset` → predictor → metric across three sanity baselines (empty / saturated grid / clip-centers); writes `hotspot_results.{json,md}`.
- **`docs/benchmarks.md`** — new docs page covering baseline numbers, reproduction, and the differentiable forward models.
- **LevelSet-ILT model** — iterative gradient-descent mask optimization using differentiable forward model
- **Neural-ILT model** — U-Net based single-pass mask prediction with pretrained weight support
- **Model Hub** — download and cache pretrained weights from HuggingFace Hub or direct URLs
- **DTCO Process Node Config** — physical parameters for 3nm-euv, 5nm-euv, 7nm, 45nm nodes
- **Resist simulation** — chemically-amplified resist model with acid diffusion and quencher
- **Jupyter integration** — `%load_ext openlithohub.jupyter` magic commands and display helpers
- **PyPI publish workflow** — automated package publishing on version tags
- **Docker image** — containerized deployment via GitHub Container Registry
- **Performance benchmarks** — pytest-benchmark suite for critical paths
- **py.typed marker** — PEP 561 type information support
- `[models]` and `[jupyter]` optional dependency groups
- 73 new tests (217 total), covering utils, models, process nodes, and integration

### Fixed

- `distance_transform` infinite loop on all-foreground masks (pre-existing bug)
- CLI `--node` parameter now auto-configures pixel size and MRC thresholds from process node presets

### Changed

- Project scaffold with 5-layer architecture
- Abstract interfaces: `DatasetAdapter`, `LithographyModel`
- CLI skeleton: `openlithohub eval`, `openlithohub optimize`
- Benchmark metric stubs: EPE, PV Band, shot count, stochastic robustness
- Compliance check stubs: MRC, DRC
- Workflow stubs: layout parsing, tiling, contour extraction, OASIS export
- Leaderboard schemas with Pydantic models
- Model registry with decorator-based registration
- Dummy identity model for pipeline testing
- CI pipeline (GitHub Actions): lint + test on Python 3.10/3.11/3.12
- Pre-commit hooks (ruff, trailing whitespace, YAML/TOML checks)
- MkDocs Material documentation configuration
- Full MkDocs documentation site (getting started, architecture, CLI reference, API docs)
- GitHub Actions workflow for docs deployment to GitHub Pages
- HuggingFace Spaces web playground (Gradio-based interactive demo)
  - Synthetic pattern evaluation (line/space, contact holes, SRAM)
  - Upload custom masks for EPE/MRC evaluation
  - Edge contour visualization overlay
  - Public leaderboard view
