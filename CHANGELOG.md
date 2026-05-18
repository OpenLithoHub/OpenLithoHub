# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
