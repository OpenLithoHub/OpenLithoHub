# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **RDP vertex decimation on OASIS export** — `export_oasis_mbw(vertex_tolerance_nm=...)` runs an iterative anchored Ramer-Douglas-Peucker simplification on each sampled curvilinear polygon. Default `0.0` keeps bit-exact academic behaviour; positive values cut full-chip OASIS data volume (MBMW shot/byte budget) without measurable wafer-image change. Reduction count and ratio logged at INFO.
- **ILT checkpointing** — `LevelSetILTModel.predict(checkpoint_dir=, save_freq=, resume_from=)` periodically `torch.save`s the mask logit, Adam state, and best-loss tracker. Deterministic resume (resume-vs-uninterrupted equality is pinned by test). SLURM preemption / CUDA crash on multi-thousand-iter runs no longer wipes prior progress. Off by default; `save_freq>0` without `checkpoint_dir` raises.
- **SRAF min-area export filter** — `export_oasis_mbw(min_area_nm2=...)` and the matching `workflow.export.{export_oasis,export_gds}` parameter drop sub-resolution polygons via shoelace area before OASIS insert (default `0.0`, Hackathon-safe). Plumbed through `optimize --export-min-area` and the `/v1/optimize` HTTP form so fab-ready exports can clear MRC without touching academic scoring runs. Dropped count logged at INFO.
- **Deterministic mode** — `openlithohub._utils.determinism.set_deterministic()` centralises the four torch backend flags needed for bit-reproducible scoring (`cudnn.deterministic`, `cudnn.benchmark=False`, `allow_tf32=False` on cudnn + matmul). Exposed via `--deterministic` on `openlithohub optimize` and `openlithohub eval`; off by default (the flags carry a real perf cost).
- **HF Hub SHA256 verification** — `ModelHub.download_weights` now verifies an expected `sha256` digest on the HuggingFace Hub path (previously only the direct-URL path did). Mutable revisions (branch names like `main`) trigger a warning so users know the digest can drift between fetches; pin a commit SHA or tag for reproducible scoring. Closes #20.
- **Measured-source / Zernike-pupil I/O on simulators public API** — `load_source_intensity`, `load_zernike_coefficients`, and `zernike_phase_map` are now re-exported from `openlithohub.simulators` (previously defined in `_utils/optics.py` with full test coverage but no `src/` callers). The README and v0.1 milestone advertised this feature; it is now backed by a public import path. Closes #65.
- **Multi-patterning regime guidance** — `levelset-ilt` README and docstring now document that the model targets single-exposure regimes; multi-patterning (LELE / SAQP / SADP) requires upstream colouring + per-mask runs.
- **DRC-vs-MRC count semantics** — eval-aggregation docs and `_repr_html_` panels disambiguate DRC violation count (per-rule hard-fail) vs MRC violation count (geometric width/spacing samples). Same number, different denominator.
- **`openilt` baseline model** (`openlithohub.models.openilt`) — clean-room
  PyTorch reimplementation of the OpenILT SimpleILT formulation
  (MIT-licensed upstream pinned at commit
  [`dabb97c`](https://github.com/OpenOPC/OpenILT/commit/dabb97c6ca3dfd159362e48273c436444c77353b)).
  Optimises the MOSAIC L2 + PVBand objective (Gao et al., DAC 2014) with
  SGD across a 3-corner dose/defocus sweep, distinct from `levelset-ilt`'s
  single-corner Adam loop. Reuses the existing Gaussian / Hopkins forward
  models. Closes #17.
- **L2 wafer-error metric** (`openlithohub.benchmark.metrics.l2_error`) —
  Neural-ILT canonical printability metric: `compute_l2_error()` returns
  `L2ErrorResult` with `l2_error_pixels` and `l2_error_nm2` between the
  forward-simulated wafer image and the target. Complements EPE for
  callers training against the same loss the upstream Neural-ILT paper
  reports.
- **Wafer-level EPE via forward physical simulation** —
  `compute_epe(..., simulate=True)` passes the predicted mask through the
  Hopkins/Gaussian forward model before extracting contours, producing
  the contest-canonical "wafer EPE" rather than the previous mask-vs-mask
  contour distance. Wired through `openlithohub eval run`.
- **GDSII export** (`openlithohub.workflow.export_gds`) — companion to
  `export_oasis`. GDSII is the academic/contest lingua franca (ICCAD,
  SPIE benchmarks). Manhattan masks dump as rectangles; curvilinear
  masks vectorise to polygons via klayout (GDSII has no native curve
  primitive).
- **ICCAD'13 gauge file IO** (`openlithohub.workflow.gauges`) — round-trip
  reader/writer for the contest gauge format alongside the existing
  Calibre `.gg` and CSV parsers, so contest-style `(x, y, angle, target)`
  EPE-evaluation tables drop into the gauge pipeline directly.
- **ONNX-runtime CI smoke test** — extends the existing `openlithohub
  export` ONNX path with a CI smoke test that loads the exported model
  via `onnxruntime` and verifies a single forward pass agrees with the
  PyTorch reference, catching dynamo/onnxscript regressions before they
  reach users.
- **DEF/LEF layout ingestion** — `openlithohub.workflow.parse_layout`
  now accepts `.def` and `.lef` inputs in addition to OASIS/GDSII.
  Pass `lef_files=[...]` to feed cell abstracts when reading a DEF file.
  Closes the gap from RTL-to-GDSII flows (Innovus / ICC2 / OpenROAD)
  that emit DEF as their canonical interchange format.
- **OpenAccess layer-purpose helper** (`openlithohub.workflow.layer_purpose`) —
  canonical purpose-name → integer map mirroring the OpenAccess (Si2)
  default registry plus a permissive `classify_purpose()` alias resolver.
  Lets downstream tooling branch on `(layer, datatype, purpose_name)`
  whether the input came from Cadence (oaPurpose) or OASIS (datatype).
- **Croissant dataset metadata** — dataset adapters expose
  `croissant_name` / `croissant_description` / `croissant_license_url` /
  `croissant_url` / `croissant_citation` properties so OpenLithoHub
  datasets can emit
  [Croissant](https://github.com/mlcommons/croissant) JSON-LD for
  ML-data discoverability.
- **Anamorphic demag flags for High-NA EUV** — `ProcessNode` gains
  `demag_scan` / `demag_slit` (both default to 4.0) and an
  `is_anamorphic` property. ASML's High-NA EXE:5000 class (NA=0.55) is
  8× along scan, 4× along slit; recording demag here unblocks downstream
  anamorphic imaging and reticle-area accounting.
- **imec-style stochastic defect classification**
  (`openlithohub.benchmark.metrics.compute_stochastic_defect_classes`,
  `StochasticDefectRates`) — per-class failure rates in failures/cm²
  for the four canonical EUV stochastic-defect classes (microbridge,
  break, missing contact, merging contact), following the imec
  defectivity-rate convention. Complements the existing aggregate
  `compute_stochastic_robustness`. Includes a shared `_NominalState`
  cache and per-component bridge/break detection for accurate counting
  on tiles with multiple disconnected line segments.
- **Object-oriented API façade** (`openlithohub.api`) — `Mask`,
  `LitheEngine`, and `Report` re-exported at the package root
  (`from openlithohub import Mask, LitheEngine`). Thin wrapper over the
  existing functional API for fab-/EDA-shaped callers who think in masks
  and engines, not tensors and registries. `Mask` is a frozen dataclass
  carrying `(tensor, pixel_size_nm, layer)` with explicit and
  suffix-sniffing constructors (`Mask.from_oasis`, `Mask.from_pt`,
  `Mask.from_npy`, `Mask.from_gds`, `Mask.load`); `LitheEngine` exposes
  `optimize` / `evaluate` / a public `load_layout` plus a teardown
  lifecycle so model resources release cleanly; `Report` aggregates
  metrics, compliance, and tile/halo provenance. The functional API is
  unchanged. Closes #10.
- **Differentiable curvilinear MRC loss** (`openlithohub.benchmark.metrics.curvilinear_mrc_loss`) —
  three-term penalty (min-CD via soft morphological opening, min-spacing
  on the inverted mask, min-curvature via boundary-band gradient
  magnitude) that drops into ILT / level-set / Neural-ILT training loops.
  PDK-first contract: pass `pdk="asap7"` / `pdk="freepdk45"` / a
  `PdkRules`, or supply explicit `min_width_nm` / `min_spacing_nm` /
  `pixel_size_nm`; per-rule kwargs win over the preset. Mirrors the
  binary verdict in `compliance.mrc.check_mrc` so loss and verdict agree
  on what a violation is. Closes #8.
- **SRAF non-printing penalty** (`openlithohub.benchmark.metrics.sraf_print_penalty`) —
  differentiable squared-ReLU loss that punishes SRAF-region aerial
  intensity rising above a configurable `print_threshold - margin`.
  Drop-in for any `torch.optim` ILT loop; complements the post-hoc
  `compliance.mrc` check by catching the failure mode while gradients
  still flow.
- **Process-window-aware OPC workflow** (`openlithohub.workflow.process_window`) —
  `ProcessWindowCorner` dataclass, `DEFAULT_PW_CORNERS` (5-corner
  dose × focus sweep), `pw_aerial_images`, and `pw_fidelity_loss`
  (weighted-MSE across corners). `LevelSetILTModel.predict()` gains
  opt-in `process_window: bool = False` and `pw_corners` kwargs so
  callers can co-optimise against the corner sweep instead of the
  nominal point. Metadata records `process_window` and
  `pw_corner_count`. Defaults stay nominal — no API break.
- **Auto-calibration notebook** (`notebooks/auto_calibration.ipynb`) —
  end-to-end demo of inverting measured-vs-simulated CD error onto
  resist-threshold and Gaussian-σ parameters using `torch.optim.Adam`.
  Runs on CPU in <30 s; pre-fit MAE 1.999 px → post-fit MAE ~0 px on
  the synthetic gauge table.
- **Sharded CI test job** — `.github/workflows/ci.yml` splits the
  pytest run into 5 directory shards (`models`, `workflow`,
  `benchmark`, `data-utils`, `other`) × 3 Python versions = 15
  parallel jobs, each running `pytest -n auto` (workflow shard runs
  serially to avoid spawn-context nesting). Wall-clock dropped from
  30+ min monolithic to ~9 min sharded.
- **`openlithohub serve` HTTP micro-service** (`openlithohub.server`) —
  FastAPI app exposing `GET /v1/health`, `GET /v1/models`, and
  `POST /v1/optimize` so fab-side schedulers (Slurm, LSF) and legacy
  C++/Perl pipelines can drive the optimization engine without
  embedding the Python interpreter. Models are loaded lazily and
  cached in-process; new `[server]` extra pulls in
  `fastapi` / `uvicorn` / `python-multipart`. See the `serve` section
  of the CLI reference for the curl example.
- **Jupyter `_repr_html_` for result dataclasses** —
  `PredictionResult`, `MRCResult`, `CurvilinearMRCResult`, `DRCResult`,
  `MonteCarloFailureResult`, and `SimulatorResult` now render as
  inline HTML panels (pass/fail badge, key/value table, violation
  rows, mask thumbnail) when displayed in Jupyter / Colab / VS Code.
  Helpers live in `openlithohub.jupyter._html` and degrade gracefully
  to plain `repr` when matplotlib is unavailable.
- **RFC 0003 — Standard MRC rule-deck schema**
  (`docs/rfcs/0003-mrc-rule-deck-schema.md`). A single JSON/TOML
  format covering every parameter the OpenLithoHub MRC checkers
  consume (`min_width_nm`, `min_spacing_nm`, `min_curvature_radius_nm`,
  `min_feature_area_nm2`) plus provenance/notes. New
  `openlithohub.benchmark.compliance.load_rule_deck()` validates the
  file against the in-tree schema (Draft 2020-12) and exposes
  `RuleDeck.kwargs_manhattan()` / `kwargs_curvilinear()` adapters to
  the existing `check_mrc` / `check_curvilinear_mrc` functions. Ships
  with a worked example (`benchmark/compliance/rule_decks/freepdk45_metal1.json`).
- **Measured-source / Zernike-pupil I/O** (`openlithohub._utils.optics`) —
  load lithography source maps and pupil aberrations from common formats
  for use with the Hopkins/SOCS forward model.
- **Calibre / CSV gauge parser** (`openlithohub.workflow.parse_gauge`) —
  ingests Calibre `.gg` and CSV gauge files and refuses unrecognized
  headers (rather than silently falling back to a wrong canonical
  column order, which would produce incorrect EPE numbers).
- **`openlithohub export` CLI** — exports trained models to
  ONNX / TorchScript / TensorRT-ready artifacts. Uses the dynamo ONNX
  path with a TorchScript fallback for models that aren't yet
  `torch.export`-able (e.g. NeuralILT). New `[export]` extra pulls in
  `onnxscript`.
- **End-to-end leaderboard submission test** — drives the full
  `auto-leaderboard.yml` pipeline (yaml load → schema validate →
  on-disk JSON) and asserts hostile YAML cannot inject extra fields,
  override `submission_id`, or smuggle Python objects.
- **`scripts/build_litho_tiny.py`** — deterministic 100-pair generator
  emitting an HF-ready parquet + dataset card under `out/litho-tiny/`.
- **HuggingFace authentication guide** (`docs/hf-auth.md`) — single-page
  walkthrough for unblocking gated Hub datasets (request access →
  `huggingface-cli login` / `HF_TOKEN` → verify), with corporate-proxy
  notes linking to `networking.md`. The `LithoSim` adapter's HTTP 401
  remediation now points users at this page.
- **`describe_simulators()`** in `openlithohub.simulators` — public
  `(name, class)` accessor for the simulator registry. Used by the CLI
  (`simulate list-backends --verbose`) to print the implementing class
  path so users can locate the source without grepping the registry.

### Changed

- **MRC `actual_nm` reports feature spine, not edge-pixel distance** — `compliance.mrc.check_mrc` now samples the local distance-transform maximum within the violating component (feature spine) instead of the edge-pixel value. The reported number is now the actual narrow-feature width, matching what foundry MRC docks would print.
- **Hopkins dose application** — `simulate_aerial_image_hopkins` no longer multiplies dose into both the aerial and the binarisation threshold (the previous double-application made `dose` cancel under the constant-threshold-resist path). Dose now affects the aerial image only; threshold is dose-independent. Closes #52.
- **Polar-grid Jacobian on Hopkins illumination samples** — source-sample weighting now applies the polar-grid Jacobian, fixing a systematic bias toward on-axis samples. Canonical Hopkins aerial mean baselines were rebaselined as part of the fix. Closes #29.
- **ILT receptive field lifted from 0 → 64** — `levelset-ilt` and `openilt` now declare a 64-px receptive field, so `--halo auto` accounts for the ILT spread when computing tile halos. Closes #75.
- **EUV H-V CD bias measured in shadowed-mask domain** — `compute_euv_hv_cd_bias` now operates on the shadowed-mask image rather than the pre-shadow aerial, matching the physical observable. Closes #24.
- **Eval-aggregation per-metric weighting + empty-mask floor** — aggregations now apply explicit per-metric weights and floor empty-mask tiles to a pass result instead of NaN-dropping silently.
- **Stochastic resist threshold consistency** — `--threshold` plumbed through `eval` / `optimize` / stochastic metrics so all three see the same value; default `0.225` everywhere. Closes #19, #33.
- **OASIS export single-tile shortcut** — when a layout is smaller than `tile_size`, the tiling pipeline now skips redundant tile dispatch and runs a single in-memory pass.
- **`align_resolution` binary-safe + 4D + deterministic** — re-binarises after rescale so masks stay {0,1}, supports 4D batched tensors, and is deterministic on non-integer scale factors.
- **`openlithohub simulate list-backends --verbose`** — adds a
  `--verbose`/`-v` flag that prints `name  module.ClassName` for every
  registered backend; bare invocation stays script-friendly (one name
  per line).

- **`--compile` defaults to `True`** on the `eval` and `optimize` CLI
  commands, with a graceful fallback to eager when `torch.compile`
  fails (Windows / non-Triton environments stay alive). The existing
  `--no-compile` escape hatch is preserved.
- **`README.md`** — prominent star CTA at the top and a JIT-acceleration
  bullet calling out the default `torch.compile` wrap.
- **`mypy --strict` enforced in CI**; pre-existing type errors cleared.

### Fixed

- **Monte-Carlo dose jitter applied as post-hoc aerial scaling** — previously `dose_jitter_sigma` was plumbed via `config.dose`, which (because `threshold = cfg.threshold * cfg.dose`) cancelled out and produced no observable jitter. Jitter now scales the aerial image and threshold offset *outside* the simulator config so the cancellation cannot bite. Closes #54.
- **Simultaneous bridge + break detection** — `_bridge_and_break_versus` builds nominal and trial component-label maps and detects each axis independently (a single trial can register on both); previously a trial that bridged one pair *and* broke a third left the net component count unchanged and was silently classified as a no-op. `failure_probability` is now clamped to `[0, 1]` instead of summing past 1. Closes #55.
- **forward_model 1-px axis raises instead of silent replicate fallback** — `simulate_aerial_image` on a degenerate H=1 or W=1 input now raises `ValueError`; the previous silent replicate-padded fallback produced meaningless aerials. Closes #10.
- **`auto_crop` replicate padding** — boundary crops near image edges now use replicate padding instead of zero-padding, eliminating a halo bias at the crop edge. Closes #32.
- **Process-window caveats documented** — `process_window` workflow docstring now states that the 5-corner sweep is a coarse approximation; production callers should pin their own corner set. Closes #27.
- **SVRF micron thresholds in `eda_bridge`** — SVRF rule decks emitted by the EDA bridge now use micron units (not nanometres) per Calibre conventions; curvilinear scope is documented. Closes #50, #51.
- **TorchScript `--verify` round-trip** — `openlithohub export run --verify` reloads the TorchScript artifact and checks the forward pass agrees with the PyTorch reference, catching dynamo / scripting regressions before they reach users.
- **Symmetric EPE + Hungarian hotspot match** — `compute_epe` now averages predicted-vs-target and target-vs-predicted contour distances; hotspot matching uses Hungarian assignment instead of greedy nearest-neighbour. Gauges report single-edge EPE consistently.
- **PVB Gaussian-vs-SOCS forward model documented** — PV-Band metric uses a fast Gaussian-PSF approximation, *not* the Hopkins/SOCS path used by `compute_l2_error` / `compute_wafer_epe`. The benchmarks doc now states this explicitly so callers do not assume PVB and L2 share a forward pass.
- **vis/contours figure leak** — opt-in `close=True` on `plot_contours` and friends so notebook callers don't leak matplotlib `Figure`s.
- **Polygon raster hole semantics** — `rasterize_polygons` now treats CCW outer rings as solid and CW inner rings as holes (consistent with OGC simple-features), not foreground regardless of orientation.
- **`contour_trace` small-feature handling** — single-pixel and single-row features are no longer silently dropped by the tracer.
- **BSpline diagnostics** — `BSplineCurve.evaluate` now raises with a clear message when control points are colinear (previously produced NaN coordinates downstream).
- **Trust-root manifest helpers + lithosim revision pin** — adapter integrity warnings on dataset open, manifest helpers exposed, and the LithoSim adapter pins a default revision so `huggingface_hub` cache misses don't silently pull `main`.
- **LitheEngine threads node-bound simulator into wafer metrics** — the OO façade no longer constructs an ad-hoc default simulator inside `evaluate`; the node-bound one configured on the engine is reused so wafer EPE and L2 see consistent optics.
- **VSB shot count via rectilinear decomposition** — `vsb_shot_count` now decomposes the mask into axis-aligned rectangles; the previous `perimeter² / area` heuristic over-counted Manhattan masks and under-counted curvilinear ones.
- **`typecheck` CI on `fb5c9b4`** — dropped two now-unused
  `# type: ignore` comments (`lithobench.py:234`, `ganopc.py:298`) and
  added `multivolumefile.*` / `py7zr.*` to the mypy
  `ignore_missing_imports` overrides. The two ignores were needed
  locally (where `py7zr` is installed and provides typing) but unused
  in CI (where the lazy optional deps are not installed and mypy
  treats the modules as `Any`). Commit `2aa14cb`.
- **Stochastic defect counting + NaN-safe aggregation** — fixed
  net-component-count formula in `compute_stochastic_robustness` so
  trials that simultaneously bridge some lines and break others
  contribute to both bridge and break probabilities. Aggregations in
  `eval run` are now NaN-safe; perimeter computation is border-safe so
  tiles touching the image edge no longer skew per-cm² rates.
- **`OASIS.MBW` → `OASIS.MASK` (SEMI P39)** — corrected naming in
  `workflow/export.py` and `workflow/layer_purpose.py` after community
  feedback that "MBW" is colloquial; SEMI P39 is the canonical name for
  the OASIS mask-data extension. Behaviour unchanged.
- **Contact email unified** — all CLA / SECURITY / DATA-LICENSES /
  COMMERCIAL-USE / community routing now points at
  `support@openlithohub.com`, with `conduct@openlithohub.com` reserved
  exclusively for Code-of-Conduct reports.
- **DRC notch detection** — `compliance.drc._find_notch_violations` now
  rejects background components that touch the image border (those are
  open exterior, not enclosed notches), eliminating false positives at
  tile boundaries. Notch semantics clarified in the docstring and
  covered by new constructed-violation tests.
- **`mypy --strict` regression in `compliance.drc`** — switched from
  `tensor.unique()` to `torch.unique(tensor)` so the typed-call gate
  passes (the bound-method overload is currently unannotated upstream).
- **`pip-audit` CI gate** — 11 unfixed PYSEC torch advisories triaged
  via `.github/pip-audit-ignore.txt` (each requires attacker-controlled
  inputs to specific torch APIs not reachable from OpenLithoHub's data
  path; revisit quarterly).
- **`contour_trace` truncation** — bound raised from `4*(h+w)` to
  `2*h*w` so serpentine boundaries no longer truncate silently.
- **Manhattan tracer X/T-junction ambiguity** — resolved by always
  picking the right-turn edge, keeping foreground consistently on
  the right; new diagonal-touch test.
- **Leaderboard schema lockdown** — `extra='forbid'`, URL-field
  validation, bounded string lengths; hostile-input tests added.
- **Leaderboard tracker** — type-checks `entries` on read;
  `secrets.token_hex(4)` for collision-free submission IDs.
- **`ModelHub._resolve_and_vet`** now returns all vetted IPs and the
  caller iterates with fallback, so dual-stack hosts work in
  IPv6-broken CI.
- **`Iccad16Dataset`** — warns per skipped row and raises if every row
  is malformed (was silent).
- **`workflow.gauges`** — refuses Calibre `.gg` files without a
  recognizable header (was silent fallback to canonical column
  order producing wrong EPE numbers).

## [0.1.0a2] - 2026-05-19

First public alpha. Establishes the `openlithohub` PyPI name; install
with `pip install --pre openlithohub` until a stable `0.1.0` is cut.
API surface is **not** stable.

### Added

- **First PyPI release** — `openlithohub-0.1.0a2` published via GitHub
  Actions trusted publishing (`.github/workflows/publish.yml`) on
  every `v*` tag. `hatch-vcs` derives the version from the git tag.
- **PDK layer registry (`openlithohub.data._layers`)** — single source
  of truth for the (layer, datatype) pairs each adapter rasterizes by
  default (`asap7=10/0`, `freepdk45=11/0`, `orfs_asap7=20/0`). Each
  adapter's `DEFAULT_DESIGN_LAYER` re-exports the registry entry.
- **Docs link-boundary lint (`scripts/lint_docs_links.py`)** — new
  Docs-CI step that fails when a Markdown link in `docs/**` resolves
  outside `docs/`, catching the class of bug that only `mkdocs build
  --strict` surfaces (and only after a page is added to nav).
- **End-to-end URL-cache test for `ModelHub.download_weights`** —
  locks the on-disk shape of URL-keyed cache entries and asserts that
  `list_cached → clear_cache` round-trips cleanly.

### Changed

- **`ModelHub` class docstring** documents the three identifier
  shapes that flow through the cache (`owner/repo`, `owner--repo`,
  `url--<hex>`); auto-rendered onto `docs/api/models.md` via
  mkdocstrings.
- **`mkdocs-material`** pinned to `>=9.4,<10` to avoid the
  backwards-incompatible mkdocs-material/mkdocs 2.0 series that
  drops the plugin system this site depends on (mkdocstrings,
  mkdocs-gen-files).

### Fixed

- **`ModelHub.clear_cache` path traversal** — caller-supplied
  `model_id` now passes through the same `_safe_cache_segment`
  validator as `download_weights`, so a `..` cannot escape `cache_dir`
  and `rmtree` a sibling. URL-keyed entries (`url--<hex>`) are
  accepted in their on-disk form so `list_cached` output round-trips.
- **`OrfsArtifactDataset` docstring layer mismatch** — corrected from
  `(10, 0)` to `(20, 0)` to match `DEFAULT_DESIGN_LAYER` (post-route
  ORFS-ASAP7 numbers M1 differently from the cell-library source).
- **`docs/README_zh.md` license link** — replaced relative `../LICENSE`
  with the canonical GitHub URL so adding the page to the mkdocs nav
  later does not break `mkdocs build --strict`.
- **Pytest deprecation noise** — narrow `filterwarnings` entry
  silences torch's internal `torch.jit.script_method` deprecation
  (14 hits in `tests/test_utils/test_hopkins.py`) without masking
  other warnings.

## [0.1.0a1] - 2026-05-19 [YANKED]

Tag exists in git history but the publish workflow failed at the OIDC
exchange step due to a case-mismatched PyPI trusted publisher
registration. No artifact reached PyPI. Superseded by `0.1.0a2`.

## [0.1.0-pre] - pre-release work

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
