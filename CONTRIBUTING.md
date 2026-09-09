# Contributing to OpenLithoHub

Thank you for your interest in contributing! This guide will help you get started.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Project Scope & Responsibility Boundary

The OpenLithoHub ecosystem consists of two independent repositories with a
clear separation of responsibilities:

| Repository | Responsibility | Deployment |
|------|------|------|
| **This repository (OpenLithoHub)** | Python SDK, metrics, model interface, workflow engine, CLI, technical docs, HF Spaces Playground | docs.openlithohub.com |
| **openlithohub-website** | Brand site, leaderboard front end, blog, community, playground embed | openlithohub.com |

**Boundary principles:**

- This repository owns **data production** (metrics, leaderboard export,
  model inference).
- The website repository owns **data presentation** (renders front end from
  JSON exported by this repository).
- Technical documentation (API reference, Getting Started, Architecture)
  lives in this repository under `docs/`.
- Brand/marketing content (feature pages, blog) lives in the website repo.
- Leaderboard data is exported via `openlithohub leaderboard export` and
  manually updated into the website repo at `src/data/leaderboard.json`.
- The landing-page hero comparison figure is generated with
  `python scripts/generate_hero_figure.py` producing
  `docs/assets/hero.{png,json}`; copy manually to the website repo's
  `public/hero.png` and `src/data/hero.json`.
- Architecture descriptions on the website features page must stay
  consistent with `docs/architecture.md` in this repository.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/OpenLithoHub/OpenLithoHub.git
cd OpenLithoHub

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Project Structure

```
src/openlithohub/
├── api/          # Object-oriented façade (LitheEngine, Mask, Report)
├── baselines/    # Baseline model weights/configs metadata
├── benchmark/    # Layer 2: Metrics, MRC/DRC compliance checks
├── cli/          # Command-line interface (Typer)
├── data/         # Layer 1: Dataset adapters + dummy generator
├── hackathon/    # Hackathon manifest / submission validation
├── inference/    # Multi-process shared-weight inference
├── jupyter/      # IPython display helpers and `%load_ext` magics
├── leaderboard/  # Layer 5: SOTA tracking and data engine
├── models/       # Layer 3: Model integration interface + registry
├── plugins/      # Optional physics plugins (DiffNano / DiffCFD, opt-in)
├── server/       # FastAPI optimization/eval service
├── simulators/   # Simulator backends + registry
├── streaming/    # RFC 0008: core/halo streaming tiling + plugin API
├── synth/        # Synthetic rule-based pattern generation
├── verify/       # B04 / RFC 0007: proof-carrying verification
├── vis/          # Paper-publication matplotlib helpers (IEEE / SPIE styles)
├── workflow/     # Layer 4: OASIS workflow engine + EDA bridge templates
└── _utils/       # Shared internal utilities (Hopkins, resist, morphology)
```

## Running Tests

```bash
# Run all tests
pytest

# Faster: run in parallel with all cores (pytest-xdist is in [dev] extras)
pytest -n auto

# Run with coverage
pytest --cov=openlithohub --cov-report=html

# Run a specific test file
pytest tests/test_models/test_interface.py
```

> **Note**: CI shards the suite into 6 directory groups × 3 Python versions.
> The `tests/test_workflow` shard runs serially because
> `tests/test_workflow/test_parallel.py` already spawns its own subprocess
> pool — nesting it under xdist deadlocks on Linux runners.

## Quality Gates (CI-enforced)

Every PR must pass all five CI jobs — they are hard gates, not warnings:

| Gate | Tool | Notes |
|------|------|-------|
| Lint + format | ruff (incl. `S` security rules) | `ruff check` + `ruff format --check` over all code dirs; `pre-commit run --all-files` also runs here |
| Type check | mypy `--strict` | Package sources only (`src/`); must be error-free |
| Tests | pytest | 6 shards × Python 3.10/3.11/3.12 |
| Coverage floor | pytest-cov | Full-suite run fails below **80%** (`coverage-gate` job; optional physics plugins are omitted from the denominator) |
| Security | pip-audit + bandit | Known-VULN allowlist in `.github/pip-audit-ignore.txt`; bandit gates on medium+ findings |
| License | scancode | Any copyleft (GPL/AGPL) detection fails the build |

Also enforced: README bilingual parity (English ⇄ Chinese H2 sections must
stay aligned via `scripts/check_readme_parity.py`).

## Code Quality

We use **ruff** for linting and formatting:

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

And **mypy** for static typing (strict mode; run over `src/`):

```bash
mypy src
```

Types are load-bearing here — CI fails on any mypy error, and new modules
must ship complete annotations.

## Adding a New Metric

1. Create a new file in `src/openlithohub/benchmark/metrics/`
2. Implement the metric function with proper type annotations
3. Export it from `benchmark/metrics/__init__.py`
4. Add tests in `tests/test_benchmark/`
5. Update the CLI to include the new metric in reports

## Adding a New Model

Implement the `LithographyModel` interface:

```python
from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry


@registry.register
class MyModel(LithographyModel):
    @property
    def name(self) -> str:
        return "my-model"

    @property
    def supports_curvilinear(self) -> bool:
        return True

    def predict(self, design, **kwargs):
        # Your optimization logic here
        return PredictionResult(mask=optimized_mask)
```

## Adding a New Dataset Adapter

Implement the `DatasetAdapter` interface in `src/openlithohub/data/`:

```python
from openlithohub.data.base import DatasetAdapter, LithoSample


class MyDataset(DatasetAdapter):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> LithoSample: ...

    def download(self, root: str) -> None: ...
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all tests pass and linting is clean
4. Open a PR with a clear description of changes
5. Link any related issues

## Code Style

- Python 3.10+ type hints throughout
- Google-style docstrings for public APIs
- No comments unless explaining non-obvious "why"
- Follow existing patterns in the codebase

## Legal & Contributor License Agreement

OpenLithoHub uses a **CLA + Dual Licensing** model. See
[COMMERCIAL-USE.md](COMMERCIAL-USE.md) for the rationale; in short, the
open-source release is permanently Apache 2.0 (CC-BY-SA 4.0 for docs), and
the CLA lets the maintainers also offer an optional commercial license to
fund continued development.

Before your first PR is merged, you (or your employer, for company-owned
work) must sign the appropriate CLA:

- Individuals: [CLA-INDIVIDUAL.md](CLA-INDIVIDUAL.md)
- Companies: [CLA-CORPORATE.md](CLA-CORPORATE.md)

Once the CLA Assistant bot is configured on the repository, it will prompt
you to sign on your first PR. Until then, include the following statement in
your PR description:

```
I have read the CLA Document and I hereby sign the CLA
```

### Third-Party Code

If your contribution uses or adapts third-party code:

- The code must be under an Apache 2.0–compatible license.
- Add an entry to [NOTICE](NOTICE) under the appropriate section.
- Preserve original copyright headers in source files you import or adapt.
- Disclose any third-party license details in your PR description.

### New Dataset Adapters

If your contribution adds a dataset adapter (under `src/openlithohub/data/`):

- Add a row to [DATA-LICENSES.md](DATA-LICENSES.md) with the dataset's
  source URL, original license (SPDX identifier where available), and
  citation requirement.
- The adapter must download from the dataset's official source — do not
  commit dataset bytes to this repository.
- Surface citation information to end users (e.g., a `citation` property on
  the adapter class).

### SPDX License Identifiers

New source files should include an SPDX header on the first or second line:

```python
# SPDX-License-Identifier: Apache-2.0
```

For documentation files added under `docs/`, use:

```markdown
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->
```

### Security Issues

Do not file public issues or PRs for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for the private disclosure process.
