# Contributing to OpenLithoHub

Thank you for your interest in contributing! This guide will help you get started.

## Project Scope & Responsibility Boundary

OpenLithoHub 生态由两个独立仓库组成，职责清晰分离：

| 仓库 | 职责 | 部署 |
|------|------|------|
| **本仓库 (OpenLithoHub)** | Python SDK、指标计算、模型接口、工作流引擎、CLI、技术文档、HF Spaces Playground | docs.openlithohub.com |
| **openlithohub-website** | 品牌官网、排行榜前端展示、Blog、Community、Playground 嵌入 | openlithohub.com |

**边界原则：**

- 本仓库负责**数据生产**（metrics, leaderboard export, model inference）
- 网站仓库负责**数据展示**（读取本仓库导出的 JSON 渲染前端）
- 技术文档（API reference, Getting Started, Architecture）放在本仓库 `docs/`
- 品牌/营销内容（Features 介绍页、Blog）放在网站仓库
- 排行榜数据通过 `openlithohub leaderboard export` 导出 JSON，手动更新到网站仓库 `src/data/leaderboard.json`
- 网站 features 页面的架构描述必须与本仓库 `docs/architecture.md` 保持一致

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
├── cli/          # Command-line interface (Typer)
├── data/         # Layer 1: Dataset adapters
├── benchmark/    # Layer 2: Metrics and compliance checks
├── models/       # Layer 3: Model integration interface
├── workflow/     # Layer 4: OASIS workflow engine
├── leaderboard/  # Layer 5: SOTA tracking and data engine
└── _utils/       # Shared internal utilities
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=openlithohub --cov-report=html

# Run a specific test file
pytest tests/test_models/test_interface.py
```

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
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> LithoSample:
        ...

    def download(self, root: str) -> None:
        ...
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
