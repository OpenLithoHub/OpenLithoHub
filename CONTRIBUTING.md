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
