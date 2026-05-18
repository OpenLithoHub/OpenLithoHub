# Contributing

Thank you for your interest in contributing to OpenLithoHub!

## Development Setup

```bash
git clone https://github.com/OpenLithoHub/OpenLithoHub.git
cd OpenLithoHub
pip install -e ".[dev]"
pre-commit install
```

## Code Style

- **Linter/Formatter:** [Ruff](https://docs.astral.sh/ruff/)
- **Type checker:** [mypy](https://mypy-lang.org/) (strict mode)
- **Line length:** 100 characters
- **Target Python:** 3.10+

Run checks locally:

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

## Testing

```bash
pytest tests/ -v
```

Tests are organized by module:

```
tests/
├── benchmarks/        # pytest-benchmark performance suite
├── test_benchmark/    # Metrics and compliance
├── test_cli/          # CLI commands
├── test_data/         # Dataset adapters + dummy generator
├── test_leaderboard/  # Leaderboard logic
├── test_models/       # Model interface
├── test_utils/        # Hopkins, resist, morphology
├── test_vis/          # Paper-ready visualization
└── test_workflow/     # Workflow pipeline + EDA bridge
```

## Adding a New Metric

1. Create `src/openlithohub/benchmark/metrics/your_metric.py`
2. Implement the computation function
3. Export from `benchmark/metrics/__init__.py`
4. Add tests in `tests/test_benchmark/`
5. Document in `docs/api/benchmark.md`

## Adding a New Model

1. Subclass `LithographyModel` in `src/openlithohub/models/`
2. Use the `@registry.register` decorator
3. Add tests in `tests/test_models/`

## Pull Request Guidelines

- One logical change per PR
- All tests must pass (`pytest tests/ -v`)
- Lint and format must pass (`ruff check && ruff format --check`)
- Type check must pass (`mypy src/`)
- Include tests for new functionality
- Update documentation if adding public APIs

## Reporting Issues

Please use [GitHub Issues](https://github.com/OpenLithoHub/OpenLithoHub/issues) with:

- Clear title describing the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
