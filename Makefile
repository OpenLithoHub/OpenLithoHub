.PHONY: check-plugins flagship flagship-ci reproduce test lint

check-plugins:
	python3 -c "from openlithohub.plugins import list_plugins; status = list_plugins(); print('Plugin status:', status); assert isinstance(status, dict); print('OK: plugin infrastructure healthy')"

flagship:
	python3 benchmarks/benchmark_multiproc.py

flagship-ci:
	python3 benchmarks/benchmark_multiproc.py

reproduce: ## One-key reproducibility (fixed seed)
	PYTHONHASHSEED=42 python3 benchmarks/benchmark_multiproc.py

test:
	python3 -m pytest tests/ -v --tb=short

lint:
	ruff check --fix src/ tests/
	ruff format src/ tests/
