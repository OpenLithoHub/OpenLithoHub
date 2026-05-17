"""Performance benchmarks for critical paths.

Run with: pytest tests/benchmarks/ --benchmark-only
Skip in normal CI with: pytest -m 'not benchmark'
"""

import pytest
import torch

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub.benchmark.metrics.epe import compute_epe
from openlithohub.workflow.tiling import tile_layout


@pytest.fixture
def mask_64() -> torch.Tensor:
    m = torch.zeros(64, 64)
    m[16:48, 16:48] = 1.0
    return m


@pytest.fixture
def mask_256() -> torch.Tensor:
    m = torch.zeros(256, 256)
    m[64:192, 64:192] = 1.0
    return m


@pytest.mark.benchmark
def test_bench_epe_64(benchmark, mask_64: torch.Tensor) -> None:  # type: ignore[no-untyped-def]
    target = mask_64.clone()
    target[14:50, 14:50] = 1.0
    benchmark(compute_epe, mask_64, target, pixel_size_nm=1.0)


@pytest.mark.benchmark
def test_bench_epe_256(benchmark, mask_256: torch.Tensor) -> None:  # type: ignore[no-untyped-def]
    target = mask_256.clone()
    target[60:196, 60:196] = 1.0
    benchmark(compute_epe, mask_256, target, pixel_size_nm=1.0)


@pytest.mark.benchmark
def test_bench_forward_model_256(benchmark, mask_256: torch.Tensor) -> None:  # type: ignore[no-untyped-def]
    benchmark(simulate_aerial_image, mask_256, sigma_px=2.0)


@pytest.mark.benchmark
def test_bench_tiling_1024(benchmark) -> None:  # type: ignore[no-untyped-def]
    layout = torch.rand(1024, 1024)
    benchmark(tile_layout, layout, tile_size=256, overlap=32)


@pytest.mark.benchmark
def test_bench_levelset_ilt_10iter(benchmark, mask_64: torch.Tensor) -> None:  # type: ignore[no-untyped-def]
    from openlithohub.models.levelset_ilt import LevelSetILTModel

    model = LevelSetILTModel(iterations=10, lr=0.1, sigma_px=1.5)
    design = mask_64
    benchmark(model.predict, design)
