"""PR-G hostile device-policy and batching tests (roadmap §24).

G2-A/B/C/D are exercised on a CUDA-less host — which is exactly the
honest environment for the fail-closed paths. G3 batching equivalence,
boundedness, per-tile ownership, and OOM no-fallback run on CPU by
design: batching is device-agnostic scheduling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openlithohub.models.base import LithographyModel
from openlithohub.models.registry import register_builtin_models, registry
from openlithohub.server.config import ServerConfig, resolve_execution_device
from openlithohub.streaming import run_streaming
from openlithohub.streaming.sinks import TensorTileSink
from openlithohub.streaming.vector_runs import (
    ExactVectorRunSource,
    VectorCell,
    rectangle,
)
from openlithohub.workflow.execution import OptimizeRequest, optimize_layout

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _models() -> None:
    register_builtin_models()


# ---- G2-A/B: explicit CUDA fails closed; auto is deterministic -------------


def test_g2a_explicit_cuda_on_cudaless_host_fails_honestly() -> None:
    if torch.cuda.is_available():  # pragma: no cover — GPU CI machines
        pytest.skip("host has CUDA; the fail-closed path needs a CUDA-less host")
    with pytest.raises(ValueError, match="refusing to fall back to CPU"):
        resolve_execution_device("cuda", cuda_available=False, device_count=0)
    with pytest.raises(ValueError, match="refusing to fall back to CPU"):
        resolve_execution_device("cuda:0", cuda_available=False, device_count=0)


def test_g2b_auto_policy_is_deterministic_and_documented() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    for _ in range(5):
        assert (
            resolve_execution_device(
                "auto",
                cuda_available=torch.cuda.is_available(),
                device_count=torch.cuda.device_count(),
            )
            == expected
        )


def test_g2a_runner_explicit_cuda_policy_fails_closed(tmp_path: Path) -> None:
    """The product runner fails closed under an explicit CUDA policy on a
    CUDA-less host — before any model load or layout work."""
    from openlithohub.server.app import _run_optimize

    layout = tmp_path / "in.npy"
    np.save(layout, np.zeros((16, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="refusing to fall back to CPU"):
        _run_optimize(
            input_path=layout,
            output_path=tmp_path / "out.oas",
            model_name="dummy-identity",
            node="3nm-euv",
            pixel_nm=1.0,
            tile_size=16,
            writer="vsb",
            layer=None,
            pretrained=False,
            device_policy="cuda",
        )


# ---- G2-C: device enters resident model-cache identity ----------------------


def test_g2c_device_enters_model_cache_identity() -> None:
    import openlithohub.server.app as app_mod

    key_a = app_mod._model_cache_key("dummy-identity", {}, "cpu")
    key_b = app_mod._model_cache_key("dummy-identity", {}, "cuda:0")
    assert key_a != key_b, "cached models on different devices must not collide"

    saved_cache = dict(app_mod._MODEL_CACHE)
    saved_locks = dict(app_mod._MODEL_LOCKS)
    saved_refs = dict(app_mod._MODEL_REFCOUNTS)
    try:
        model_a, lock_a, key_a = app_mod._get_or_load_model("dummy-identity", {}, "cpu")
        model_b, lock_b, key_b = app_mod._get_or_load_model("dummy-identity", {}, "cuda:0")
        assert key_a != key_b
        # Separate identities = separate resident models (never one cached
        # model moved CPU<->GPU per request).
        assert model_a is not model_b
        assert key_a in app_mod._MODEL_CACHE and key_b in app_mod._MODEL_CACHE
    finally:
        app_mod._MODEL_CACHE.clear()
        app_mod._MODEL_CACHE.update(saved_cache)
        app_mod._MODEL_LOCKS.clear()
        app_mod._MODEL_LOCKS.update(saved_locks)
        app_mod._MODEL_REFCOUNTS.clear()
        app_mod._MODEL_REFCOUNTS.update(saved_refs)


# ---- G2-D: CPU execution is never labeled GPU-executed -----------------------


def test_g2d_capabilities_report_cpu_execution_honestly(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from openlithohub.server import create_app

    if torch.cuda.is_available():  # pragma: no cover
        pytest.skip("host has CUDA; the CPU-honesty labeling needs a CUDA-less host")
    app = create_app(ServerConfig(device="auto"))
    with TestClient(app) as client:
        gpu = client.get("/v1/capabilities").json()["gpu"]
    assert gpu["available"] is False
    assert gpu["selected_device"] == "cpu"
    assert gpu["product_execution_enabled"] is True  # CPU execution is real execution


def test_g2d2_unsatisfiable_policy_reports_disabled(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from openlithohub.server import create_app

    if torch.cuda.is_available():  # pragma: no cover
        pytest.skip("host has CUDA")
    app = create_app(
        ServerConfig(device="cuda:0", state_dir=str(tmp_path / "s"), job_backend="sqlite")
    )
    with TestClient(app) as client:
        gpu = client.get("/v1/capabilities").json()["gpu"]
    assert gpu["product_execution_enabled"] is False
    assert gpu["selected_device"] == "unavailable"


# ---- G2/E3: batching capability gate ------------------------------------------


class BatchAwareIdentity(LithographyModel):
    """Identity model whose predict accepts BxCxHxW and returns the same."""

    NAME = "prg-batch-identity"
    SUPPORTED_DEVICES = ("cpu", "cuda")
    SUPPORTS_BATCHED_PREDICT = True

    def predict(self, design: torch.Tensor, **kwargs: object):
        from openlithohub.models.base import PredictionResult

        return PredictionResult(mask=design.clone())


class BatchRefuser(LithographyModel):
    NAME = "prg-batch-refuser"

    def predict(self, design: torch.Tensor, **kwargs: object):
        from openlithohub.models.base import PredictionResult

        return PredictionResult(mask=design.clone())


@pytest.fixture(autouse=True)
def _register_prg_models() -> None:

    registry.register(BatchAwareIdentity)
    registry.register(BatchRefuser)


def _batch_req(model: LithographyModel, tensor: torch.Tensor, batch: int) -> OptimizeRequest:
    return OptimizeRequest(
        model=model,
        pixel_size_nm=1.0,
        input_tensor=tensor,
        output_kind="tensor-mask",
        tile_size=16,
        threshold=0.5,
        execution_mode="dense",
        gpu_batch_tiles=batch,
    )


def test_batch_request_without_capability_fails_closed(tmp_path: Path) -> None:
    """Batching lives on the streaming branch; a request for batch>1 on a
    model without the capability fails closed before any work."""
    layout = tmp_path / "in.npy"
    np.save(layout, (np.random.default_rng(1).random((48, 64)) > 0.5).astype(np.float32))
    req = OptimizeRequest(
        model=BatchRefuser(),
        pixel_size_nm=1.0,
        input_path=layout,
        output_path=tmp_path / "out.npy",
        output_kind="raster-npy",
        tile_size=16,
        threshold=0.5,
        execution_mode="streaming",
        gpu_batch_tiles=4,
    )
    with pytest.raises(ValueError, match="SUPPORTS_BATCHED_PREDICT"):
        optimize_layout(req)


def test_g2h_batch1_matches_plain_execution(tmp_path: Path) -> None:
    from openlithohub.api.engine import LitheEngine

    rng = np.random.default_rng(5)
    design = torch.from_numpy((rng.random((64, 64)) > 0.5).astype(np.float32))
    with LitheEngine(model="dummy-identity", tile_size=16) as engine:
        reference = engine.optimize(design, execution_mode="dense")
    result = optimize_layout(_batch_req(BatchAwareIdentity(), design, batch=1))
    assert torch.equal(reference.tensor, result.mask)


def test_g2i_batch_n_matches_n_independent_tiles(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    layout = tmp_path / "in.npy"
    tensor = (rng.random((64, 80)) > 0.5).astype(np.float32)
    np.save(layout, tensor)

    def run(batch: int) -> tuple[np.ndarray, int, int]:
        req = OptimizeRequest(
            model=BatchAwareIdentity(),
            pixel_size_nm=1.0,
            input_path=layout,
            output_path=tmp_path / f"out-{batch}.npy",
            output_kind="raster-npy",
            tile_size=16,
            threshold=0.5,
            execution_mode="streaming",
            gpu_batch_tiles=batch,
        )
        result = optimize_layout(req)
        assert result.plan.mode == "streaming"
        out = np.load(str(result.output_path), mmap_mode="r")
        return np.asarray(out), result.batch_size, result.forward_batches

    sequential, seq_batch, seq_batches = run(1)
    batched, got_batch, got_batches = run(4)
    assert got_batch == 4
    assert np.array_equal(sequential, batched), "batched output must match sequential"
    assert np.array_equal(batched, tensor), "identity model returns the input"
    # 4x5 core grid at batch=1: exactly one forward per tile. Batch=4
    # strictly reduces forward invocations.
    assert seq_batches == 20
    assert 0 < got_batches <= 5, (got_batches, seq_batches)


def test_g2j_verification_contexts_forbid_batching() -> None:
    """Verifier re-entrancy would reorder tile execution; the pipeline
    must refuse batching with verifiers attached (roadmap §10)."""
    from openlithohub.streaming.verification import VerificationContext

    class NoopVerifier:
        name = "noop"

        def required_halo(self, ctx: VerificationContext):
            return None

        def prepare(self, ctx: VerificationContext) -> None:
            return None

        def verify_tile(self, ctx):
            from openlithohub.streaming.verification import TileVerificationResult

            return TileVerificationResult(status="PASS")

    source = ExactVectorRunSource(
        shape=(32, 32),
        cells={"TOP": VectorCell("TOP", polygons=(rectangle("r", 0, 0, 32, 32),))},
        top="TOP",
    )
    sink = TensorTileSink((32, 32))
    with pytest.raises(ValueError, match="no verification plugins"):
        run_streaming(
            source,
            sink,
            lambda t: t,
            core_size=16,
            batch_size=4,
            batched_forward_fn=lambda b: b,
            verifiers=[NoopVerifier()],
        )


def test_g2k_batching_stays_bounded() -> None:
    """The scheduler may never accumulate more than `batch_size` windows,
    independent of the total tile count."""
    seen_sizes: list[int] = []

    def counting_batched(batch: torch.Tensor) -> torch.Tensor:
        seen_sizes.append(int(batch.shape[0]))
        return batch

    side = 160
    source = ExactVectorRunSource(
        shape=(side, side),
        cells={"TOP": VectorCell("TOP", polygons=(rectangle("fill", 0, 0, side - 1, side - 1),))},
        top="TOP",
    )
    sink = TensorTileSink((side, side))
    report = run_streaming(
        source,
        sink,
        lambda t: t,
        core_size=16,
        halo_policy=None,
        batch_size=4,
        batched_forward_fn=counting_batched,
    )
    assert report.n_tiles == (side // 16) ** 2
    assert seen_sizes, "batched forward never invoked"
    assert max(seen_sizes) <= 4
    assert sum(seen_sizes) == report.n_tiles


def test_g2l_boundary_tiles_flush_without_padding() -> None:
    """Boundary tiles (different read shape) must never be padded into a
    batch — each flush is exact."""
    shapes_seen: list[tuple[int, ...]] = []

    def shape_echo(batch: torch.Tensor) -> torch.Tensor:
        shapes_seen.append(tuple(batch.shape))
        return batch

    side = 40  # 16px cores -> interior 16x16 windows + 8x16/16x8/8x8 boundary
    source = ExactVectorRunSource(
        shape=(side, side),
        cells={"TOP": VectorCell("TOP", polygons=(rectangle("f", 0, 0, side - 1, side - 1),))},
        top="TOP",
    )
    sink = TensorTileSink((side, side))
    run_streaming(
        source,
        sink,
        lambda t: t,
        core_size=16,
        batch_size=8,
        batched_forward_fn=shape_echo,
    )
    assert shapes_seen, "no batches ran"
    assert max(b[0] for b in shapes_seen) <= 8
    total_windows = sum(b[0] for b in shapes_seen)
    assert total_windows == ((40 + 15) // 16) ** 2


def test_g2_oom_never_silently_falls_back_to_cpu(tmp_path: Path) -> None:
    """A CUDA OOM inside the batched forward must propagate — the
    executor has no CPU-fallback path (roadmap §11)."""
    calls: list[int] = []

    class OomModel(BatchAwareIdentity):
        NAME = "prg-oom-model"

        def predict(self, design: torch.Tensor, **kwargs: object):
            calls.append(1)
            raise torch.cuda.OutOfMemoryError()

    registry.register(OomModel)
    tensor = (torch.rand(32, 32) > 0.5).float()
    with pytest.raises(torch.cuda.OutOfMemoryError):  # noqa: B017 — attribute exception type
        optimize_layout(_batch_req(OomModel(), tensor, batch=4))
    assert calls, "model was never invoked"
