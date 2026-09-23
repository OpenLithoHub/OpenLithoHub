"""PR-C execution planner: typed reasons, fail-honest ``auto``, policy gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from openlithohub.models.base import LithographyModel
from openlithohub.models.registry import register_builtin_models, registry
from openlithohub.workflow.execution import (
    DENSE_REQUESTED_EXPLICITLY,
    DENSE_SMALL_LAYOUT,
    MAX_DENSE_BYTES_ENV,
    STREAMING_REQUIRED_BY_MEMORY_POLICY,
    STREAMING_SUPPORTED,
    STREAMING_UNSUPPORTED_INPUT,
    STREAMING_UNSUPPORTED_MODEL,
    STREAMING_UNSUPPORTED_OUTPUT,
    InputProbe,
    OptimizeRequest,
    StreamingUnsupportedError,
    coerce_execution_mode,
    dense_memory_budget,
    klayout_available,
    plan_execution,
    probe_input,
    streaming_capability_matrix,
)


@pytest.fixture(autouse=True)
def _models() -> None:
    register_builtin_models()


@pytest.fixture
def identity() -> LithographyModel:
    return registry.get("dummy-identity")


@pytest.fixture
def npy_probe(tmp_path: Path) -> InputProbe:
    arr = np.zeros((64, 64), dtype=np.float32)
    path = tmp_path / "in.npy"
    np.save(path, arr)
    req = OptimizeRequest(
        model=registry.get("dummy-identity"),
        pixel_size_nm=1.0,
        input_path=path,
        output_path=tmp_path / "out.npy",
        output_kind="raster-npy",
    )
    return probe_input(req)


# ---- env budget ----------------------------------------------------------


def test_budget_default_is_8gib() -> None:
    assert dense_memory_budget({}) == 8 * 1024**3


def test_budget_env_override_and_unlimited() -> None:
    assert dense_memory_budget({MAX_DENSE_BYTES_ENV: "1024"}) == 1024
    assert dense_memory_budget({MAX_DENSE_BYTES_ENV: "0"}) == 0


@pytest.mark.parametrize("raw", ["nope", "-4", "1.5"])
def test_budget_env_invalid_fails(raw: str) -> None:
    with pytest.raises(ValueError):
        dense_memory_budget({MAX_DENSE_BYTES_ENV: raw})


def test_coerce_execution_mode() -> None:
    assert coerce_execution_mode("auto") == "auto"
    assert coerce_execution_mode("dense") == "dense"
    assert coerce_execution_mode("streaming") == "streaming"
    with pytest.raises(ValueError, match="execution_mode"):
        coerce_execution_mode("fast")


# ---- auto: fail-honest ----------------------------------------------------


def test_auto_under_policy_is_dense(npy_probe: InputProbe, identity: LithographyModel) -> None:
    plan = plan_execution(
        input_probe=npy_probe,
        output_kind="raster-npy",
        writer="vsb",
        model=identity,
        max_dense_bytes=npy_probe.estimated_dense_bytes,
        requested_mode="auto",
    )
    assert plan.mode == "dense"
    assert plan.reason == DENSE_SMALL_LAYOUT
    assert not plan.over_memory_policy


def test_auto_over_policy_supported_streams(
    npy_probe: InputProbe, identity: LithographyModel
) -> None:
    plan = plan_execution(
        input_probe=npy_probe,
        output_kind="raster-npy",
        writer="vsb",
        model=identity,
        max_dense_bytes=npy_probe.estimated_dense_bytes - 1,
        requested_mode="auto",
    )
    assert plan.mode == "streaming"
    assert plan.reason == STREAMING_REQUIRED_BY_MEMORY_POLICY
    assert plan.over_memory_policy


def test_auto_over_policy_unsupported_input_fails_closed(
    tmp_path: Path, identity: LithographyModel
) -> None:
    torch.save(torch.zeros((64, 64)), str(tmp_path / "in.pt"))
    req = OptimizeRequest(
        model=identity,
        pixel_size_nm=1.0,
        input_path=tmp_path / "in.pt",
        output_path=tmp_path / "out.oas",
        output_kind="oasis",
        writer="vsb",
    )
    probe = probe_input(req)
    assert probe.materialized_for_probe, ".pt probing inherently materialises"
    with pytest.raises(StreamingUnsupportedError) as exc:
        plan_execution(
            input_probe=probe,
            output_kind="oasis",
            writer="vsb",
            model=identity,
            max_dense_bytes=probe.estimated_dense_bytes - 1,
            requested_mode="auto",
        )
    assert exc.value.reason == STREAMING_UNSUPPORTED_INPUT


def test_auto_over_policy_unsupported_output_fails_closed(
    npy_probe: InputProbe, identity: LithographyModel
) -> None:
    with pytest.raises(StreamingUnsupportedError) as exc:
        plan_execution(
            input_probe=npy_probe,
            output_kind="oasis",
            writer="mbmw",
            model=identity,
            max_dense_bytes=npy_probe.estimated_dense_bytes - 1,
            requested_mode="auto",
        )
    assert exc.value.reason == STREAMING_UNSUPPORTED_OUTPUT


def test_auto_over_policy_unsupported_model_fails_closed(
    npy_probe: InputProbe, identity: LithographyModel
) -> None:
    class DenseOnlyModel(LithographyModel):
        NAME = "execution-dense-only"
        SUPPORTS_STREAMING = False

        def predict(self, design: torch.Tensor, **kwargs: object) -> object:
            raise AssertionError("predict must not be reached by the planner")

    with pytest.raises(StreamingUnsupportedError) as exc:
        plan_execution(
            input_probe=npy_probe,
            output_kind="raster-npy",
            writer="vsb",
            model=DenseOnlyModel(),
            max_dense_bytes=npy_probe.estimated_dense_bytes - 1,
            requested_mode="auto",
        )
    assert exc.value.reason == STREAMING_UNSUPPORTED_MODEL


# ---- explicit modes --------------------------------------------------------


def test_explicit_streaming_supported(npy_probe: InputProbe, identity: LithographyModel) -> None:
    plan = plan_execution(
        input_probe=npy_probe,
        output_kind="raster-npy",
        writer="vsb",
        model=identity,
        max_dense_bytes=8 * 1024**3,
        requested_mode="streaming",
    )
    assert plan.mode == "streaming"
    assert plan.reason == STREAMING_SUPPORTED


def test_explicit_streaming_curvilinear_fails(
    npy_probe: InputProbe, identity: LithographyModel
) -> None:
    with pytest.raises(StreamingUnsupportedError) as exc:
        plan_execution(
            input_probe=npy_probe,
            output_kind="oasis",
            writer="mbmw",
            model=identity,
            max_dense_bytes=8 * 1024**3,
            requested_mode="streaming",
        )
    assert exc.value.reason == STREAMING_UNSUPPORTED_OUTPUT


def test_explicit_dense_over_policy_is_documented_override(
    npy_probe: InputProbe, identity: LithographyModel
) -> None:
    plan = plan_execution(
        input_probe=npy_probe,
        output_kind="oasis",
        writer="mbmw",
        model=identity,
        max_dense_bytes=npy_probe.estimated_dense_bytes - 1,
        requested_mode="dense",
    )
    assert plan.mode == "dense"
    assert plan.reason == DENSE_REQUESTED_EXPLICITLY
    assert "explicit" in plan.detail


def test_explicit_streaming_in_memory_endpoints_allowed_with_caveat(
    identity: LithographyModel,
) -> None:
    tensor = torch.zeros((32, 32))
    probe = probe_input(
        OptimizeRequest(
            model=identity,
            pixel_size_nm=1.0,
            input_tensor=tensor,
            output_kind="tensor-mask",
        )
    )
    assert not probe.streaming_supported
    assert probe.explicit_only_ok
    plan = plan_execution(
        input_probe=probe,
        output_kind="tensor-mask",
        writer="vsb",
        model=identity,
        max_dense_bytes=8 * 1024**3,
        requested_mode="streaming",
    )
    assert plan.mode == "streaming"
    assert "in-memory" in plan.detail
    # ...but auto above the policy still fails closed on the same request.
    with pytest.raises(StreamingUnsupportedError) as exc:
        plan_execution(
            input_probe=probe,
            output_kind="tensor-mask",
            writer="vsb",
            model=identity,
            max_dense_bytes=probe.estimated_dense_bytes - 1,
            requested_mode="auto",
        )
    assert exc.value.reason in (STREAMING_UNSUPPORTED_INPUT, STREAMING_UNSUPPORTED_OUTPUT)


# ---- probes / capability table consistency ---------------------------------


def test_probe_pt_is_unsupported_but_inspectable(tmp_path: Path) -> None:
    arr = torch.zeros((24, 30))
    arr[4:10, 6:20] = 1.0
    path = tmp_path / "in.pt"
    torch.save(arr, str(path))
    req = OptimizeRequest(
        model=registry.get("dummy-identity"),
        pixel_size_nm=1.0,
        input_path=path,
        output_path=tmp_path / "out.oas",
        output_kind="oasis",
    )
    probe = probe_input(req)
    assert probe.kind == "pt"
    assert probe.shape == (24, 30)
    assert not probe.streaming_supported
    assert probe.materialized_for_probe
    assert probe.unsupported_reason


def test_capability_matrix_matches_probe(tmp_path: Path) -> None:
    """The /v1/capabilities table is code-owned; cross-check it against the
    real probe behaviour so the endpoint can never drift into fiction."""
    matrix = streaming_capability_matrix()
    for fmt, claimed in matrix["inputs"].items():
        path = tmp_path / f"in.{fmt}"
        if fmt == "npy":
            np.save(path, np.zeros((4, 4), dtype=np.float32))
        elif fmt == "pt":
            torch.save(torch.zeros((4, 4)), str(path))
        else:
            if not klayout_available():
                pytest.skip("klayout required for gds/oas probe")
            import klayout.db as db

            ly = db.Layout()
            ly.dbu = 0.001
            cell = ly.create_cell("TOP")
            cell.shapes(ly.layer(1, 0)).insert(db.Box(0, 0, 8000, 8000))
            ly.write(str(path))
        req = OptimizeRequest(
            model=registry.get("dummy-identity"),
            pixel_size_nm=8.0,
            input_path=path,
            output_path=tmp_path / "out.npy",
            output_kind="raster-npy",
        )
        probe = probe_input(req)
        assert probe.streaming_supported == claimed, f"capability drift for {fmt}"
    assert matrix["outputs"]["curvilinear_oasis"] is False
    assert matrix["outputs"]["manhattan_oasis"] == klayout_available()
    assert matrix["engine"] is True and matrix["product_execution"] is True
    assert matrix["memory_policy"]["fail_closed"] is True
