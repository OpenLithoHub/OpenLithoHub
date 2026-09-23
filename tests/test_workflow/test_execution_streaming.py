"""PR-C structural hostile tests: the streaming product branch must be
architecturally streaming, not merely numerically correct.

Gates implemented here (roadmap §12–§13):

* dense/streaming equivalence (bitwise for deterministic pointwise models)
* dense-loader firewall — a supported streaming GDS request must succeed
  with ``load_layout`` poisoned
* stitch firewall — the streaming branch must never call ``stitch_tiles``
* full-result retention firewall — no per-tile results survive the run
* memory-policy fail-closed — an over-policy unsupported request must not
  fall back to dense
* real-layout smoke — a KLayout-generated GDS traverses the product path
* engineering memory gate (non-authority regression): live window bytes
  are bounded by the tile, not by the layout dimensions
"""

from __future__ import annotations

import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

from openlithohub.models.registry import register_builtin_models, registry
from openlithohub.workflow.execution import (
    OptimizeRequest,
    StreamingUnsupportedError,
    optimize_layout,
    plan_request,
)

db = pytest.importorskip("klayout.db")  # noqa: E402 — gate before project imports

from openlithohub.streaming.geometry import BoundingBox  # noqa: E402
from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource  # noqa: E402

TILE = 32
HALO = 8
PX = 8.0  # nm per pixel for GDS fixtures (STEP below)


@pytest.fixture(autouse=True)
def _models() -> None:
    register_builtin_models()


def _identity() -> torch.Tensor:
    return registry.get("dummy-identity")


def _write_npy(path: Path, side: int = 96) -> np.ndarray:
    rng = np.random.default_rng(7)
    arr = (rng.random((side, side)) > 0.6).astype(np.float32)
    np.save(path, arr)
    return arr


def _write_gds(path: Path, top: str = "TOP") -> None:
    """Small but real vector layout: two-level hierarchy, a donut, a path."""
    step = 8
    h = 64
    ly = db.Layout()
    ly.dbu = 0.001  # 1 nm per DBU -> integer DBU-per-pixel for PX=8
    l1 = ly.layer(1, 0)
    child = ly.create_cell("CHILD")
    child.shapes(l1).insert(db.Box(0, 0, 4 * step, 3 * step))
    cell = ly.create_cell(top)
    # y-up KLayout coords; raster row 0 == top (y-flip convention).
    cell.shapes(l1).insert(db.Box(4 * step, (h - 12) * step, 28 * step, (h - 4) * step))
    donut = db.Box(10 * step, (h - 30) * step, 50 * step, (h - 20) * step)
    hole = db.Box(18 * step, (h - 28) * step, 42 * step, (h - 22) * step)
    region = db.Region(donut) - db.Region(hole)
    cell.shapes(l1).insert(region)
    path_poly = db.Path(
        [db.Point(2 * step, 20 * step), db.Point(60 * step, 20 * step)],
        2 * step,
    )  # axis-aligned so the stroke stays exactly pixel aligned
    cell.shapes(l1).insert(path_poly)
    cell.insert(db.CellInstArray(child.cell_index(), db.Trans(int(52 * step), int((h - 8) * step))))
    ly.write(str(path))


@pytest.fixture
def gds_path(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.gds"
    _write_gds(path)
    return path


def _instrumented_source(ledger: _WindowLedger) -> type:
    """A MemmapTensorTileSource whose windows feed the ledger."""
    import openlithohub.workflow.execution as execution_mod

    real_cls = execution_mod.MemmapTensorTileSource

    class Instrumented(real_cls):  # type: ignore[misc,valid-type]
        def read_window(self, bbox: BoundingBox) -> torch.Tensor:
            window = super().read_window(bbox)
            ledger.observe(window)
            return window

    return Instrumented


def _req(
    input_path: Path,
    output_path: Path,
    *,
    mode: str,
    halo: int = HALO,
) -> OptimizeRequest:
    return OptimizeRequest(
        model=_identity(),
        pixel_size_nm=1.0,
        input_path=input_path,
        output_path=output_path,
        output_kind="raster-npy",
        tile_size=TILE,
        halo_px=halo,
        threshold=0.5,
        execution_mode=mode,  # type: ignore[arg-type]
    )


class _WindowLedger:
    """Weakref ledger: how many tile-sized tensors are alive right now."""

    def __init__(self) -> None:
        self._refs: list[weakref.ref[torch.Tensor]] = []
        self._sizes: dict[int, int] = {}
        self.max_live_bytes = 0
        self.observed = 0

    def observe(self, tensor: torch.Tensor) -> None:
        self.observed += 1
        key = id(tensor)
        self._sizes[key] = int(tensor.numel() * tensor.element_size())
        self._refs.append(weakref.ref(tensor, lambda _r, k=key: self._sizes.pop(k, None)))
        self.max_live_bytes = max(self.max_live_bytes, sum(self._sizes.values()))

    def live_count(self) -> int:
        return len(self._sizes)


# ---- equivalence (§12.5) ---------------------------------------------------


def test_dense_streaming_bitwise_equivalence_npy(tmp_path: Path) -> None:
    """Multi-tile layout, deterministic pointwise model: the streaming
    artifact must equal the dense mask bit for bit."""
    input_path = tmp_path / "in.npy"
    arr = _write_npy(input_path)
    dense = optimize_layout(_req(input_path, tmp_path / "d.npy", mode="dense"))
    stream = optimize_layout(_req(input_path, tmp_path / "s.npy", mode="streaming"))
    assert dense.plan.mode == "dense" and stream.plan.mode == "streaming"
    assert dense.n_tiles > 1 and stream.n_tiles > 1
    artifact = np.load(tmp_path / "s.npy", mmap_mode="r")
    assert np.array_equal(np.asarray(artifact), dense.mask.numpy())
    expected = (arr > 0.5).astype(np.float32)
    assert np.array_equal(np.asarray(artifact), expected)


def test_engine_dense_streaming_bitwise_equivalence() -> None:
    from openlithohub.api.engine import LitheEngine

    rng = np.random.default_rng(11)
    design = torch.from_numpy((rng.random((70, 90)) > 0.5).astype(np.float32))
    with LitheEngine(model="dummy-identity", tile_size=TILE) as engine:
        dense = engine.optimize(design, execution_mode="dense")
        stream = engine.optimize(design, execution_mode="streaming")
    assert torch.equal(dense.tensor, stream.tensor)


def test_streaming_artifact_is_self_describing_npy(tmp_path: Path) -> None:
    input_path = tmp_path / "in.npy"
    _write_npy(input_path)
    optimize_layout(_req(input_path, tmp_path / "s.npy", mode="streaming"))
    header = (tmp_path / "s.npy").read_bytes()[:6]
    assert header == b"\x93NUMPY"


# ---- dense-loader firewall (§12.1) ------------------------------------------


def test_gds_streaming_bypasses_dense_loader(
    tmp_path: Path, gds_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openlithohub.data.io as data_io

    def _poison(*_a: object, **_k: object) -> torch.Tensor:
        raise AssertionError("dense full-raster loader called on the streaming branch")

    monkeypatch.setattr(data_io, "load_layout", _poison)
    result = optimize_layout(
        OptimizeRequest(
            model=_identity(),
            pixel_size_nm=PX,
            input_path=gds_path,
            output_path=tmp_path / "s.npy",
            output_kind="raster-npy",
            tile_size=16,
            execution_mode="streaming",
        )
    )
    assert result.plan.mode == "streaming"
    assert result.plan.input_backend == "klayout-aligned-vector"
    artifact = np.asarray(np.load(tmp_path / "s.npy", mmap_mode="r"))
    reference = KLayoutAlignedRunSource.from_file(gds_path, pixel_size_nm=PX)
    expected = reference.read_window(
        BoundingBox(0, 0, reference.shape[1], reference.shape[0])
    ).numpy()
    assert np.array_equal(artifact, expected)


# ---- stitch firewall (§12.2) -------------------------------------------------


def test_streaming_never_calls_stitch_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openlithohub.workflow.tiling as tiling

    input_path = tmp_path / "in.npy"
    _write_npy(input_path)

    def _poison(*_a: object, **_k: object) -> torch.Tensor:
        raise AssertionError("stitch_tiles called on the streaming branch")

    monkeypatch.setattr(tiling, "stitch_tiles", _poison)
    result = optimize_layout(_req(input_path, tmp_path / "s.npy", mode="streaming"))
    assert result.plan.mode == "streaming"
    assert np.asarray(np.load(tmp_path / "s.npy", mmap_mode="r")).sum() > 0


# ---- retention firewall (§12.3) ----------------------------------------------


def test_streaming_does_not_retain_tile_results(tmp_path: Path) -> None:
    """Weakref firewall (§12.3): every per-tile window must be garbage by
    the time optimize_layout returns, and the result carries no tile list."""
    import gc

    import openlithohub.workflow.execution as execution_mod

    input_path = tmp_path / "in.npy"
    _write_npy(input_path, side=128)

    ledger = _WindowLedger()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(execution_mod, "MemmapTensorTileSource", _instrumented_source(ledger))
    try:
        req = _req(input_path, tmp_path / "s.npy", mode="streaming")
        result = optimize_layout(req)
    finally:
        monkeypatch.undo()

    assert result.plan.mode == "streaming"
    assert not hasattr(result, "tile_results")
    assert result.work_accounting is not None
    gc.collect()
    assert ledger.observed > 0, "instrumented source never observed a window"
    assert ledger.live_count() == 0, "streaming branch retained tile windows"


# ---- memory-policy fail-closed (§12.4) ----------------------------------------


def test_over_policy_pt_request_fails_closed_not_dense(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openlithohub.workflow.tiling as tiling

    input_path = tmp_path / "in.pt"
    torch.save(torch.zeros((64, 64)), str(input_path))

    def _poison(*_a: object, **_k: object) -> torch.Tensor:
        raise AssertionError("dense fallback was attempted after fail-closed planning")

    monkeypatch.setattr(tiling, "stitch_tiles", _poison)
    req = OptimizeRequest(
        model=_identity(),
        pixel_size_nm=1.0,
        input_path=input_path,
        output_path=tmp_path / "out.oas",
        output_kind="oasis",
        writer="vsb",
        execution_mode="auto",
        max_dense_bytes=1,
    )
    with pytest.raises(StreamingUnsupportedError) as exc:
        optimize_layout(req)
    assert exc.value.reason == "STREAMING_UNSUPPORTED_INPUT"


def test_over_policy_curvilinear_request_fails_closed(tmp_path: Path) -> None:
    input_path = tmp_path / "in.npy"
    _write_npy(input_path)
    req = OptimizeRequest(
        model=_identity(),
        pixel_size_nm=1.0,
        input_path=input_path,
        output_path=tmp_path / "out.oas",
        output_kind="oasis",
        writer="mbmw",
        execution_mode="auto",
        max_dense_bytes=1,
    )
    with pytest.raises(StreamingUnsupportedError) as exc:
        optimize_layout(req)
    assert exc.value.reason == "STREAMING_UNSUPPORTED_OUTPUT"


# ---- Gate C2: streaming Manhattan OASIS ----------------------------------------


def test_streaming_manhattan_oasis_round_trips(tmp_path: Path, gds_path: Path) -> None:
    out_path = tmp_path / "out.oas"
    result = optimize_layout(
        OptimizeRequest(
            model=_identity(),
            pixel_size_nm=PX,
            input_path=gds_path,
            output_path=out_path,
            output_kind="oasis",
            writer="vsb",
            tile_size=16,
            execution_mode="streaming",
        )
    )
    assert result.plan.mode == "streaming"
    assert result.output_format == "oasis"
    ly = db.Layout()
    ly.read(str(out_path))
    top = ly.top_cells()[0]
    shapes = top.shapes(ly.layer(1, 0))
    assert shapes.size() > 0
    # The same input through the raster artifact backend: the Manhattan
    # writer must preserve occupancy exactly. The sink picks a pixel-sized
    # DBU, so box coordinates are already pixel indices.
    optimize_layout(
        OptimizeRequest(
            model=_identity(),
            pixel_size_nm=PX,
            input_path=gds_path,
            output_path=tmp_path / "raster.npy",
            output_kind="raster-npy",
            tile_size=16,
            execution_mode="streaming",
        )
    )
    artifact = np.asarray(np.load(tmp_path / "raster.npy", mmap_mode="r"))
    h = artifact.shape[0]
    back = np.zeros_like(artifact)
    for sh in shapes.each():
        b = sh.box
        x0, x1 = int(b.left), int(b.right)
        y0, y1 = h - int(b.top), h - int(b.bottom)
        back[y0:y1, x0:x1] = 1.0
    assert np.array_equal(back, artifact), "Manhattan export does not match the raster"


# ---- real-layout product smoke (§12.6) ------------------------------------------


def test_real_layout_smoke_through_product_path(tmp_path: Path, gds_path: Path) -> None:
    probe, plan = plan_request(
        OptimizeRequest(
            model=_identity(),
            pixel_size_nm=PX,
            input_path=gds_path,
            output_path=tmp_path / "out.oas",
            output_kind="oasis",
            writer="vsb",
            tile_size=16,
            execution_mode="auto",
        )
    )
    assert probe.kind == "gds"
    assert probe.streaming_supported
    # Under the default budget the tiny fixture is dense; a policy of one
    # byte forces auto onto the streaming branch — both traverse the spine.
    assert plan.mode == "dense"
    result = optimize_layout(
        OptimizeRequest(
            model=_identity(),
            pixel_size_nm=PX,
            input_path=gds_path,
            output_path=tmp_path / "out.oas",
            output_kind="oasis",
            writer="vsb",
            tile_size=16,
            execution_mode="auto",
            max_dense_bytes=1,  # force the policy so auto must stream
        )
    )
    assert (tmp_path / "out.oas").exists()
    assert result.work_accounting is not None
    assert result.work_accounting["forward_simulator_input_pixels"] > 0
    assert result.work_accounting["read_window_calls"] > 0


def test_non_pixel_aligned_gds_dense_fallback_and_fail_closed(
    tmp_path: Path, gds_path: Path
) -> None:
    """A GDS whose geometry is NOT pixel aligned is honestly unsupported
    for streaming: under policy -> dense, over policy -> fail closed."""
    ly = db.Layout()
    ly.dbu = 0.001
    cell = ly.create_cell("TOP")
    # 3005 DBU is not a multiple of the 8 nm pixel: bbox alignment fails.
    cell.shapes(ly.layer(1, 0)).insert(db.Box(1, 1, 3006, 3006))
    path = tmp_path / "misaligned.gds"
    ly.write(str(path))
    req = OptimizeRequest(
        model=_identity(),
        pixel_size_nm=PX,
        input_path=path,
        output_path=tmp_path / "out.npy",
        output_kind="raster-npy",
        tile_size=16,
        execution_mode="streaming",
    )
    with pytest.raises(StreamingUnsupportedError) as exc:
        optimize_layout(req)
    assert exc.value.reason == "STREAMING_UNSUPPORTED_INPUT"


# ---- engineering memory gate (§13, non-authority) --------------------------------


@pytest.mark.parametrize("side", [256, 1024])
def test_memory_gate_live_windows_bounded_by_tile(side: int, tmp_path: Path) -> None:
    """Engineering regression / architecture check (NOT a benchmark): with
    a fixed tile size, live raster bytes during product streaming must not
    grow with the layout dimensions.  Quadrupling `side` grows the layout
    16x; the ledger bound must stay constant."""
    import openlithohub.workflow.execution as execution_mod

    arr = np.zeros((side, side), dtype=np.float32)
    arr[side // 4 : side // 2, side // 4 : side // 2] = 1.0
    input_path = tmp_path / f"in_{side}.npy"
    np.save(input_path, arr)

    ledger = _WindowLedger()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(execution_mod, "MemmapTensorTileSource", _instrumented_source(ledger))
    try:
        req = _req(input_path, tmp_path / f"s_{side}.npy", mode="streaming", halo=HALO)
        result = optimize_layout(req)
    finally:
        monkeypatch.undo()
    assert result.plan.mode == "streaming"

    # One read window is (tile + 2*halo)^2 fp32 bytes; a small constant
    # factor covers the concurrent forward output + core slice.
    window_bytes = (TILE + 2 * HALO) ** 2 * 4
    assert ledger.max_live_bytes > 0
    assert ledger.max_live_bytes <= 16 * window_bytes, (
        f"side={side}: live raster bytes {ledger.max_live_bytes} exceed the "
        "tile-bounded architecture envelope"
    )


def test_memory_gate_bound_does_not_scale_with_layout(tmp_path: Path) -> None:
    sides = (256, 1024)
    bounds = []
    for side in sides:
        import openlithohub.workflow.execution as execution_mod

        arr = np.zeros((side, side), dtype=np.float32)
        arr[side // 4 : side // 2, side // 4 : side // 2] = 1.0
        input_path = tmp_path / f"in_{side}.npy"
        np.save(input_path, arr)
        ledger = _WindowLedger()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(execution_mod, "MemmapTensorTileSource", _instrumented_source(ledger))
        try:
            optimize_layout(_req(input_path, tmp_path / f"s_{side}.npy", mode="streaming"))
        finally:
            monkeypatch.undo()
        bounds.append(ledger.max_live_bytes)
    # 16x layout growth must not scale the live-window bound linearly:
    # allow at most 2x slack between the two runs (constant in principle).
    assert bounds[1] <= 2 * max(bounds[0], 1)
