"""Product execution spine (PR-C): one planner owning dense vs streaming.

Every product surface — the CLI ``optimize run``, :class:`LitheEngine`,
the synchronous ``POST /v1/optimize`` and the async job API — executes
through :func:`optimize_layout`.  The planner
(:func:`plan_execution`) chooses between the two executors and always
returns a typed reason; nothing changes execution topology silently.

Dense executor
    The historical path: ``load_layout`` → ``tile_layout`` (overlapping
    windows) → ``model.predict`` per tile → ``stitch_tiles`` (weight-map
    blend) → threshold → export.  Kept byte-for-byte compatible for
    small jobs.

Streaming executor
    The RFC 0008 authority: :func:`openlithohub.streaming.run_streaming`
    over a windowed :class:`~openlithohub.streaming.TileSource`
    (memmap raster or KLayout-aligned exact vector) writing trusted
    cores into an out-of-core sink — either a disk-backed ``.npy``
    memmap raster (Gate C1) or :class:`~openlithohub.streaming.
    StreamingManhattanTileSink` Manhattan OASIS (Gate C2).  The
    streaming branch never calls ``load_layout`` / ``stitch_tiles`` and
    never retains per-tile results; peak host raster memory is bounded
    by tile/read-window + model state + writer buffers, not full-layout
    area.

Fail-honest ``auto``
    Under the dense memory policy the planner may pick dense.  Above the
    policy it picks streaming **only** when the whole combination
    (input, output, model) is streaming-capable; otherwise it fails
    closed with :class:`StreamingUnsupportedError` instead of silently
    allocating a full-chip raster.  Explicit ``mode="dense"`` is the
    documented user override.

Engineering invariant, not a benchmark claim: the Industrial Benchmark
v1.1 and P-054 authorities are untouched by this module.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from openlithohub.models.base import LithographyModel
from openlithohub.streaming import (
    LegacyFixedHaloPolicy,
    MemmapTensorTileSource,
    MemmapTileSink,
    StreamingManhattanTileSink,
    StreamingRunReport,
    TensorTileSink,
    TensorTileSource,
    TileSink,
    run_streaming,
)
from openlithohub.streaming.geometry import BoundingBox
from openlithohub.workflow.process_node import ProcessNodeConfig

ExecutionMode = Literal["auto", "dense", "streaming"]
OutputKind = Literal["oasis", "raster-npy", "tensor-mask"]

# ---- typed planner reasons (the only topology vocabulary) ----------------

DENSE_SMALL_LAYOUT = "DENSE_SMALL_LAYOUT"
"""``auto`` under the dense memory policy: dense is allowed."""

DENSE_REQUESTED_EXPLICITLY = "DENSE_REQUESTED_EXPLICITLY"
"""The caller explicitly requested ``mode="dense"`` (documented override,
including above the dense memory policy)."""

STREAMING_SUPPORTED = "STREAMING_SUPPORTED"
"""The caller explicitly requested ``mode="streaming"`` on a supported
combination."""

STREAMING_REQUIRED_BY_MEMORY_POLICY = "STREAMING_REQUIRED_BY_MEMORY_POLICY"
"""``auto`` above the dense memory policy with a fully supported
streaming combination."""

STREAMING_UNSUPPORTED_INPUT = "STREAMING_UNSUPPORTED_INPUT"
STREAMING_UNSUPPORTED_OUTPUT = "STREAMING_UNSUPPORTED_OUTPUT"
STREAMING_UNSUPPORTED_MODEL = "STREAMING_UNSUPPORTED_MODEL"

DENSE_EXECUTION_BACKEND = "dense-raster"
STREAMING_INPUT_BACKENDS = {
    "npy": "memmap-raster",
    "gds": "klayout-aligned-vector",
    "oas": "klayout-aligned-vector",
    "tensor": "in-memory-tensor",
    "pt": "dense-file",
}

DEFAULT_MAX_DENSE_BYTES = 8 * 1024**3
"""Default dense memory policy: the largest single fp32 raster the dense
path may materialise under ``auto``.  Pure policy — not benchmark
authority.  ``OPENLITHOHUB_MAX_DENSE_BYTES`` overrides; ``0`` = unlimited."""

MAX_DENSE_BYTES_ENV = "OPENLITHOHUB_MAX_DENSE_BYTES"

# Code-owned streaming capability table (single source for the
# /v1/capabilities matrix; the probe logic must stay consistent with it
# and tests cross-check the two).
STREAMING_INPUT_SUPPORT: dict[str, bool] = {
    "npy": True,
    "gds": True,
    "oas": True,
    "pt": False,
}


class StreamingUnsupportedError(ValueError):
    """Fail-closed error: a request needs streaming but the combination
    cannot provide it (or explicitly requested streaming is unsupported).

    Subclasses :class:`ValueError` so existing ``ValueError`` error
    mappings (HTTP 400, CLI error path) keep working.  ``reason`` carries
    the typed planner reason.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def coerce_execution_mode(value: str) -> ExecutionMode:
    """Validate + narrow an externally supplied execution mode."""
    if value == "auto":
        return "auto"
    if value == "dense":
        return "dense"
    if value == "streaming":
        return "streaming"
    raise ValueError(f"execution_mode must be 'auto', 'dense' or 'streaming', got {value!r}")


def dense_memory_budget(env: dict[str, str] | None = None) -> int:
    """Resolve the dense memory policy in bytes (``0`` = unlimited)."""
    source = env if env is not None else os.environ
    raw = source.get(MAX_DENSE_BYTES_ENV, "")
    if raw == "":
        return DEFAULT_MAX_DENSE_BYTES
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{MAX_DENSE_BYTES_ENV} must be an integer byte count, got {raw!r}"
        ) from None
    if value < 0:
        raise ValueError(f"{MAX_DENSE_BYTES_ENV} must be >= 0 (0 = unlimited), got {value}")
    return value


def klayout_available() -> bool:
    """Whether the real KLayout Python bindings import.

    Probes ``klayout.db`` specifically: a bare ``klayout`` namespace (e.g.
    the repository's own ``klayout/`` tooling directory shadowing a missing
    PyPI install) is not sufficient for Manhattan output.
    """
    try:
        import klayout.db  # noqa: F401,PLC0415 — optional dependency probe
    except Exception:  # noqa: BLE001 — any import failure means unavailable
        return False
    return True


# ---- planner types --------------------------------------------------------


@dataclass(frozen=True)
class ExecutionPlan:
    """The planner's decision, with its typed reason and backends."""

    mode: Literal["dense", "streaming"]
    reason: str
    estimated_dense_bytes: int
    max_dense_bytes: int
    over_memory_policy: bool
    input_backend: str
    output_backend: str
    detail: str = ""


@dataclass
class InputProbe:
    """Header-only (or geometry-only) inspection of an optimize input.

    ``streaming_source`` is a ready-to-use windowed
    :class:`~openlithohub.streaming.TileSource` when the input is
    streaming-capable — built once here so execution never re-parses.

    Honesty flags: ``materialized_for_probe`` marks inputs (.pt) whose
    inspection already materialised the full raster under the current
    representation; ``unsupported_reason`` records why streaming is not
    available for this input.
    """

    kind: str
    shape: tuple[int, int]
    pixel_size_nm: float
    streaming_source: Any = None
    dense_tensor: torch.Tensor | None = None
    """Materialised input for inherently dense representations (.pt): the
    probe already paid the load, so the dense branch reuses it instead of
    reading the file twice."""
    streaming_supported: bool = False
    explicit_only_ok: bool = False
    """True when an explicit ``execution_mode="streaming"`` may still run
    over this input with a documented caveat (in-memory tensor: the caller
    already materialised it, so the input side cannot be out-of-core)."""
    unsupported_reason: str = ""
    materialized_for_probe: bool = False

    @property
    def estimated_dense_bytes(self) -> int:
        return int(self.shape[0]) * int(self.shape[1]) * 4


@dataclass
class OptimizeRequest:
    """One product optimize request, surface-independent.

    Exactly one of ``input_path`` / ``input_tensor`` must be set.
    ``output_kind``:

    * ``"oasis"`` — OASIS/GDS mask-writer file at ``output_path``
      (Manhattan for ``writer="vsb"``, curvilinear for ``"mbmw"``).
    * ``"raster-npy"`` — self-describing ``.npy`` raster artifact
      (memmap-backed on the streaming branch — Gate C1).
    * ``"tensor-mask"`` — return the binarized mask in memory (the
      :class:`LitheEngine` contract; never selected by ``auto`` for the
      streaming branch, allowed only as an explicit
      ``execution_mode="streaming"`` endpoint with a documented caveat).

    ``threshold`` binarizes the model output (applied after the weight-map
    blend on the dense branch, per trusted core on the streaming branch).
    """

    model: LithographyModel
    pixel_size_nm: float
    input_path: Path | None = None
    input_tensor: torch.Tensor | None = None
    output_path: Path | None = None
    output_kind: OutputKind = "oasis"
    writer: str = "mbmw"
    layer: str | None = None
    node: ProcessNodeConfig | None = None
    tile_size: int = 2048
    halo_px: int | None = None
    """Explicit tile halo override (CLI --halo/--overlap). ``None`` computes
    the RFC 0005 halo from the process node and model, matching the dense
    path's historical default."""
    threshold: float = 0.5
    min_area_nm2: float = 0.0
    execution_mode: ExecutionMode = "auto"
    max_dense_bytes: int | None = None
    forward_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """What one executed optimize produced (JSON-friendly plus optional
    in-memory mask for the engine surface)."""

    plan: ExecutionPlan
    shape: tuple[int, int]
    n_tiles: int
    halo_px: int
    threshold: float
    output_kind: OutputKind
    output_format: str
    output_path: Path | None = None
    mask: torch.Tensor | None = None
    work_accounting: dict[str, float | int] | None = None
    streaming_report: Any = None


# ---- probing ---------------------------------------------------------------


def _check_request(req: OptimizeRequest) -> None:
    if (req.input_path is None) == (req.input_tensor is None):
        raise ValueError("OptimizeRequest needs exactly one of input_path / input_tensor")
    if req.output_kind != "tensor-mask" and req.output_path is None:
        raise ValueError(f"output_kind={req.output_kind!r} requires output_path")
    if req.tile_size <= 1:
        raise ValueError(f"tile_size must be > 1, got {req.tile_size}")
    if req.pixel_size_nm <= 0:
        raise ValueError(f"pixel_size_nm must be positive, got {req.pixel_size_nm}")
    if req.writer not in ("mbmw", "vsb"):
        raise ValueError(f"unknown writer {req.writer!r}; expected 'mbmw' or 'vsb'")


def probe_input(req: OptimizeRequest) -> InputProbe:
    """Inspect the request input without materialising a full raster where
    the representation allows it.

    * ``.npy`` — header-only via ``mmap_mode="r"``; the memmap source is
      kept for execution.
    * ``.gds``/``.oas`` — parser-only geometry pass through
      :class:`~openlithohub.streaming.vector_runs.KLayoutAlignedRunSource`;
      no rasterization.  Requires KLayout and exactly pixel-aligned
      geometry (integer DBU-per-pixel); anything else is recorded as
      streaming-unsupported, never silently degraded.
    * ``.pt`` — must be loaded (inherently dense representation) and is
      recorded as streaming-unsupported with that honesty flag set.
    * in-memory tensor — already materialised by the caller; streaming
      cannot un-materialise it, so it is recorded as streaming-unsupported
      for the input side.
    """
    _check_request(req)
    if req.input_tensor is not None:
        t = req.input_tensor
        if t.ndim != 2:
            raise ValueError(
                f"expected a 2-D layout tensor, got ndim={t.ndim} shape {tuple(t.shape)}"
            )
        return InputProbe(
            kind="tensor",
            shape=(int(t.shape[0]), int(t.shape[1])),
            pixel_size_nm=req.pixel_size_nm,
            streaming_source=TensorTileSource(t, req.pixel_size_nm),
            streaming_supported=False,
            explicit_only_ok=True,
            unsupported_reason=(
                "input is an in-memory dense tensor; out-of-core input applies "
                "to path-based inputs only"
            ),
        )

    if req.input_path is None:
        raise ValueError("OptimizeRequest needs input_path for file inputs")
    path = Path(req.input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()

    if suffix == ".npy":
        map_2d = np.load(str(path), mmap_mode="r", allow_pickle=False)
        if map_2d.ndim != 2:
            raise ValueError(
                f"{path}: expected a 2-D ndarray for layout input, got ndim={map_2d.ndim}"
            )
        shape = (int(map_2d.shape[0]), int(map_2d.shape[1]))
        return InputProbe(
            kind="npy",
            shape=shape,
            pixel_size_nm=req.pixel_size_nm,
            streaming_source=MemmapTensorTileSource(path, pixel_size_nm=req.pixel_size_nm),
            streaming_supported=True,
        )

    if suffix == ".pt":
        loaded = torch.load(str(path), weights_only=True)
        if not isinstance(loaded, torch.Tensor) or loaded.ndim != 2:
            raise ValueError(
                f"{path}: expected a 2-D torch.Tensor for layout input, got {type(loaded).__name__}"
            )
        return InputProbe(
            kind="pt",
            shape=(int(loaded.shape[0]), int(loaded.shape[1])),
            pixel_size_nm=req.pixel_size_nm,
            dense_tensor=loaded,
            streaming_supported=False,
            unsupported_reason=(
                ".pt layout input is an inherently dense representation under "
                "the current implementation; streaming requires .npy or a "
                "pixel-aligned GDS/OASIS layout"
            ),
            materialized_for_probe=True,
        )

    if suffix in (".gds", ".oas"):
        try:
            from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource

            source = KLayoutAlignedRunSource.from_file(
                path, pixel_size_nm=req.pixel_size_nm, layer=req.layer
            )
        except ImportError as e:
            return InputProbe(
                kind=path.suffix.lower().lstrip("."),
                shape=(0, 0),
                pixel_size_nm=req.pixel_size_nm,
                unsupported_reason=f"KLayout unavailable for vector streaming: {e}",
            )
        except ValueError as e:
            return InputProbe(
                kind=path.suffix.lower().lstrip("."),
                shape=(0, 0),
                pixel_size_nm=req.pixel_size_nm,
                unsupported_reason=(f"layout is not eligible for exact vector streaming: {e}"),
            )
        return InputProbe(
            kind=path.suffix.lower().lstrip("."),
            shape=source.shape,
            pixel_size_nm=req.pixel_size_nm,
            streaming_source=source,
            streaming_supported=True,
        )

    raise ValueError(
        f"Unsupported input format {suffix!r} for optimize execution; "
        "expected .npy, .pt, .gds or .oas"
    )


# ---- planning ---------------------------------------------------------------


def _streaming_output_support(output_kind: OutputKind, writer: str) -> tuple[bool, str, bool]:
    """``(supported, detail, explicit_only_ok)``.

    ``supported`` means the output backend can honour the streaming
    memory invariant end to end (``auto`` may select it above the dense
    policy).  ``explicit_only_ok`` marks backends that an explicit
    ``execution_mode="streaming"`` may still use with a documented
    caveat — the exact-core streaming scheduler runs, but the endpoint
    itself is in-memory, so the end-to-end invariant does not hold.
    """
    if output_kind == "raster-npy":
        return True, "", True
    if output_kind == "oasis":
        if writer == "vsb":
            if klayout_available():
                return True, "", True
            return (
                False,
                "KLayout is not installed; Manhattan OASIS streaming output unavailable",
                False,
            )
        return (
            False,
            "streaming + curvilinear (mbmw) output is unsupported for large jobs; "
            "use writer='vsb' for streaming Manhattan output or run under the "
            "dense memory policy",
            False,
        )
    return (
        False,
        "in-memory tensor output cannot satisfy the streaming memory invariant "
        "(the caller receives a full dense mask); use output_kind='raster-npy' "
        "or 'oasis'",
        True,
    )


def plan_execution(
    *,
    input_probe: InputProbe,
    output_kind: OutputKind,
    writer: str,
    model: LithographyModel,
    max_dense_bytes: int,
    requested_mode: ExecutionMode,
) -> ExecutionPlan:
    """Decide dense vs streaming and return the typed reason.

    Never changes execution topology silently: above the dense memory
    policy, ``auto`` either streams a fully supported combination or
    raises :class:`StreamingUnsupportedError`.
    """
    estimated = input_probe.estimated_dense_bytes
    over = max_dense_bytes > 0 and estimated > max_dense_bytes
    output_ok, output_detail, output_explicit_ok = _streaming_output_support(output_kind, writer)
    model_ok = bool(getattr(model, "streaming_supported", True))
    input_ok = input_probe.streaming_supported

    if requested_mode == "dense":
        return ExecutionPlan(
            mode="dense",
            reason=DENSE_REQUESTED_EXPLICITLY if over else DENSE_SMALL_LAYOUT,
            estimated_dense_bytes=estimated,
            max_dense_bytes=max_dense_bytes,
            over_memory_policy=over,
            input_backend=DENSE_EXECUTION_BACKEND,
            output_backend=_dense_output_backend(output_kind),
            detail=("explicit dense override above the dense memory policy" if over else ""),
        )

    blockers: list[tuple[str, str]] = []
    if not input_ok:
        blockers.append((STREAMING_UNSUPPORTED_INPUT, input_probe.unsupported_reason))
    input_explicit_ok = input_probe.explicit_only_ok
    if not output_ok:
        blockers.append((STREAMING_UNSUPPORTED_OUTPUT, output_detail))
    if not model_ok:
        blockers.append(
            (
                STREAMING_UNSUPPORTED_MODEL,
                f"model {model.name!r} declares SUPPORTS_STREAMING=False",
            )
        )

    if requested_mode == "streaming":
        # Blockers that are merely "explicit-only" (in-memory tensor input,
        # tensor-mask output) do not hard-block an explicit streaming
        # request; anything else does.
        explicit_ok = {
            STREAMING_UNSUPPORTED_INPUT: input_explicit_ok,
            STREAMING_UNSUPPORTED_OUTPUT: output_explicit_ok,
            STREAMING_UNSUPPORTED_MODEL: False,
        }
        hard_blockers = [b for b in blockers if not explicit_ok[b[0]]]
        if hard_blockers:
            reason, detail = hard_blockers[0]
            raise StreamingUnsupportedError(
                reason,
                f"execution-mode='streaming' is unsupported for this request [{reason}]: {detail}",
            )
        plan = _streaming_plan(
            input_probe, output_kind, writer, estimated, max_dense_bytes, over, STREAMING_SUPPORTED
        )
        if blockers:
            # Explicit-only endpoints: the streaming scheduler runs, but the
            # caller materialises input and/or output, so the end-to-end
            # memory invariant is documented as not holding.
            plan = replace(
                plan,
                detail=(
                    "explicit streaming over in-memory endpoint(s): resident "
                    "memory stays O(layout); the exact-core scheduling "
                    "semantics still apply"
                ),
            )
        return plan

    # auto
    if not over:
        return ExecutionPlan(
            mode="dense",
            reason=DENSE_SMALL_LAYOUT,
            estimated_dense_bytes=estimated,
            max_dense_bytes=max_dense_bytes,
            over_memory_policy=False,
            input_backend=DENSE_EXECUTION_BACKEND,
            output_backend=_dense_output_backend(output_kind),
        )
    if not blockers:
        return _streaming_plan(
            input_probe,
            output_kind,
            writer,
            estimated,
            max_dense_bytes,
            over,
            STREAMING_REQUIRED_BY_MEMORY_POLICY,
        )
    # FAIL CLOSED: above the policy there is no honest dense fallback.
    reason, detail = blockers[0]
    raise StreamingUnsupportedError(
        reason,
        f"estimated dense footprint {estimated} bytes exceeds the dense memory "
        f"policy ({max_dense_bytes} bytes, {MAX_DENSE_BYTES_ENV}) and streaming "
        f"is unsupported for this request [{reason}]: {detail} — failing closed "
        f"instead of silently allocating a full-chip raster",
    )


def _streaming_plan(
    input_probe: InputProbe,
    output_kind: OutputKind,
    writer: str,
    estimated: int,
    max_dense_bytes: int,
    over: bool,
    reason: str,
) -> ExecutionPlan:
    if output_kind == "raster-npy":
        output_backend = "streaming-memmap-raster"
    elif output_kind == "oasis":
        output_backend = "streaming-manhattan-oasis"
    else:
        output_backend = "streaming-tensor-endpoint"
    return ExecutionPlan(
        mode="streaming",
        reason=reason,
        estimated_dense_bytes=estimated,
        max_dense_bytes=max_dense_bytes,
        over_memory_policy=over,
        input_backend=STREAMING_INPUT_BACKENDS.get(input_probe.kind, input_probe.kind),
        output_backend=output_backend,
    )


def _dense_output_backend(output_kind: OutputKind) -> str:
    if output_kind == "raster-npy":
        return "dense-raster-npy"
    if output_kind == "oasis":
        return "dense-oasis"
    return "dense-tensor"


# ---- execution ---------------------------------------------------------------

ProgressFn = Callable[[int, int], None]


def _resolve_halo_px(req: OptimizeRequest) -> int:
    if req.halo_px is not None:
        if not 0 <= req.halo_px < req.tile_size:
            raise ValueError(
                f"halo_px must be >= 0 and < tile_size ({req.tile_size}), got {req.halo_px}"
            )
        return req.halo_px

    from openlithohub.workflow.halo import compute_halo_px

    return compute_halo_px(
        node=req.node,
        model=req.model,
        pixel_nm=req.pixel_size_nm,
        tile_size=req.tile_size,
    )


def _run_dense(
    req: OptimizeRequest,
    probe: InputProbe,
    progress: ProgressFn | None,
) -> tuple[torch.Tensor, int, int]:
    """The historical dense path, verbatim: overlapping tiles + blend."""
    from openlithohub.workflow.tiling import stitch_tiles, tile_layout

    if req.input_tensor is not None:
        tensor = req.input_tensor
    elif probe.dense_tensor is not None:
        tensor = probe.dense_tensor
    else:
        from openlithohub.data.io import load_layout

        if req.input_path is None:
            raise ValueError("dense execution of a path request requires input_path")
        tensor = load_layout(req.input_path, req.pixel_size_nm, layer=req.layer)

    halo_px = _resolve_halo_px(req)
    tiles = tile_layout(tensor, tile_size=req.tile_size, overlap=halo_px)
    shape = (int(tensor.shape[0]), int(tensor.shape[1]))
    tile_results: list[tuple[Any, torch.Tensor]] = []
    total = len(tiles)
    for index, tile in enumerate(tiles):
        result = req.model.predict(tile.tensor, **req.forward_kwargs)
        tile_results.append((tile, result.mask))
        if progress is not None:
            progress(index + 1, total)
    stitched = stitch_tiles(tile_results, shape)
    mask = (stitched > req.threshold).float()
    return mask, len(tiles), halo_px


class _CountingSink:
    """Delegate TileSink that counts committed cores for progress reporting.

    run_streaming owns the tile loop and must not be rewritten for UI
    hooks (§4); observing committed cores at the sink is the adapter.
    """

    def __init__(
        self, inner: TileSink, on_commit: Callable[[int, int], None], planned: int
    ) -> None:
        self._inner = inner
        self._on_commit = on_commit
        self._planned = planned
        self._done = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self._inner.shape

    def write_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._inner.write_core(tile_id, bbox, tensor, metadata)
        self._done += 1
        self._on_commit(self._done, self._planned)

    def record_certified_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        *,
        exact_fill: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        recorder = getattr(self._inner, "record_certified_core", None)
        if not callable(recorder):
            raise TypeError("inner sink cannot represent certified commits")
        recorder(tile_id, bbox, exact_fill=exact_fill, metadata=metadata)
        self._done += 1
        self._on_commit(self._done, self._planned)

    def finalize(self) -> Any:
        return self._inner.finalize()


def _run_streaming(
    req: OptimizeRequest,
    probe: InputProbe,
    progress: ProgressFn | None,
) -> tuple[Path | None, int, int, dict[str, float | int], Any, torch.Tensor | None]:
    """The product streaming branch: run_streaming as the scheduling
    authority, out-of-core source/sink, threshold applied per window
    before the pipeline slices each trusted core (pointwise, so it is
    exact per core)."""
    source = probe.streaming_source
    if source is None:  # the planner only dispatches streaming on a capable probe
        raise RuntimeError("streaming execution requires a streaming-capable input probe")
    shape = probe.shape
    halo_px = _resolve_halo_px(req)

    model_predict = req.model.predict
    forward_kwargs = req.forward_kwargs
    threshold = req.threshold

    def forward_fn(tile: torch.Tensor) -> torch.Tensor:
        return (model_predict(tile, **forward_kwargs).mask > threshold).float()

    if req.output_kind == "raster-npy":
        if req.output_path is None:
            raise RuntimeError("raster-npy output requires output_path")
        sink: Any = MemmapTileSink(shape, req.output_path, dtype=np.dtype(np.float32), npy=True)
    elif req.output_kind == "oasis":
        if req.output_path is None:
            raise RuntimeError("oasis output requires output_path")
        sink = StreamingManhattanTileSink(shape, req.output_path, pixel_size_nm=req.pixel_size_nm)
    else:
        # Explicit-only in-memory endpoint: exact-core streaming scheduler,
        # dense tensor destination (documented caveat on the plan).
        sink = TensorTileSink(shape)

    counting = _CountingSink(
        sink,
        (lambda done, total: progress(done, total) if progress else None),
        _planned_tile_count(shape, req.tile_size, halo_px),
    )

    report: StreamingRunReport = run_streaming(
        source,
        counting,
        forward_fn,
        core_size=req.tile_size,
        halo_policy=LegacyFixedHaloPolicy(halo_px),
        max_halo_px=halo_px,
        pixel_nm=req.pixel_size_nm,
    )
    output: Any = sink.finalize()
    mask = None
    out_path = req.output_path
    if req.output_kind == "tensor-mask":
        mask = output.detach().clone().float()
        out_path = None
    return (
        out_path,
        report.n_tiles,
        halo_px,
        dict(report.work_accounting),
        report,
        mask,
    )


def _planned_tile_count(shape: tuple[int, int], core_size: int, halo_px: int) -> int:
    from openlithohub.streaming import plan_tile_requests

    return len(plan_tile_requests(shape, core_size, halo_px))


def plan_request(req: OptimizeRequest) -> tuple[InputProbe, ExecutionPlan]:
    """Probe the input and plan the execution without running it.

    Surfaces that want to display (or log) the planner's decision before
    committing to a long run call this first and pass the result to
    :func:`optimize_layout` via ``prepared`` — the input is probed and any
    vector layout parsed exactly once.
    """
    _check_request(req)
    budget = req.max_dense_bytes if req.max_dense_bytes is not None else dense_memory_budget()
    probe = probe_input(req)
    plan = plan_execution(
        input_probe=probe,
        output_kind=req.output_kind,
        writer=req.writer,
        model=req.model,
        max_dense_bytes=budget,
        requested_mode=req.execution_mode,
    )
    return probe, plan


def optimize_layout(
    req: OptimizeRequest,
    *,
    progress: ProgressFn | None = None,
    prepared: tuple[InputProbe, ExecutionPlan] | None = None,
) -> ExecutionResult:
    """Plan and execute one product optimize through the single spine.

    ``prepared`` accepts the output of :func:`plan_request` so the probe
    (and for GDS/OASIS the vector parse) happens once across planning and
    execution.
    """
    if prepared is not None:
        probe, plan = prepared
    else:
        probe, plan = plan_request(req)

    if plan.mode == "streaming":
        output_path, n_tiles, halo_px, accounting, report, mask = _run_streaming(
            req, probe, progress
        )
        return ExecutionResult(
            plan=plan,
            shape=probe.shape,
            n_tiles=n_tiles,
            halo_px=halo_px,
            threshold=req.threshold,
            output_kind=req.output_kind,
            output_format={
                "raster-npy": "npy",
                "oasis": "oasis",
                "tensor-mask": "tensor",
            }[req.output_kind],
            output_path=output_path,
            mask=mask,
            work_accounting=accounting,
            streaming_report=report,
        )

    mask, n_tiles, halo_px = _run_dense(req, probe, progress)
    if req.output_kind == "tensor-mask":
        return ExecutionResult(
            plan=plan,
            shape=probe.shape,
            n_tiles=n_tiles,
            halo_px=halo_px,
            threshold=req.threshold,
            output_kind=req.output_kind,
            output_format="tensor",
            mask=mask,
        )
    if req.output_path is None:  # validated by _check_request; narrowed here
        raise RuntimeError("file outputs require output_path")
    if req.output_kind == "raster-npy":
        np.save(str(req.output_path), mask.numpy().astype(np.float32, copy=False))
        output_path = req.output_path
        output_format = "npy"
    else:
        from openlithohub.workflow.export import export_oasis

        export_mode = "curvilinear" if req.writer == "mbmw" else "manhattan"
        try:
            export_oasis(
                mask,
                req.output_path,
                mode=export_mode,
                pixel_size_nm=req.pixel_size_nm,
                min_area_nm2=req.min_area_nm2,
            )
            output_path = req.output_path
            output_format = "oasis"
        except ImportError:
            # Same honest fallback the server has always had when the
            # export stack is absent: keep the result as a tensor file.
            fallback = req.output_path.with_suffix(".pt")
            torch.save(mask, str(fallback))
            output_path = fallback
            output_format = "torch"
    return ExecutionResult(
        plan=plan,
        shape=probe.shape,
        n_tiles=n_tiles,
        halo_px=halo_px,
        threshold=req.threshold,
        output_kind=req.output_kind,
        output_format=output_format,
        output_path=output_path,
        mask=mask,
    )


# ---- code-owned capability matrix -------------------------------------------


def streaming_capability_matrix() -> dict[str, Any]:
    """Truthful streaming support matrix for ``GET /v1/capabilities``.

    Populated from code-owned tables, not documentation.  ``gds``/``oas``
    ``true`` means *eligible* inputs: the planner still requires exact
    pixel-aligned geometry (integer DBU-per-pixel) and fails closed above
    the dense memory policy when the adapter refuses a layout.
    """
    matrix: dict[str, Any] = {
        "engine": True,
        "product_execution": True,
        "inputs": dict(STREAMING_INPUT_SUPPORT),
        "outputs": {
            "memmap_raster": True,
            "manhattan_oasis": klayout_available(),
            "curvilinear_oasis": False,
        },
        "scheduling_authority": "openlithohub.streaming.run_streaming",
        "memory_policy": {
            "env": MAX_DENSE_BYTES_ENV,
            "default_bytes": DEFAULT_MAX_DENSE_BYTES,
            "unlimited_marker": 0,
            "fail_closed": True,
        },
        "conditions": {
            "gds_oas": (
                "requires KLayout and exactly pixel-aligned geometry "
                "(integer DBU-per-pixel); non-aligned layouts fall back per "
                "the dense memory policy and fail closed above it"
            ),
            "curvilinear": (
                "streaming + curvilinear (mbmw) output is unsupported for "
                "large jobs; dense remains available under the memory policy"
            ),
        },
    }
    return matrix
