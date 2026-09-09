"""B04 source-native verification snapshot (prompt §B04-B, RFC 0007).

A *source snapshot* freezes the discrete model a verifier will actually
evaluate — not a high-level configuration record.  Everything here is the
realized discrete model: exact source-bin indices, normalized weights,
pupil-support bits, grid conventions, and content hashes.  IEEE-754
floats belonging to the realized model may be imported as exact dyadic
rationals (:func:`dyadic_from_float`) so downstream interval evaluation
is outward-rounded rather than heuristic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORWARD_MODEL_ID = "openlithohub.source_native_full.hopkins.discrete"

FOCUS_DOSE_CONVENTION = (
    "focus in nm (0 = best focus, + into wafer); dose as relative exposure "
    "multiplier on the normalized aerial intensity"
)


@dataclass(frozen=True)
class SourceSnapshot:
    """Frozen realized discrete Hopkins/SOCS model.

    ``source_bin_indices`` / ``source_weights`` are the full discrete
    source: weights are already normalized and NO dynamic top-K spectral
    branch selection is applied on this path.  ``pupil_support_bits`` is
    the flattened binary pupil support (1 = pass).  ``mask_sha256`` pins
    the input mask bytes; ``git_commit`` pins the implementation.
    """

    forward_model_id: str
    git_commit: str
    source_bin_indices: tuple[int, ...]
    source_weights: tuple[float, ...]
    pupil_support_bits: tuple[int, ...]
    pupil_shape: tuple[int, int]
    wavelength_nm: float
    na_x: float
    na_y: float
    pixel_size_nm: float
    grid_shape: tuple[int, int]
    focus_dose_convention: str
    mask_sha256: str
    process_parameters: dict[str, float] = field(default_factory=dict)
    schema: str = "B04.source_snapshot.v1"

    def __post_init__(self) -> None:
        if len(self.source_bin_indices) != len(self.source_weights):
            raise ValueError("source bins and weights must have the same length")
        total = sum(self.source_weights)
        if total <= 0.0 or abs(total - 1.0) > 1e-6:
            raise ValueError(f"source weights must sum to ~1, got {total}")
        if len(self.pupil_support_bits) != self.pupil_shape[0] * self.pupil_shape[1]:
            raise ValueError("pupil bits do not match pupil shape")
        if self.pixel_size_nm <= 0.0:
            raise ValueError("pixel_size_nm must be positive")

    @property
    def is_topk_truncated(self) -> bool:
        """True only if the caller explicitly built a truncated snapshot.

        The source-native path freezes the *full* discrete source; a
        truncated top-K representation must carry an explicit
        truncation/equivalence error bound before its numbers may be
        promoted to a continuous certificate (prompt §B04-A).
        """
        return False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def content_sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def dyadic_from_float(value: float) -> tuple[int, int]:
    """Exact dyadic-rational import of an IEEE-754 double.

    Returns ``(numerator, denominator)`` with ``denominator`` a power of
    two — the exact rational the float represents.  Outward rounding of
    downstream evaluation then has a well-defined exact starting point
    instead of a heuristic re-interpolation.
    """
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("cannot import a non-finite float as a dyadic rational")
    return value.as_integer_ratio()


def outward_round_interval(lo: float, hi: float) -> tuple[float, float]:
    """Widen an interval by one ULP outward on each side."""
    import math

    if hi < lo:
        raise ValueError("interval lo must not exceed hi")
    return math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)


def freeze_source_snapshot(
    *,
    source_bin_indices: list[int] | tuple[int, ...],
    source_weights: list[float] | tuple[float, ...],
    pupil_support: list[int] | tuple[int, ...],
    pupil_shape: tuple[int, int],
    wavelength_nm: float,
    na_x: float,
    na_y: float,
    pixel_size_nm: float,
    grid_shape: tuple[int, int],
    mask_bytes: bytes,
    git_commit: str,
    process_parameters: dict[str, float] | None = None,
) -> SourceSnapshot:
    """Freeze a snapshot, normalizing weights and hashing the mask bytes."""
    weights = tuple(float(w) for w in source_weights)
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("source weights must have positive mass")
    normalized = tuple(w / total for w in weights)
    return SourceSnapshot(
        forward_model_id=FORWARD_MODEL_ID,
        git_commit=git_commit,
        source_bin_indices=tuple(int(i) for i in source_bin_indices),
        source_weights=normalized,
        pupil_support_bits=tuple(int(bool(b)) for b in pupil_support),
        pupil_shape=(int(pupil_shape[0]), int(pupil_shape[1])),
        wavelength_nm=float(wavelength_nm),
        na_x=float(na_x),
        na_y=float(na_y),
        pixel_size_nm=float(pixel_size_nm),
        grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
        focus_dose_convention=FOCUS_DOSE_CONVENTION,
        mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
        process_parameters=dict(process_parameters or {}),
    )
