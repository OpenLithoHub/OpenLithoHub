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
from enum import Enum
from pathlib import Path
from typing import Any

from .types import ProofLevel

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
    def is_topk_truncated(self) -> bool | None:
        """Whether the frozen spectral representation is top-K truncated.

        P-054 repo integration: the v1 schema carries **no** truncation
        metadata, so v1 honestly answers ``None`` (unknown) — it can no
        longer claim ``False`` (proven full) without evidence.  The v2
        schema (``B04.source_snapshot.v2``) declares
        ``spectral_representation`` explicitly and derives a real answer.
        """
        return None

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


# ---------------------------------------------------------------------------
# P-054 repo integration — SourceSnapshot v2: exact dyadic provenance
# ---------------------------------------------------------------------------


class SpectralRepresentation(str, Enum):
    """What the frozen spectral representation actually is."""

    FULL_DISCRETE_SOURCE = "FULL_DISCRETE_SOURCE"
    SOCS_TRUNCATED = "SOCS_TRUNCATED"
    IMPORTED_FROZEN_FINITE_OPERATOR = "IMPORTED_FROZEN_FINITE_OPERATOR"


@dataclass(frozen=True)
class ExactDyadic:
    """Exact rational ``numerator/denominator`` provenance of one float.

    ``source_hex`` pins the raw IEEE-754 bit pattern; the pair is the
    exact rational the float represents.  Raw bits are provenance — they
    are never re-derived or fabricated after the fact.
    """

    numerator: int
    denominator: int
    source_dtype: str
    source_hex: str

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("dyadic denominator must be positive")
        try:
            bits = float.fromhex(self.source_hex)
        except ValueError as exc:
            raise ValueError(f"invalid source_hex {self.source_hex!r}") from exc
        if bits.as_integer_ratio() != (self.numerator, self.denominator):
            raise ValueError("dyadic provenance does not round-trip its source bits")

    @classmethod
    def from_float(cls, value: float, dtype: str = "float64") -> ExactDyadic:
        num, den = value.as_integer_ratio()
        return cls(
            numerator=num,
            denominator=den,
            source_dtype=dtype,
            source_hex=float(value).hex(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceSample:
    """One source-plane sample: grid position plus raw/normalized weights."""

    sy: int
    sx: int
    raw_weight: ExactDyadic | None
    normalized_weight: ExactDyadic | None

    def __post_init__(self) -> None:
        if self.raw_weight is None and self.normalized_weight is None:
            raise ValueError(f"source sample ({self.sy},{self.sx}) carries no weights")


@dataclass(frozen=True)
class NormalizationRecord:
    """What normalization was applied, as exact rationals where known."""

    raw_weight_sum: ExactDyadic | None
    source_normalization_factor: ExactDyadic | None
    open_frame_factor: ExactDyadic | None
    policy: str


@dataclass(frozen=True)
class SourceSnapshotV2:
    """``B04.source_snapshot.v2`` — complete reconstruction provenance.

    Distinguishes raw IEEE-754 source weights, the raw mass, the
    normalization factor and the normalized model, and declares the
    spectral representation explicitly (P-054 audit P0.2/P0.3).  Truncated
    representations may not back theorem-facing certificates unless they
    ship a certified truncation error bound.
    """

    forward_model_id: str
    git_commit: str
    source_bins: tuple[tuple[int, int], ...]
    samples: tuple[SourceSample, ...]
    pupil_support_bits: tuple[int, ...]
    pupil_shape: tuple[int, int]
    wavelength_nm: float
    na_x: float
    na_y: float
    pixel_size_nm: float
    grid_shape: tuple[int, int]
    focus_dose_convention: str
    mask_sha256: str
    normalization: NormalizationRecord
    spectral_representation: SpectralRepresentation
    process_parameters: dict[str, float] = field(default_factory=dict)
    truncation_rank: int | None = None
    truncation_error_upper: float | None = None
    truncation_error_proof_level: ProofLevel | None = None
    schema: str = "B04.source_snapshot.v2"

    def __post_init__(self) -> None:
        if len(self.source_bins) != len(self.samples):
            raise ValueError("source bins and samples must have the same length")
        if len(self.pupil_support_bits) != self.pupil_shape[0] * self.pupil_shape[1]:
            raise ValueError("pupil bits do not match pupil shape")
        if self.pixel_size_nm <= 0.0:
            raise ValueError("pixel_size_nm must be positive")
        if self.spectral_representation is SpectralRepresentation.SOCS_TRUNCATED and (
            self.truncation_error_upper is None or self.truncation_rank is None
        ):
            raise ValueError(
                "a truncated spectral representation must carry its rank "
                "and a truncation error bound"
            )

    @property
    def is_topk_truncated(self) -> bool:
        """Derived from the declared representation — no guessing."""
        return self.spectral_representation is SpectralRepresentation.SOCS_TRUNCATED

    def truncation_bound_is_certified(self) -> bool:
        return self.truncation_error_proof_level in (
            ProofLevel.INTERVAL_CERTIFIED,
            ProofLevel.IMPORTED_QDM_CERTIFIED,
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["spectral_representation"] = self.spectral_representation.value
        if self.truncation_error_proof_level is not None:
            out["truncation_error_proof_level"] = self.truncation_error_proof_level.value
        return out

    def content_sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def source_native_certificate_admissible(self) -> bool:
        """Theorem-facing admissibility rule (audit P0.3 verification rule)."""
        if self.spectral_representation is SpectralRepresentation.SOCS_TRUNCATED:
            return self.truncation_bound_is_certified()
        return True


def freeze_source_snapshot_v2(
    *,
    source_bins: list[tuple[int, int]] | tuple[tuple[int, int], ...],
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
    spectral_representation: SpectralRepresentation = SpectralRepresentation.FULL_DISCRETE_SOURCE,
    truncation_rank: int | None = None,
    truncation_error_upper: float | None = None,
    truncation_error_proof_level: ProofLevel | None = None,
    open_frame_factor: float | None = None,
    process_parameters: dict[str, float] | None = None,
) -> SourceSnapshotV2:
    """Freeze a v2 snapshot: raw dyadic weights + full normalization record."""
    raw = tuple(float(w) for w in source_weights)
    total = sum(raw)
    if total <= 0.0:
        raise ValueError("source weights must have positive mass")
    normalized = tuple(w / total for w in raw)
    samples = tuple(
        SourceSample(
            sy=int(sy),
            sx=int(sx),
            raw_weight=ExactDyadic.from_float(w),
            normalized_weight=ExactDyadic.from_float(nw),
        )
        for (sy, sx), w, nw in zip(source_bins, raw, normalized, strict=True)
    )
    normalization = NormalizationRecord(
        raw_weight_sum=ExactDyadic.from_float(total),
        source_normalization_factor=ExactDyadic.from_float(1.0 / total),
        open_frame_factor=(
            ExactDyadic.from_float(open_frame_factor) if open_frame_factor is not None else None
        ),
        policy="raw_mass_normalized",
    )
    return SourceSnapshotV2(
        forward_model_id=FORWARD_MODEL_ID,
        git_commit=git_commit,
        source_bins=tuple((int(sy), int(sx)) for sy, sx in source_bins),
        samples=samples,
        pupil_support_bits=tuple(int(bool(b)) for b in pupil_support),
        pupil_shape=(int(pupil_shape[0]), int(pupil_shape[1])),
        wavelength_nm=float(wavelength_nm),
        na_x=float(na_x),
        na_y=float(na_y),
        pixel_size_nm=float(pixel_size_nm),
        grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
        focus_dose_convention=FOCUS_DOSE_CONVENTION,
        mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
        normalization=normalization,
        spectral_representation=spectral_representation,
        process_parameters=dict(process_parameters or {}),
        truncation_rank=truncation_rank,
        truncation_error_upper=truncation_error_upper,
        truncation_error_proof_level=truncation_error_proof_level,
    )


def migrate_v1_to_v2(v1: SourceSnapshot) -> SourceSnapshotV2:
    """v1 → v2 migration with honest provenance fidelity.

    The v1 schema kept only normalized Python floats: the raw bits are
    gone and are never fabricated.  The migration marks every record
    ``LEGACY_NORMALIZED_FLOAT_ONLY``.
    """
    if v1.schema != "B04.source_snapshot.v1":
        raise ValueError("migrate_v1_to_v2 expects a v1 snapshot")
    grid_h, grid_w = v1.grid_shape
    samples = []
    for index, weight in enumerate(v1.source_weights):
        sy, sx = divmod(int(index), grid_w)
        samples.append(
            SourceSample(
                sy=sy,
                sx=sx,
                raw_weight=None,
                normalized_weight=ExactDyadic.from_float(weight),
            )
        )
    bins = tuple(divmod(i, grid_w) for i in range(len(v1.source_weights)))
    return SourceSnapshotV2(
        forward_model_id=v1.forward_model_id,
        git_commit=v1.git_commit,
        source_bins=bins,
        samples=tuple(samples),
        pupil_support_bits=v1.pupil_support_bits,
        pupil_shape=v1.pupil_shape,
        wavelength_nm=v1.wavelength_nm,
        na_x=v1.na_x,
        na_y=v1.na_y,
        pixel_size_nm=v1.pixel_size_nm,
        grid_shape=v1.grid_shape,
        focus_dose_convention=v1.focus_dose_convention,
        mask_sha256=v1.mask_sha256,
        normalization=NormalizationRecord(
            raw_weight_sum=None,
            source_normalization_factor=None,
            open_frame_factor=None,
            policy="LEGACY_NORMALIZED_FLOAT_ONLY",
        ),
        spectral_representation=SpectralRepresentation.FULL_DISCRETE_SOURCE,
        process_parameters=dict(v1.process_parameters),
    )


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
