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
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .types import ProofLevel


def _is_hex(value: str) -> bool:
    return all(c in "0123456789abcdefABCDEF" for c in value)


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
    """What the frozen spectral representation actually is.

    ``LEGACY_UNKNOWN`` is the honest state of a v1 migration: the v1
    schema carried no truncation metadata, so the representation is
    unknown and never theorem-facing admissible (PR-2C).
    """

    FULL_DISCRETE_SOURCE = "FULL_DISCRETE_SOURCE"
    SOCS_TRUNCATED = "SOCS_TRUNCATED"
    IMPORTED_FROZEN_FINITE_OPERATOR = "IMPORTED_FROZEN_FINITE_OPERATOR"
    LEGACY_UNKNOWN = "LEGACY_UNKNOWN"


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
    """One source-plane sample: position plus raw/normalized weights.

    Position semantics (P-054 re-audit PR-2C): ``sy``/``sx`` are populated
    only when the source-grid mapping is *known* — either frozen directly
    or supplied by an explicit, bijective ``source_index_map``.  A legacy
    v1 migration never invents coordinates: it records the stored flat
    index in ``legacy_flat_index`` and leaves ``sy``/``sx`` as ``None``.
    """

    sy: int | None
    sx: int | None
    raw_weight: ExactDyadic | None
    normalized_weight: ExactDyadic | None
    legacy_flat_index: int | None = None

    def __post_init__(self) -> None:
        if self.raw_weight is None and self.normalized_weight is None:
            raise ValueError("source sample carries no weights")
        if (self.sy is None) != (self.sx is None):
            raise ValueError("source sample coordinates must be both known or both unknown")


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
    ship a certified truncation error bound.  A ``LEGACY_UNKNOWN``
    representation (v1 migration without spectral evidence) is likewise
    inadmissible — unknown is never promoted to certified (PR-2C).
    """

    forward_model_id: str
    git_commit: str
    source_bins: tuple[tuple[int, int] | None, ...]
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
    # PR-2D: an IMPORTED_FROZEN_FINITE_OPERATOR must name the frozen
    # artifact it was imported from, with a certified import proof level.
    imported_artifact_sha256: str | None = None
    imported_artifact_proof_level: ProofLevel | None = None
    schema: str = "B04.source_snapshot.v2"

    def __post_init__(self) -> None:
        if len(self.source_bins) != len(self.samples):
            raise ValueError("source bins and samples must have the same length")
        if len(self.pupil_support_bits) != self.pupil_shape[0] * self.pupil_shape[1]:
            raise ValueError("pupil bits do not match pupil shape")
        if self.pixel_size_nm <= 0.0:
            raise ValueError("pixel_size_nm must be positive")
        if self.spectral_representation is SpectralRepresentation.SOCS_TRUNCATED:
            self._validate_truncation()
        if self.spectral_representation is SpectralRepresentation.IMPORTED_FROZEN_FINITE_OPERATOR:
            self._validate_imported_provenance()
        if self.spectral_representation is SpectralRepresentation.FULL_DISCRETE_SOURCE and (
            self.normalization.policy == "LEGACY_NORMALIZED_FLOAT_ONLY"
        ):
            # M6: a legacy-normalized snapshot cannot claim a full discrete
            # spectral representation — the upgrade was never discharged.
            raise ValueError(
                "LEGACY_NORMALIZED_FLOAT_ONLY provenance is incompatible with "
                "FULL_DISCRETE_SOURCE; the representation is LEGACY_UNKNOWN"
            )
        # when coordinates are declared known, bins and samples must agree
        for index, (bin_pos, sample) in enumerate(zip(self.source_bins, self.samples, strict=True)):
            if (
                bin_pos is not None
                and sample.sy is not None
                and tuple(bin_pos)
                != (
                    sample.sy,
                    sample.sx,
                )
            ):
                raise ValueError(f"source bin {index} disagrees with its sample coordinates")

    def _validate_imported_provenance(self) -> None:
        sha = self.imported_artifact_sha256
        if sha is None:
            raise ValueError(
                "an imported frozen finite operator must carry the sha256 of "
                "the artifact it was imported from (PR-2D: an enum rewrite is "
                "not provenance)"
            )
        if len(sha) != 64 or not all(c in "0123456789abcdefABCDEF" for c in sha):
            raise ValueError("imported artifact sha256 must be 64 hex characters")
        if self.imported_artifact_proof_level is not ProofLevel.IMPORTED_QDM_CERTIFIED:
            raise ValueError(
                "an imported frozen finite operator requires an "
                "IMPORTED-QDM-CERTIFIED import proof level"
            )

    def _validate_truncation(self) -> None:
        if self.truncation_error_upper is None or self.truncation_rank is None:
            raise ValueError(
                "a truncated spectral representation must carry its rank "
                "and a truncation error bound"
            )
        if self.truncation_rank <= 0:
            raise ValueError("truncation rank must be positive")
        if not math.isfinite(self.truncation_error_upper):
            raise ValueError("truncation error bound must be finite")
        if self.truncation_error_upper < 0.0:
            raise ValueError("truncation error bound must be nonnegative")
        if self.truncation_error_proof_level is None:
            raise ValueError("a truncation error bound must declare its proof level")

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
        """Theorem-facing admissibility rule (audit P0.3 + re-audit PR-2C/2D).

        - ``LEGACY_UNKNOWN`` is never admissible: unknown is not certified;
        - ``SOCS_TRUNCATED`` needs a certified error bound;
        - ``IMPORTED_FROZEN_FINITE_OPERATOR`` needs a valid 64-hex artifact
          identity plus a certified import proof level — an enum rewrite
          alone is never an upgrade.
        """
        if self.spectral_representation is SpectralRepresentation.LEGACY_UNKNOWN:
            return False
        if self.spectral_representation is SpectralRepresentation.SOCS_TRUNCATED:
            return self.truncation_bound_is_certified()
        if self.spectral_representation is SpectralRepresentation.IMPORTED_FROZEN_FINITE_OPERATOR:
            return (
                self.imported_artifact_sha256 is not None
                and len(self.imported_artifact_sha256) == 64
                and _is_hex(self.imported_artifact_sha256)
                and self.imported_artifact_proof_level is ProofLevel.IMPORTED_QDM_CERTIFIED
            )
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
    imported_artifact_sha256: str | None = None,
    imported_artifact_proof_level: ProofLevel | None = None,
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
        imported_artifact_sha256=imported_artifact_sha256,
        imported_artifact_proof_level=imported_artifact_proof_level,
    )


def migrate_v1_to_v2(
    v1: SourceSnapshot,
    *,
    source_index_map: dict[int, tuple[int, int]] | None = None,
) -> SourceSnapshotV2:
    """v1 → v2 migration with honest provenance fidelity (PR-2C).

    The v1 schema kept only normalized Python floats and flat source-bin
    indices, with no truncation metadata and no source-grid shape.  The
    migration therefore:

    - marks the spectral representation ``LEGACY_UNKNOWN`` (never
      ``FULL_DISCRETE_SOURCE`` — unknown is not silently upgraded);
    - preserves the stored flat indices verbatim in
      ``SourceSample.legacy_flat_index``;
    - leaves ``sy``/``sx`` as ``None`` unless the caller supplies an
      explicit, bijective ``source_index_map`` from flat index to source
      coordinates — the spatial image grid is *not* a source grid.

    Raw bits are gone and are never fabricated; every record is marked
    ``LEGACY_NORMALIZED_FLOAT_ONLY``.
    """
    if v1.schema != "B04.source_snapshot.v1":
        raise ValueError("migrate_v1_to_v2 expects a v1 snapshot")

    indices = [int(i) for i in v1.source_bin_indices]
    if len(indices) != len(v1.source_weights):
        raise ValueError("v1 bins and weights disagree")
    if source_index_map is not None:
        mapped_keys = sorted(source_index_map)
        if mapped_keys != sorted(set(indices)):
            raise ValueError("source_index_map must cover exactly the stored v1 flat indices")
        if len(set(source_index_map.values())) != len(source_index_map):
            raise ValueError("source_index_map must be bijective")

    samples = []
    bins: list[tuple[int, int] | None] = []
    for flat_index, weight in zip(indices, v1.source_weights, strict=True):
        mapped = source_index_map.get(flat_index) if source_index_map else None
        if mapped:
            sy: int | None = mapped[0]
            sx: int | None = mapped[1]
        else:
            sy, sx = None, None
        samples.append(
            SourceSample(
                sy=sy,
                sx=sx,
                raw_weight=None,
                normalized_weight=ExactDyadic.from_float(weight),
                legacy_flat_index=flat_index,
            )
        )
        bins.append((mapped[0], mapped[1]) if mapped else None)
    return SourceSnapshotV2(
        forward_model_id=v1.forward_model_id,
        git_commit=v1.git_commit,
        source_bins=tuple(bins),
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
        spectral_representation=SpectralRepresentation.LEGACY_UNKNOWN,
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
