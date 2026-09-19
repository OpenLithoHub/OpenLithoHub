"""P-054 repo integration — SourceSnapshot v2 exact-provenance semantics.

Audit P0.2/P0.3: a frozen snapshot must distinguish raw IEEE-754 source
weights, the raw mass, the normalization factor, and the normalized
model; truncated spectral representations may not back theorem-facing
certificates without a certified error bound; the v1 → v2 migration
never fabricates the raw bits it does not have.
"""

import pytest

from openlithohub.verify.source_snapshot import (
    ExactDyadic,
    NormalizationRecord,
    SourceSample,
    SourceSnapshotV2,
    SpectralRepresentation,
    dyadic_from_float,
    freeze_source_snapshot,
    freeze_source_snapshot_v2,
    migrate_v1_to_v2,
)
from openlithohub.verify.types import ProofLevel


def _kwargs(**overrides):
    base = dict(
        source_bins=[(0, 1), (1, 0)],
        source_weights=[1.0, 3.0],
        pupil_support=[1, 1, 0, 1],
        pupil_shape=(2, 2),
        wavelength_nm=193.0,
        na_x=1.35,
        na_y=1.35,
        pixel_size_nm=8.0,
        grid_shape=(8, 8),
        mask_bytes=b"mask-bytes",
        git_commit="0" * 40,
    )
    base.update(overrides)
    return base


def test_v2_freeze_records_exact_dyadic_provenance():
    snap = freeze_source_snapshot_v2(**_kwargs())
    assert snap.schema == "B04.source_snapshot.v2"
    assert snap.source_bins == ((0, 1), (1, 0))
    raws = [s.raw_weight for s in snap.samples]
    assert raws[0].numerator == 1 and raws[0].denominator == 1
    assert raws[1].numerator == 3 and raws[1].denominator == 1
    assert snap.normalization.raw_weight_sum.numerator == 4
    factor = snap.normalization.source_normalization_factor
    assert factor.numerator == 1 and factor.denominator == 4
    # normalized weights are the exact dyadics 1/4 and 3/4
    assert snap.samples[0].normalized_weight.numerator == 1
    assert snap.samples[0].normalized_weight.denominator == 4
    assert snap.samples[1].normalized_weight.numerator == 3
    # reconstruction: raw * factor == normalized, exactly
    for s in snap.samples:
        assert (
            s.raw_weight.numerator * factor.numerator,
            s.raw_weight.denominator * factor.denominator,
        ) == (s.normalized_weight.numerator, s.normalized_weight.denominator)


def test_v2_hash_is_content_pinned():
    snap = freeze_source_snapshot_v2(**_kwargs())
    other = freeze_source_snapshot_v2(**_kwargs())
    assert snap.content_sha256() == other.content_sha256()
    changed = freeze_source_snapshot_v2(**_kwargs(source_weights=[1.0, 2.0]))
    assert snap.content_sha256() != changed.content_sha256()


def test_truncated_without_bound_is_rejected_and_inadmissible():
    with pytest.raises(ValueError, match="must carry its rank"):
        freeze_source_snapshot_v2(
            **_kwargs(
                spectral_representation=SpectralRepresentation.SOCS_TRUNCATED,
            )
        )
    truncated = freeze_source_snapshot_v2(
        **_kwargs(
            spectral_representation=SpectralRepresentation.SOCS_TRUNCATED,
            truncation_rank=12,
            truncation_error_upper=1e-3,
            truncation_error_proof_level=ProofLevel.HEURISTIC,
        )
    )
    assert truncated.is_topk_truncated is True
    assert truncated.source_native_certificate_admissible() is False

    certified = freeze_source_snapshot_v2(
        **_kwargs(
            spectral_representation=SpectralRepresentation.SOCS_TRUNCATED,
            truncation_rank=12,
            truncation_error_upper=1e-3,
            truncation_error_proof_level=ProofLevel.INTERVAL_CERTIFIED,
        )
    )
    assert certified.source_native_certificate_admissible() is True


def test_full_discrete_source_is_admissible():
    snap = freeze_source_snapshot_v2(**_kwargs())
    assert snap.is_topk_truncated is False
    assert snap.source_native_certificate_admissible() is True
    assert snap.spectral_representation is SpectralRepresentation.FULL_DISCRETE_SOURCE


def _v1(indices):
    return freeze_source_snapshot(
        source_bin_indices=indices,
        source_weights=[1.0, 3.0, 2.0][: len(indices)],
        pupil_support=[1, 1, 0, 1],
        pupil_shape=(2, 2),
        wavelength_nm=193.0,
        na_x=1.35,
        na_y=1.35,
        pixel_size_nm=8.0,
        grid_shape=(8, 8),
        mask_bytes=b"mask-bytes",
        git_commit="0" * 40,
    )


def test_v1_migration_keeps_unknown_representation_m1():
    v2 = migrate_v1_to_v2(_v1([0, 1]))
    assert v2.spectral_representation is SpectralRepresentation.LEGACY_UNKNOWN
    assert v2.source_native_certificate_admissible() is False


def test_v1_migration_preserves_flat_source_ids_m3():
    # nonconsecutive indices are preserved verbatim — never re-enumerated
    v2 = migrate_v1_to_v2(_v1([5, 17, 42]))
    assert [s.legacy_flat_index for s in v2.samples] == [5, 17, 42]
    assert all(s.sy is None and s.sx is None for s in v2.samples)  # M4
    assert v2.source_bins == (None, None, None)


def test_migration_with_explicit_map_populates_coordinates_m5():
    v2 = migrate_v1_to_v2(_v1([5, 17]), source_index_map={5: (0, 2), 17: (3, 1)})
    assert v2.source_bins == ((0, 2), (3, 1))
    assert all(s.sy is not None and s.sx is not None for s in v2.samples)
    # bijectivity/coverage violations fail closed
    with pytest.raises(ValueError, match="cover exactly"):
        migrate_v1_to_v2(_v1([5, 17]), source_index_map={5: (0, 0)})
    with pytest.raises(ValueError, match="bijective"):
        migrate_v1_to_v2(_v1([5, 17]), source_index_map={5: (1, 1), 17: (1, 1)})


def test_legacy_normalized_cannot_claim_full_discrete_m6():
    v2 = migrate_v1_to_v2(_v1([0, 1]))
    with pytest.raises(ValueError, match="LEGACY_NORMALIZED_FLOAT_ONLY"):
        SourceSnapshotV2(
            forward_model_id=v2.forward_model_id,
            git_commit=v2.git_commit,
            source_bins=v2.source_bins,
            samples=v2.samples,
            pupil_support_bits=v2.pupil_support_bits,
            pupil_shape=v2.pupil_shape,
            wavelength_nm=v2.wavelength_nm,
            na_x=v2.na_x,
            na_y=v2.na_y,
            pixel_size_nm=v2.pixel_size_nm,
            grid_shape=v2.grid_shape,
            focus_dose_convention=v2.focus_dose_convention,
            mask_sha256=v2.mask_sha256,
            normalization=v2.normalization,
            spectral_representation=SpectralRepresentation.FULL_DISCRETE_SOURCE,
        )


def test_truncation_metadata_validation():
    base = dict(
        spectral_representation=SpectralRepresentation.SOCS_TRUNCATED,
        truncation_rank=0,
        truncation_error_upper=1e-3,
        truncation_error_proof_level=ProofLevel.INTERVAL_CERTIFIED,
    )
    with pytest.raises(ValueError, match="rank must be positive"):
        freeze_source_snapshot_v2(**_kwargs(**base))
    with pytest.raises(ValueError, match="finite"):
        freeze_source_snapshot_v2(
            **_kwargs(
                spectral_representation=SpectralRepresentation.SOCS_TRUNCATED,
                truncation_rank=4,
                truncation_error_upper=float("inf"),
                truncation_error_proof_level=ProofLevel.INTERVAL_CERTIFIED,
            )
        )
    with pytest.raises(ValueError, match="nonnegative"):
        freeze_source_snapshot_v2(
            **_kwargs(
                spectral_representation=SpectralRepresentation.SOCS_TRUNCATED,
                truncation_rank=4,
                truncation_error_upper=-1.0,
                truncation_error_proof_level=ProofLevel.INTERVAL_CERTIFIED,
            )
        )
    with pytest.raises(ValueError, match="declare its proof level"):
        freeze_source_snapshot_v2(
            **_kwargs(
                spectral_representation=SpectralRepresentation.SOCS_TRUNCATED,
                truncation_rank=4,
                truncation_error_upper=1e-3,
                truncation_error_proof_level=None,
            )
        )


def test_exact_dyadic_round_trip_and_validation():
    d = ExactDyadic.from_float(0.1)
    assert d.source_dtype == "float64"
    assert float(d.numerator) / d.denominator == 0.1
    with pytest.raises(ValueError):
        ExactDyadic(numerator=1, denominator=0, source_dtype="f", source_hex="0x1p0")


def test_source_sample_requires_at_least_one_weight():
    with pytest.raises(ValueError, match="carries no weights"):
        SourceSample(sy=0, sx=0, raw_weight=None, normalized_weight=None)


def test_v2_json_round_trip():
    import json

    snap = freeze_source_snapshot_v2(**_kwargs())
    blob = json.dumps(snap.to_dict(), sort_keys=True)
    restored = json.loads(blob)
    assert restored["schema"] == "B04.source_snapshot.v2"
    assert restored["normalization"]["policy"] == "raw_mass_normalized"
    assert restored["samples"][1]["raw_weight"]["numerator"] == 3
    assert dyadic_from_float(0.5) == (1, 2)
    assert NormalizationRecord  # imported for API completeness
