"""Source-native backend + snapshot contract tests (prompt §10A/§B04-B)."""

import pytest

from openlithohub.verify.source_native import (
    BACKEND_ID as SOURCE_NATIVE_BACKEND_ID,
)
from openlithohub.verify.source_native import (
    OutwardRoundedCPUBackend,
    ProcessBox,
    SpatialDerivativeEnclosure,
)
from openlithohub.verify.source_snapshot import (
    FORWARD_MODEL_ID,
    SourceSnapshot,
    dyadic_from_float,
    freeze_source_snapshot,
    outward_round_interval,
)


def _snapshot() -> SourceSnapshot:
    return freeze_source_snapshot(
        source_bin_indices=[0, 1, 2, 3],
        source_weights=[1.0, 1.0, 2.0, 4.0],
        pupil_support=[1, 1, 1, 0],
        pupil_shape=(2, 2),
        wavelength_nm=13.5,
        na_x=0.33,
        na_y=0.33,
        pixel_size_nm=1.0,
        grid_shape=(64, 64),
        mask_bytes=b"\x00\x01" * 32,
        git_commit="deadbeef" * 8,
    )


class TestSourceSnapshot:
    def test_weights_are_normalized(self):
        snap = _snapshot()
        assert sum(snap.source_weights) == pytest.approx(1.0, abs=1e-12)
        assert snap.source_bin_indices == (0, 1, 2, 3)

    def test_topk_flag_is_false_on_source_native_path(self):
        assert _snapshot().is_topk_truncated is False

    def test_hash_pinned_and_stable(self):
        snap = _snapshot()
        assert len(snap.mask_sha256) == 64
        assert snap.content_sha256() == snap.content_sha256()

    def test_weight_count_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            freeze_source_snapshot(
                source_bin_indices=[0, 1],
                source_weights=[1.0],
                pupil_support=[1],
                pupil_shape=(1, 1),
                wavelength_nm=13.5,
                na_x=0.33,
                na_y=0.33,
                pixel_size_nm=1.0,
                grid_shape=(8, 8),
                mask_bytes=b"x",
                git_commit="c",
            )

    def test_zero_weight_mass_rejected(self):
        with pytest.raises(ValueError, match="positive mass"):
            freeze_source_snapshot(
                source_bin_indices=[0],
                source_weights=[0.0],
                pupil_support=[1],
                pupil_shape=(1, 1),
                wavelength_nm=13.5,
                na_x=0.33,
                na_y=0.33,
                pixel_size_nm=1.0,
                grid_shape=(8, 8),
                mask_bytes=b"x",
                git_commit="c",
            )

    def test_roundtrip_json(self, tmp_path):
        snap = _snapshot()
        path = snap.to_json(tmp_path / "snapshot.json")
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["schema"] == "B04.source_snapshot.v1"
        assert raw["forward_model_id"] == FORWARD_MODEL_ID
        assert raw["source_bin_indices"] == [0, 1, 2, 3]


class TestDyadicImport:
    def test_exact_dyadic_import(self):
        # 0.1 is not exact in binary, but as_integer_ratio is exact for the
        # double that represents it.
        num, den = dyadic_from_float(0.1)
        assert den % 2 == 0 and num / den == 0.1

    def test_dyadic_of_powers_of_two_is_exact_small(self):
        num, den = dyadic_from_float(0.25)
        assert (num, den) == (1, 4)

    def test_nonfinite_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            dyadic_from_float(float("nan"))
        with pytest.raises(ValueError, match="non-finite"):
            dyadic_from_float(float("inf"))


class TestOutwardRounding:
    def test_interval_widens_outward(self):
        lo, hi = outward_round_interval(0.1, 0.2)
        assert lo < 0.1 and hi > 0.2

    def test_rejects_inverted(self):
        with pytest.raises(ValueError, match="lo must not exceed hi"):
            outward_round_interval(0.3, 0.2)


class TestSourceNativeBackend:
    def test_backend_id_is_opt_in_source_native(self):
        assert SOURCE_NATIVE_BACKEND_ID == "source_native_full"
        assert OutwardRoundedCPUBackend.backend_id == "source_native_full"

    def test_field_enclosure_covers_process_box(self):
        backend = OutwardRoundedCPUBackend()
        snapshot = _snapshot()
        box = ProcessBox(intervals=(("focus_nm", -50.0, 50.0), ("dose", 0.9, 1.1)))
        enclosure = backend.enclose_field(snapshot, box, (0, 0, 8, 8))
        assert enclosure.field_lower <= 0.0 <= enclosure.field_upper
        assert enclosure.sup_process_perturbation >= (50.0 - (-50.0)) + (1.1 - 0.9)

    def test_derivative_enclosure_bounds(self):
        backend = OutwardRoundedCPUBackend()
        snapshot = _snapshot()
        der = backend.enclose_gradient_and_hessian(snapshot, (0, 0, 8, 8))
        assert isinstance(der, SpatialDerivativeEnclosure)
        assert der.gradient_lower >= 0.0
        assert der.hessian_op_upper >= der.gradient_lower

    def test_freeze_snapshot_from_context(self):
        class Ctx:
            snapshot = _snapshot()

        backend = OutwardRoundedCPUBackend()
        assert backend.freeze_snapshot(Ctx()).mask_sha256 == _snapshot().mask_sha256

    def test_freeze_snapshot_rejects_foreign_context(self):
        with pytest.raises(ValueError, match="SourceSnapshot"):
            OutwardRoundedCPUBackend().freeze_snapshot(object())

    def test_process_box_validation(self):
        with pytest.raises(ValueError, match="focus"):
            ProcessBox(intervals=(("focus_nm", 50.0, -50.0),))
