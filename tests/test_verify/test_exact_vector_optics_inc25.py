from dataclasses import dataclass

import numpy as np
import pytest

from openlithohub.verify.exact_vector_optics import (
    ExactVectorMaskContract,
    exact_vector_normalized_spectrum,
    exact_vector_representation_bridge_upper,
    materialized_reference_spectrum,
)
from openlithohub.verify.interface_runs import HorizontalRun


@dataclass
class FakeExactRunSource:
    runs: tuple[HorizontalRun, ...]
    layout_hash: str = "fixture-hash"
    pixel_size_nm: float = 8.0
    backend_kind: str = "exact-vector-scanline"

    def iter_runs_for_bbox(
        self,
        bbox_px: tuple[int, int, int, int],
    ):
        x0, y0, x1, y1 = bbox_px
        return tuple(
            HorizontalRun(
                y=r.y,
                x0=max(r.x0, x0),
                x1=min(r.x1, x1),
            )
            for r in self.runs
            if y0 <= r.y < y1 and min(r.x1, x1) > max(r.x0, x0)
        )


def test_exact_run_spectrum_matches_same_indicator_fft():
    source = FakeExactRunSource(runs=tuple(HorizontalRun(y, 2, 6) for y in range(1, 5)))
    got = exact_vector_normalized_spectrum(source, bbox_xyxy=(0, 0, 8, 8))
    mask = np.zeros((8, 8))
    mask[1:5, 2:6] = 1.0
    want = materialized_reference_spectrum(mask)
    assert np.max(np.abs(got.normalized_spectrum - want)) < 2e-15
    assert got.occupied_pixels == 16
    assert got.run_count == 4
    assert exact_vector_representation_bridge_upper(got.contract) == 0.0


def test_nonzero_global_bbox_is_translated_to_local_tile():
    source = FakeExactRunSource(
        runs=(
            HorizontalRun(11, 22, 25),
            HorizontalRun(12, 23, 27),
        )
    )
    got = exact_vector_normalized_spectrum(source, bbox_xyxy=(20, 10, 28, 18))
    mask = np.zeros((8, 8))
    mask[1, 2:5] = 1
    mask[2, 3:7] = 1
    assert np.max(np.abs(got.normalized_spectrum - materialized_reference_spectrum(mask))) < 2e-15


def test_dense_loader_is_forbidden_by_contract():
    with pytest.raises(ValueError, match="forbids"):
        ExactVectorMaskContract(
            layout_hash="x",
            backend_kind="legacy-dense",
            pixel_size_nm=8.0,
            bbox_xyxy=(0, 0, 8, 8),
            dense_loader_used=True,
        )


def test_non_square_proof_tile_is_rejected():
    source = FakeExactRunSource(runs=())
    with pytest.raises(ValueError, match="square"):
        exact_vector_normalized_spectrum(source, bbox_xyxy=(0, 0, 8, 7))
