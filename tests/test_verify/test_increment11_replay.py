from pathlib import Path

import pytest

from openlithohub.verify.replay import replay_expanded_band


def test_increment11_external_replay_when_artifact_is_present():
    # CI may omit proof blobs.  When the B04 proof artifact is mounted, this
    # becomes an exact regression of the final expanded-band gate.
    grid = (
        Path(__file__).resolve().parents[2]
        / "proof_artifacts"
        / "B04_ExactSource_ExpandedBand_Grid_2026-09-09.npz"
    )
    if not grid.exists():
        pytest.skip("B04 proof blob not installed")
    result = replay_expanded_band(
        grid,
        threshold=0.225,
        pixel_nm=8.0,
        spatial_third_derivative_upper_per_nm3=4.972184653511609e-05,
        delta_total=0.005752571687594228,
    )
    assert result.active_cells == 164
    assert result.worst_cell_yx == (18, 20)
    assert result.kappa_lower_per_nm == pytest.approx(0.005891396342472183, abs=1e-15)
    assert result.hausdorff_upper_nm == pytest.approx(0.976436035396712, abs=1e-12)
