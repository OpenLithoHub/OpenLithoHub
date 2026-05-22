"""Tests for scripts/leaderboard_validate_submissions.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the script as a module by path — it lives outside the package so
# pytest can't import it the normal way.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "leaderboard_validate_submissions.py"
spec = importlib.util.spec_from_file_location("leaderboard_validate_submissions", _SCRIPT)
assert spec and spec.loader
_mod = importlib.util.module_from_spec(spec)
sys.modules["leaderboard_validate_submissions"] = _mod
spec.loader.exec_module(_mod)


_VALID_YAML = """\
model_name: test-model
dataset: lithobench
process_node: 7nm
mask_topology: curvilinear
epe_mean_nm: 1.5
epe_max_nm: 3.0
l2_error_pixels: 42.0
"""

_BROKEN_YAML_BAD_NODE = """\
model_name: bad-node
dataset: lithobench
process_node: 9999nm
mask_topology: curvilinear
epe_mean_nm: 1.5
epe_max_nm: 3.0
l2_error_pixels: 42.0
"""

_BROKEN_YAML_NEGATIVE_EPE = """\
model_name: bad-epe
dataset: lithobench
process_node: 7nm
mask_topology: curvilinear
epe_mean_nm: -1.5
epe_max_nm: 3.0
l2_error_pixels: 42.0
"""

_BROKEN_YAML_MISSING_L2 = """\
model_name: cheater
dataset: lithobench
process_node: 7nm
mask_topology: curvilinear
epe_mean_nm: 0.0
epe_max_nm: 0.0
"""


def test_no_files_returns_error(tmp_path: Path) -> None:
    rc = _mod.main(tmp_path, tmp_path / "_validated.json")
    assert rc == 1


def test_all_valid_writes_output(tmp_path: Path) -> None:
    sub_dir = tmp_path / "submissions" / "team-a"
    sub_dir.mkdir(parents=True)
    (sub_dir / "a.yaml").write_text(_VALID_YAML)
    out = tmp_path / "_validated.json"

    rc = _mod.main(tmp_path / "submissions", out)
    assert rc == 0
    assert out.exists()


def test_collects_all_errors_before_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two broken submissions should both be reported in one pass — the
    submitter shouldn't have to fix one, push, and discover the next."""
    sub_dir = tmp_path / "submissions" / "team-b"
    sub_dir.mkdir(parents=True)
    (sub_dir / "good.yaml").write_text(_VALID_YAML)
    (sub_dir / "bad-node.yaml").write_text(_BROKEN_YAML_BAD_NODE)
    (sub_dir / "bad-epe.yaml").write_text(_BROKEN_YAML_NEGATIVE_EPE)

    rc = _mod.main(tmp_path / "submissions", tmp_path / "_validated.json")
    assert rc == 1
    captured = capsys.readouterr().out
    # Both bad files must have surfaced an annotation.
    assert "bad-node.yaml" in captured
    assert "bad-epe.yaml" in captured
    # Good file still got logged.
    assert "good.yaml" in captured
    # Validated output is NOT written when any file failed (atomic semantics).
    assert not (tmp_path / "_validated.json").exists()


def test_missing_l2_error_is_rejected_by_ci(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Forward-sim gate at the CI surface: a YAML without ``l2_error_pixels``
    must be rejected with a clear remediation message, mirroring
    ``LeaderboardStore.submit``."""
    sub_dir = tmp_path / "submissions" / "team-c"
    sub_dir.mkdir(parents=True)
    (sub_dir / "cheater.yaml").write_text(_BROKEN_YAML_MISSING_L2)

    rc = _mod.main(tmp_path / "submissions", tmp_path / "_validated.json")
    assert rc == 1
    captured = capsys.readouterr().out
    assert "l2_error_pixels is required" in captured
    assert "openlithohub.simulators" in captured
