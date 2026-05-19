"""End-to-end test of the auto-leaderboard submission firewall.

The `auto-leaderboard.yml` workflow runs untrusted PR-supplied YAML
through `yaml.safe_load → BenchmarkResult.model_validate →
LeaderboardStore.submit`. The schema is the only firewall between
contributor input and the committed JSON, so this test exercises the
full pipeline (not the schema in isolation) and asserts that:

1. A well-formed submission round-trips end-to-end and lands on disk.
2. Malicious YAML cannot inject extra fields into the committed JSON.
3. Schema-violating YAML is rejected before anything is written.
4. YAML-specific tricks (anchors, type tags) cannot smuggle Python
   objects through `yaml.safe_load`.

If this test fails, the workflow's safety guarantee is broken — the
issue is more important than the test failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from openlithohub.leaderboard.schema import BenchmarkResult
from openlithohub.leaderboard.tracker import LeaderboardStore


def _run_workflow(yaml_text: str, store: LeaderboardStore) -> str:
    """Replicate the auto-leaderboard.yml validate+submit steps."""
    data = yaml.safe_load(yaml_text)
    result = BenchmarkResult.model_validate(data)
    return store.submit(result)


def _store(tmp_path: Path) -> LeaderboardStore:
    return LeaderboardStore(tmp_path / "leaderboard.json")


_VALID_YAML = """
model_name: example-model
dataset: lithobench
process_node: 7nm
mask_topology: curvilinear
epe_mean_nm: 1.5
epe_max_nm: 4.2
paper_url: https://example.com/paper
code_url: https://github.com/example/model
"""


def test_valid_submission_round_trips_end_to_end(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sub_id = _run_workflow(_VALID_YAML, store)

    assert sub_id.startswith("example-model-")
    on_disk = json.loads(store.path.read_text())
    assert "entries" in on_disk
    assert len(on_disk["entries"]) == 1
    entry = on_disk["entries"][0]
    assert entry["model_name"] == "example-model"
    assert entry["epe_mean_nm"] == 1.5
    assert entry["submission_id"] == sub_id


def test_extra_fields_in_yaml_cannot_reach_disk(tmp_path: Path) -> None:
    """A submitter sneaking unknown keys must be rejected — not silently
    persisted under the canonical entry."""
    store = _store(tmp_path)
    malicious = _VALID_YAML + "\nis_admin_override: true\nshell_payload: 'rm -rf /'\nrank: 1\n"
    with pytest.raises(ValidationError):
        _run_workflow(malicious, store)
    # And — critically — nothing was written.
    assert not store.path.exists() or json.loads(store.path.read_text())["entries"] == []


def test_oversized_strings_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    bloated = _VALID_YAML.replace("model_name: example-model", f"model_name: {'x' * 500}")
    with pytest.raises(ValidationError):
        _run_workflow(bloated, store)


def test_javascript_url_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    xss = _VALID_YAML.replace(
        "paper_url: https://example.com/paper",
        "paper_url: 'javascript:alert(1)'",
    )
    with pytest.raises(ValidationError):
        _run_workflow(xss, store)


def test_file_url_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    leak = _VALID_YAML.replace(
        "code_url: https://github.com/example/model",
        "code_url: 'file:///etc/passwd'",
    )
    with pytest.raises(ValidationError):
        _run_workflow(leak, store)


def test_negative_metric_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    bad = _VALID_YAML.replace("epe_mean_nm: 1.5", "epe_mean_nm: -1.0")
    with pytest.raises(ValidationError):
        _run_workflow(bad, store)


def test_yaml_python_tag_rejected(tmp_path: Path) -> None:
    """`yaml.safe_load` must refuse `!!python/object` tags. If it ever
    grows looser, this test catches it before the schema even runs."""
    store = _store(tmp_path)
    payload = _VALID_YAML + "\nnotes: !!python/object/apply:os.system ['ls']\n"
    with pytest.raises(yaml.YAMLError):
        _run_workflow(payload, store)


def test_submission_id_field_in_yaml_cannot_override(tmp_path: Path) -> None:
    """A submitter cannot pin their own submission_id. The schema accepts
    the field (it's used internally), but any value passed in is replaced
    by the tracker on `submit`."""
    store = _store(tmp_path)
    spoofed = _VALID_YAML + "\nsubmission_id: hijacked\n"
    sub_id = _run_workflow(spoofed, store)

    assert sub_id != "hijacked"
    entry = json.loads(store.path.read_text())["entries"][0]
    assert entry["submission_id"] == sub_id
    assert entry["submission_id"] != "hijacked"


def test_unknown_enum_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    bad = _VALID_YAML.replace("process_node: 7nm", "process_node: 999nm")
    with pytest.raises(ValidationError):
        _run_workflow(bad, store)


def test_committed_entry_is_strict_subset_of_schema(tmp_path: Path) -> None:
    """Defense in depth — even if validation passes, the on-disk JSON
    should only contain known schema keys plus `submission_id`."""
    store = _store(tmp_path)
    _run_workflow(_VALID_YAML, store)
    entry = json.loads(store.path.read_text())["entries"][0]

    allowed = set(BenchmarkResult.model_fields.keys()) | {"submission_id"}
    assert set(entry.keys()) <= allowed, set(entry.keys()) - allowed
