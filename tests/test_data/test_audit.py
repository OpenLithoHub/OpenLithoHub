"""Tests for the OPENLITHOHUB_AUDIT_DIR download-audit hook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openlithohub.data.base import DatasetAdapter, LithoSample


class _SuccessAdapter(DatasetAdapter):
    """Minimal concrete adapter — download() touches the filesystem so the
    size_bytes recorder has something to count."""

    def __len__(self) -> int:  # pragma: no cover — not under test here
        return 0

    def __getitem__(self, index: int) -> LithoSample:  # pragma: no cover
        raise IndexError(index)

    def download(self, root: str) -> None:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "marker.bin").write_bytes(b"hello")


class _FailingAdapter(DatasetAdapter):
    def __len__(self) -> int:  # pragma: no cover
        return 0

    def __getitem__(self, index: int) -> LithoSample:  # pragma: no cover
        raise IndexError(index)

    def download(self, root: str) -> None:
        raise RuntimeError("upstream offline")


def test_audit_writes_success_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_dir = tmp_path / "audit"
    monkeypatch.setenv("OPENLITHOHUB_AUDIT_DIR", str(audit_dir))

    dest = tmp_path / "data"
    _SuccessAdapter().download(str(dest))

    records = _read_records(audit_dir / "_SuccessAdapter.jsonl")
    assert len(records) == 1
    rec = records[0]
    assert rec["outcome"] == "success"
    assert rec["adapter"].endswith("_SuccessAdapter")
    assert rec["root"] == str(dest)
    assert rec["size_bytes"] >= len(b"hello")
    assert "timestamp" in rec
    assert "elapsed_ms" in rec


def test_audit_writes_error_record_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audit"
    monkeypatch.setenv("OPENLITHOHUB_AUDIT_DIR", str(audit_dir))

    with pytest.raises(RuntimeError, match="upstream offline"):
        _FailingAdapter().download(str(tmp_path / "data"))

    records = _read_records(audit_dir / "_FailingAdapter.jsonl")
    assert len(records) == 1
    rec = records[0]
    assert rec["outcome"] == "error"
    assert rec["error_class"] == "RuntimeError"
    assert "upstream offline" in rec["error_message"]


def test_audit_noop_when_env_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENLITHOHUB_AUDIT_DIR", raising=False)
    _SuccessAdapter().download(str(tmp_path / "data"))
    # Nothing should have been written anywhere we can check trivially —
    # but the import side-effect of :func:`_audit_dir` returning None is
    # the contract under test. The download itself succeeded silently.
    assert not (tmp_path / "_SuccessAdapter.jsonl").exists()


def _read_records(path: Path) -> list[dict]:
    assert path.exists(), f"expected audit JSONL at {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
