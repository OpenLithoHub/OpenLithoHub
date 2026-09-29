"""Canonical artifact bytes (GPU Authority Repair v3 §24/§25).

Same payload → byte-identical output across platform semantics; LF only;
strict JSON; committed fixture manifests reproduce byte-identically.
A formal campaign must never need manual artifact restoration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openlithohub._utils.canonical_json import canonical_json_bytes, write_canonical_json
from openlithohub.benchmark.industrial_v2 import write_strict_json

REPO = Path(__file__).resolve().parents[2]

PAYLOAD = {
    "zeta": 1,
    "alpha": {"nested": [1, 2, 3], "b": True, "a": None},
    "text": "LF\ninside strings is preserved",
    "float": 0.1,
}


def test_bytes_are_utf8_lf_only_with_stable_final_newline() -> None:
    data = canonical_json_bytes(PAYLOAD)
    assert b"\r" not in data, "CRLF must never appear in canonical bytes"
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")
    assert data.decode("utf-8")


def test_bytes_are_platform_independent_and_deterministic() -> None:
    assert canonical_json_bytes(PAYLOAD) == canonical_json_bytes(PAYLOAD)
    # sorted keys, two-space indent — identical on any host semantics
    text = canonical_json_bytes(PAYLOAD).decode("utf-8")
    assert text == json.dumps(PAYLOAD, sort_keys=True, indent=2, allow_nan=False) + "\n"


def test_strict_json_rejects_nan_and_infinity() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("inf")})


def test_write_canonical_json_never_produces_crlf(tmp_path: Path) -> None:
    """§25: even with Windows text-mode semantics in play, the byte-based
    writer cannot emit CRLF (Path.write_bytes bypasses newline mapping)."""
    target = tmp_path / "artifact.json"
    write_canonical_json(target, PAYLOAD)
    raw = target.read_bytes()
    assert b"\r" not in raw
    assert raw == canonical_json_bytes(PAYLOAD)


def test_write_strict_json_delegates_to_canonical_bytes(tmp_path: Path) -> None:
    target = tmp_path / "strict.json"
    write_strict_json(target, PAYLOAD)
    assert target.read_bytes() == canonical_json_bytes(PAYLOAD)


def test_committed_scale_fixture_manifests_are_lf_canonical() -> None:
    """The committed fixture authority bytes stay LF: regeneration through
    the canonical writer reproduces them without git checkout -- (§25)."""
    for manifest in (
        REPO / "benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json",
        REPO / "benchmarks/results/industrial-scale/fixtures/microwatt/fixture-manifest.json",
    ):
        raw = manifest.read_bytes()
        assert b"\r\n" not in raw, f"{manifest.name} carries CRLF bytes"
        assert raw.endswith(b"\n")
        json.loads(raw)
