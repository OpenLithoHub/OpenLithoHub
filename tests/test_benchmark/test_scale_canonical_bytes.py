"""Scale canonical artifact bytes (PR-S8, GPU Authority Repair v3 §24/§25).

The Scale family adopts the shared canonical byte-based strict-JSON
writer: same payload → byte-identical output across platform semantics;
LF only; strict JSON; committed fixture manifests reproduce
byte-identically; fixture prep leaves the tracked tree clean with no
manual ``git checkout --`` restoration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openlithohub._utils.canonical_json import canonical_json_bytes
from openlithohub.benchmark.industrial_scale import write_strict_json

REPO = Path(__file__).resolve().parents[2]

FIXTURE_MANIFESTS = (
    REPO / "benchmarks/results/industrial-scale/fixtures/ibex/fixture-manifest.json",
    REPO / "benchmarks/results/industrial-scale/fixtures/microwatt/fixture-manifest.json",
)

PAYLOAD = {
    "schema": "OpenLithoHub.industrial-scale-test.v1",
    "window": 8192,
    "lane": "C_SINGLE_GPU_SATURATION",
    "microbatch_ladder": [1, 2, 4, 8, 16, 32],
    "selected_layer": "66:44",
}


def test_scale_writer_produces_canonical_lf_bytes(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    write_strict_json(str(target), PAYLOAD)
    raw = target.read_bytes()
    assert b"\r" not in raw, "CRLF must never appear in Scale artifact bytes"
    assert raw == canonical_json_bytes(PAYLOAD)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")


def test_scale_writer_is_strict_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_strict_json(str(tmp_path / "bad.json"), {"bad": float("nan")})


def test_scale_writer_matches_the_v2_canonical_writer(tmp_path: Path) -> None:
    """One canonical writer across both formal families — V2 and Scale
    artifacts share the exact byte contract (§24)."""
    from openlithohub.benchmark.industrial_v2 import write_strict_json as v2_writer

    scale_target = tmp_path / "scale.json"
    v2_target = tmp_path / "v2.json"
    write_strict_json(str(scale_target), PAYLOAD)
    v2_writer(v2_target, PAYLOAD)
    assert scale_target.read_bytes() == v2_target.read_bytes()


@pytest.mark.skipif(
    not any(manifest.is_file() for manifest in FIXTURE_MANIFESTS),
    reason="scale fixture manifests are host-local (gitignored)",
)
def test_committed_fixture_manifests_regenerate_byte_identically() -> None:
    """§25: fixture regeneration through the canonical writer reproduces
    the committed manifest bytes exactly — no CRLF drift, no manual
    restoration, tracked tree stays clean."""
    for manifest in FIXTURE_MANIFESTS:
        if not manifest.is_file():
            continue
        raw = manifest.read_bytes()
        assert b"\r\n" not in raw, f"{manifest.parent.parent.name} manifest carries CRLF"
        payload = json.loads(raw)
        assert canonical_json_bytes(payload) == raw, (
            f"{manifest} would not regenerate byte-identically through the "
            "canonical writer (§25 hostile: fixture authority drift)"
        )


def test_scale_json_writer_adoption_is_source_level() -> None:
    """The adoption is at the single write site (industrial_scale module),
    so every Scale consumer — fixture prep, verifiers, campaign builder —
    inherits the canonical bytes without individual changes."""
    import inspect

    from openlithohub.benchmark import industrial_scale

    source = inspect.getsource(industrial_scale.write_strict_json)
    assert "write_canonical_json" in source
    assert "write_text" not in source, "text-mode JSON writes must be gone (CRLF risk)"
