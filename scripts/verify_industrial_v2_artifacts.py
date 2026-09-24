"""Industrial Benchmark v2 artifact verifier (PR-G §28).

Closes the canonical v2 family: strict JSON, known schema, exact family
membership, SHA256SUMS closure, manifest closure, run-identity
recomputation, environment-lock completeness, GPU tier admission rules
(synchronized timing, correctness witness), and claim linkage.  Fails
closed on unknown files in the canonical root — a partial or augmented
authority can never pass.

Usage::

    python scripts/verify_industrial_v2_artifacts.py \\
        [--canonical-root benchmarks/results/industrial-v2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openlithohub.benchmark.industrial_v2 import (  # noqa: E402
    CANONICAL_FAMILY,
    SCHEMA_NAME,
    compute_run_identity_v2,
)

GPU_LOCK_REQUIRED_KEYS = (
    "available",
    "count",
    "devices",
    "torch_cuda_version",
    "torch_version",
    "tf32_matmul",
    "tf32_cudnn",
)
GPU_ROW_REQUIRED_KEYS = (
    "device",
    "dtype",
    "timing_method",
    "max_memory_allocated",
    "max_memory_reserved",
    "correctness_witness_pass",
    "status",
)
KNOWN_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class VerifyError(Exception):
    pass


def _fail(message: str) -> None:
    raise VerifyError(message)


def _load_strict_json(path: Path) -> dict:
    text = path.read_text()
    for token in ("NaN", "Infinity", "-Infinity"):
        if re.search(rf"(?<!\")\b{token}\b(?!\")", text):
            _fail(f"{path.name}: non-strict JSON token {token!r} (B2: strict JSON only)")
    data = json.loads(text)
    if not isinstance(data, dict):
        _fail(f"{path.name}: top-level JSON must be an object")
    return data


def verify(canonical_root: Path) -> list[str]:
    """Run every closure check; returns the list of verified claims.
    Raises :class:`VerifyFailure` on the first hard failure."""
    if not canonical_root.is_dir():
        _fail(f"canonical root missing: {canonical_root}")

    present = {p.name for p in canonical_root.iterdir() if p.is_file()}
    unknown = sorted(present - CANONICAL_FAMILY)
    if unknown:
        _fail(f"unknown file(s) in canonical root (fail closed): {unknown}")
    missing = sorted(CANONICAL_FAMILY - present)
    if missing:
        _fail(f"canonical family incomplete, missing: {missing} (all-or-nothing)")

    run_config = _load_strict_json(canonical_root / "industrial-v2-run-config.json")
    if run_config.get("schema") != "OpenLithoHub.industrial-run-config.v2":
        _fail("run-config schema is not OpenLithoHub.industrial-run-config.v2")
    if not run_config.get("tracked_tree_clean"):
        _fail("run-config records a dirty tracked tree (B2-A)")
    if run_config.get("provisional"):
        _fail("provisional run cannot be promoted to the canonical family")

    # SHA256SUMS closure over the exact family
    sums: dict[str, str] = {}
    for line in (canonical_root / "SHA256SUMS.txt").read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        name = name.strip()
        if name.startswith("/") or ".." in name:
            _fail(f"SHA256SUMS contains unsafe path {name!r}")
        if name in sums:
            _fail(f"SHA256SUMS has duplicate entry for {name!r}")
        sums[name] = digest
    if set(sums) != (CANONICAL_FAMILY - {"SHA256SUMS.txt"}):
        _fail("SHA256SUMS set != canonical member set (stale or incomplete)")
    for name, digest in sorted(sums.items()):
        actual = hashlib.sha256((canonical_root / name).read_bytes()).hexdigest()
        if actual != digest:
            _fail(f"SHA256 mismatch for {name!r}")
        if not KNOWN_HASH_RE.fullmatch(digest):
            _fail(f"malformed digest for {name!r}")

    manifest = _load_strict_json(canonical_root / "manifest.json")
    members = manifest.get("members")
    if not isinstance(members, list):
        _fail("manifest has no members list")
    manifest_names = {entry.get("name") for entry in members}
    # manifest.json cannot contain its own byte count: it lists every
    # canonical member except itself and SHA256SUMS.txt.
    if manifest_names != CANONICAL_FAMILY - {"manifest.json", "SHA256SUMS.txt"}:
        _fail("manifest membership != canonical member set")
    for entry in members:
        if int(entry.get("bytes", -1)) != (canonical_root / entry["name"]).stat().st_size:
            _fail(f"manifest byte count mismatch for {entry['name']!r}")

    # run identity recomputation (B2-C: any semantic change alters it)
    run_config_payload = run_config.get("run_config")
    source_hashes = run_config.get("source_hashes")
    if not isinstance(run_config_payload, dict) or not isinstance(source_hashes, dict):
        _fail("run-config missing run_config/source_hashes")
    from openlithohub.benchmark.industrial_v2 import RunConfigV2

    known_fields = set(RunConfigV2.__dataclass_fields__.keys())
    unknown_config_fields = set(run_config_payload) - known_fields
    if unknown_config_fields:
        _fail(f"run config carries unknown semantic fields: {sorted(unknown_config_fields)}")
    recomputed = compute_run_identity_v2(
        RunConfigV2(
            **{
                **run_config_payload,
                "tiers": tuple(run_config_payload.get("tiers", [])),
                "window_sizes": tuple(run_config_payload.get("window_sizes", [])),
            }
        ),
        measurement_commit=run_config["measurement_commit"],
        harness_sha256=source_hashes["harness"],
        core_sha256=source_hashes["core"],
        claim_generator_sha256=source_hashes["claim_generator"],
        verifier_sha256=source_hashes["verifier"],
    )
    recorded_identity = run_config.get("run_identity")
    recorded_identity = recorded_identity or manifest.get("run_identity")
    if recorded_identity != recomputed:
        _fail(
            f"run identity drift: recorded {recorded_identity!r} != recomputed {recomputed!r}"
        )

    # environment lock completeness (§9, B2-D).  The freeze file carries
    # the full environment record with the gpu lock nested inside.
    freeze = _load_strict_json(canonical_root / "industrial-v2-distribution-freeze.txt")
    env_lock = freeze.get("gpu") if isinstance(freeze.get("gpu"), dict) else None
    if env_lock is None:
        _fail("distribution freeze has no gpu environment lock")
    for key in GPU_LOCK_REQUIRED_KEYS:
        if key not in env_lock:
            _fail(f"GPU environment lock missing required key {key!r} (B2-D)")

    # tiers: correctness precedes performance; GPU rows fully locked (§16-18)
    claims: list[str] = []
    for member, tier_key in (
        ("industrial-v2-index.json", "a"),
        ("industrial-v2-gpu-runtime.json", "b"),
        ("industrial-v2-hopkins.json", "c"),
    ):
        payload = _load_strict_json(canonical_root / member)
        if payload.get("schema") != SCHEMA_NAME:
            _fail(f"{member}: wrong schema {payload.get('schema')!r}")
        if payload.get("run_identity") != recomputed:
            _fail(f"{member}: bound to a different run identity")
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            _fail(f"{member}: no measurement rows")
        if not payload.get("correctness_witness_pass"):
            _fail(f"{member}: correctness witness missing (B2-G)")
        for row in rows:
            if row.get("status") == "NOT_RUN_ENVIRONMENT":
                _fail(
                    f"{member}: NOT_RUN_ENVIRONMENT row in a CANONICAL family — "
                    "formal GPU measurement is missing (B2-D/§7)"
                )
            if row.get("device_requires_cuda") or str(row.get("device", "")).startswith(
                "cuda"
            ):
                for key in GPU_ROW_REQUIRED_KEYS:
                    if key not in row:
                        _fail(f"{member}: GPU row missing {key!r} (B2-D)")
                if row.get("timing_method") != "cuda_synchronized":
                    _fail(f"{member}: GPU row timing is not synchronized (B2-E)")
                if not row.get("correctness_witness_pass"):
                    _fail(f"{member}: GPU row failed correctness (B2-G)")
        if payload.get("correctness_witness_pass") and payload.get("claim_level"):
            claims.append(f"{tier_key}:{payload['claim_level']}")

    # cold/warm separation on hopkins (B2-F)
    hopkins = _load_strict_json(canonical_root / "industrial-v2-hopkins.json")
    if hopkins.get("cold_wall_s") is not None and "warm_wall_s" not in hopkins:
        _fail("hopkins reports cold timing without warm steady-state (B2-F)")

    return claims


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-root",
        default="benchmarks/results/industrial-v2",
        help="canonical v2 family root",
    )
    args = parser.parse_args()
    try:
        claims = verify(Path(args.canonical_root))
    except VerifyError as exc:
        print(f"V2 VERIFIER: FAIL — {exc}", file=sys.stderr)
        return 1
    print(f"V2 VERIFIER: PASS — canonical family closed; claims: {claims or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
