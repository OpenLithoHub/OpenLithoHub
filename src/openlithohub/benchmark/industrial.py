"""Industrial Benchmark v1 — shared schema, statistics, and claim math.

This module is the machine-readable contract behind the industrial
benchmark harness (``benchmarks/industrial/run_industrial_benchmark.py``)
and the automatic claim generator (``scripts/generate_industrial_claims.py``).

Design rules enforced here:

- Every benchmark artifact carries ``schema``, ``git_commit``, hardware,
  software, dataset provenance, and a ``claim_scope`` block.  A number
  without provenance cannot become a claim.
- Speedups are medians-of-repeats ratios, never best-of-N.
- Memory reductions compare peak resident-set medians of the same run
  protocol (fresh worker process per timed run).
- Claim levels separate what was actually measured from what would need
  external evidence (:data:`REPRODUCED_INTERNAL` up to
  :data:`FOUNDRY_CALIBRATED`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess  # noqa: S404 - only fixed-argv probes of the local host
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_NAME = "OpenLithoHub.industrial-benchmark.v1"

# Explicit outcome vocabulary: benchmark stages must never silently map
# "could not run" onto SUCCESS.
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"
STATUS_UNSUPPORTED = "UNSUPPORTED"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"
# A dense row the harness declined to run because the memory *policy*
# forbids it — a protocol decision, not an impossibility proof.
STATUS_NOT_RUN_MEMORY_POLICY = "NOT_RUN_MEMORY_POLICY"
# Dense raster cannot fit the *reference machine's* physical RAM
# (machine-relative arithmetic, e.g. the 1.23 TB full-die raster vs 48 GiB).
STATUS_INFEASIBLE_ON_REFERENCE_MACHINE = "INFEASIBLE_ON_REFERENCE_MACHINE"
# Reserved for genuinely size-independent impossibilities (none currently
# emitted); kept distinct so policy and machine limits are never laundered
# into "structurally impossible".
STATUS_INFEASIBLE_STRUCTURAL = "INFEASIBLE_STRUCTURAL"
KNOWN_STATUSES = (
    STATUS_SUCCESS,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_UNSUPPORTED,
    STATUS_INCONCLUSIVE,
    STATUS_NOT_RUN_MEMORY_POLICY,
    STATUS_INFEASIBLE_ON_REFERENCE_MACHINE,
    STATUS_INFEASIBLE_STRUCTURAL,
)

# Claim provenance levels, weakest first.  Everything produced by this
# repository's own harness on documented hardware is REPRODUCED_INTERNAL;
# the stronger levels require evidence this repo cannot self-certify.
REPRODUCED_INTERNAL = "REPRODUCED_INTERNAL"
PUBLIC_BENCHMARK = "PUBLIC_BENCHMARK"
THIRD_PARTY_REPRODUCED = "THIRD_PARTY_REPRODUCED"
FOUNDRY_CALIBRATED = "FOUNDRY_CALIBRATED"
CLAIM_LEVELS = (REPRODUCED_INTERNAL, PUBLIC_BENCHMARK, THIRD_PARTY_REPRODUCED, FOUNDRY_CALIBRATED)

_REQUIRED_SECTIONS = (
    "schema",
    "kind",
    "git_commit",
    "timestamp_utc",
    "hardware",
    "software",
    "claim_scope",
    "reproducibility",
    "measurement_source",
    "fixture",
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file, streamed (fixtures can be tens of MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize(obj: Any) -> Any:
    """Recursively map non-finite floats to None (strict-JSON contract)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _find_non_finite(obj: Any, path: str = "") -> list[str]:
    """Paths of any non-finite float in a parsed artifact (fail-closed)."""
    bad: list[str] = []
    if isinstance(obj, float) and not math.isfinite(obj):
        bad.append(path or "<root>")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(_find_non_finite(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad.extend(_find_non_finite(v, f"{path}[{i}]"))
    return bad


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100] (numpy convention)."""
    if not values:
        raise ValueError("percentile() of empty sequence")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Median / p10 / p90 summary over repeats — never best-of-N."""
    return {
        "n": len(values),
        "median": percentile(values, 50.0),
        "p10": percentile(values, 10.0),
        "p90": percentile(values, 90.0),
        "min": min(values),
        "max": max(values),
    }


def speedup(baseline_median_s: float, candidate_median_s: float) -> float:
    """speedup = baseline_median / candidate_median (spec §4)."""
    if candidate_median_s <= 0.0:
        raise ValueError(f"candidate median must be positive, got {candidate_median_s}")
    return baseline_median_s / candidate_median_s


def memory_reduction_pct(baseline_bytes: int, candidate_bytes: int) -> float:
    """Percentage peak-memory reduction, baseline vs candidate."""
    if baseline_bytes <= 0:
        raise ValueError(f"baseline bytes must be positive, got {baseline_bytes}")
    return (1.0 - candidate_bytes / baseline_bytes) * 100.0


def relative_reduction_pct(baseline: float, candidate: float) -> float:
    """Percentage reduction of a quality metric (lower is better)."""
    if baseline == 0.0:
        return 0.0
    return (baseline - candidate) / baseline * 100.0


def git_commit(repo_root: Path | None = None) -> str | None:
    """HEAD commit of the measuring checkout; None when undeterminable.

    Returns None (never a placeholder string) so downstream validation
    cannot mistake an unknown checkout for a 40-hex commit.
    """
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


_FULL_HEX_RE = re.compile(r"^[0-9a-f]{40}$")


def is_full_commit(value: str | None) -> bool:
    """True when value is a full 40-character lowercase hex commit."""
    return value is not None and _FULL_HEX_RE.match(value) is not None


def working_tree_clean(repo_root: Path | None = None) -> bool:
    """True when no *tracked* file is modified (untracked scratch ignored)."""
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.stdout.strip() == ""


def file_sha256_relative(repo_root: Path, relative: str) -> str | None:
    """SHA-256 of a repo file at measurement time; None when absent."""
    path = repo_root / relative
    if not path.exists():
        return None
    return sha256_file(path)


def measurement_source(repo_root: Path | None = None) -> dict[str, Any]:
    """Exact provenance of the source that produced a measurement.

    Binds an artifact to the committed source tree that produced it so
    the measurement can be reproduced bit-for-bit from a clean checkout.
    """
    root = repo_root or Path(__file__).resolve().parents[3]
    commit = git_commit(root)
    return {
        "commit": commit,
        "commit_valid": is_full_commit(commit),
        "working_tree_dirty": not working_tree_clean(root),
        "harness_path": "benchmarks/industrial/run_industrial_benchmark.py",
        "harness_sha256": file_sha256_relative(
            root, "benchmarks/industrial/run_industrial_benchmark.py"
        ),
        "industrial_core_sha256": file_sha256_relative(
            root, "src/openlithohub/benchmark/industrial.py"
        ),
        "claim_generator_sha256": file_sha256_relative(
            root, "scripts/generate_industrial_claims.py"
        ),
    }


def _cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _physical_ram_bytes() -> int | None:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return int(out.stdout.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def environment_snapshot(device: str = "cpu") -> dict[str, Any]:
    """Hardware + software identity of the measuring machine."""
    hardware: dict[str, Any] = {
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "physical_ram_bytes": _physical_ram_bytes(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "gpu": None,
    }
    software: dict[str, Any] = {
        "python": platform.python_version(),
        "openlithohub_commit": git_commit(),
    }
    try:
        import torch

        software["torch"] = torch.__version__
        software["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        software["torch"] = None
        software["cuda_available"] = False
    try:
        from openlithohub._version import __version__

        software["openlithohub"] = __version__
    except Exception:  # pragma: no cover - version lookup is best-effort
        software["openlithohub"] = None
    try:
        import klayout.db as _kdb  # noqa: F401

        software["klayout"] = getattr(_kdb, "__version__", "present")
    except ImportError:  # pragma: no cover
        software["klayout"] = None
    software["device"] = device
    return {"hardware": hardware, "software": software}


def artifact_sha256(path: str | os.PathLike[str]) -> str:
    """Alias kept for artifact-manifest readability."""
    return sha256_file(path)


@dataclass
class Claim:
    """One public-facing claim bound to its measured provenance."""

    claim_id: str
    metric: str
    value: float | str
    unit: str
    baseline: str
    dataset: str
    hardware: str
    scope: str
    level: str
    artifact: str
    artifact_sha256: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "baseline": self.baseline,
            "dataset": self.dataset,
            "hardware": self.hardware,
            "scope": self.scope,
            "level": self.level,
            "artifact": self.artifact,
            "artifact_sha256": self.artifact_sha256,
            "context": self.context,
        }


def build_claim(
    *,
    claim_id: str,
    metric: str,
    value: float | str,
    unit: str,
    baseline: str,
    dataset: str,
    hardware: str,
    scope: str,
    artifact: str,
    artifact_sha256: str = "",
    level: str = REPRODUCED_INTERNAL,
    context: dict[str, Any] | None = None,
) -> Claim:
    """Build a claim, rejecting unauditable provenance up front."""
    if level not in CLAIM_LEVELS:
        raise ValueError(f"unknown claim level {level!r}")
    if not dataset or not hardware or not scope:
        raise ValueError(f"claim {claim_id!r} needs dataset, hardware and scope")
    return Claim(
        claim_id=claim_id,
        metric=metric,
        value=value,
        unit=unit,
        baseline=baseline,
        dataset=dataset,
        hardware=hardware,
        scope=scope,
        level=level,
        artifact=artifact,
        artifact_sha256=artifact_sha256,
        context=context or {},
    )


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    """Structural validation of an industrial benchmark artifact.

    Returns a list of human-readable problems; empty means valid.
    Fail-closed: artifacts that do not validate must not feed claims.
    """
    problems: list[str] = []
    if not isinstance(artifact, dict):
        return ["artifact is not a JSON object"]
    for section in _REQUIRED_SECTIONS:
        if section not in artifact:
            problems.append(f"missing required section: {section}")
    if artifact.get("schema") != SCHEMA_NAME:
        problems.append(f"schema mismatch: expected {SCHEMA_NAME}, got {artifact.get('schema')!r}")
    kind = artifact.get("kind")
    if not isinstance(kind, str) or not kind:
        problems.append("missing or invalid 'kind'")
    status = artifact.get("status")
    if status is not None and status not in KNOWN_STATUSES:
        problems.append(f"unknown status {status!r} (must be one of {KNOWN_STATUSES})")
    commit = artifact.get("git_commit")
    if commit is not None and not is_full_commit(commit):
        problems.append(
            "git_commit must be a full 40-character lowercase hex commit "
            "(placeholders like UNKNOWN are not auditable)"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("run_identity") or "")):
        problems.append("run_identity must be a 64-hex sha256 (missing or malformed)")
    env_lock = artifact.get("environment_lock")
    if not isinstance(env_lock, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(env_lock.get("lock_sha256") or "")
    ):
        problems.append("environment_lock.lock_sha256 must be a 64-hex sha256")
    source = artifact.get("measurement_source")
    if isinstance(source, dict):
        if not is_full_commit(source.get("commit")):
            problems.append("measurement_source.commit must be a full 40-hex commit")
        if source.get("working_tree_dirty") is True:
            problems.append(
                "measurement_source.working_tree_dirty must be false: artifacts "
                "must be produced from a clean committed checkout"
            )
        for key in (
            "harness_sha256",
            "industrial_core_sha256",
            "claim_generator_sha256",
            "run_support_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(source.get(key) or "")):
                problems.append(f"measurement_source.{key} must be a 64-hex sha256")
    fixture = artifact.get("fixture")
    fixture_hash = str(fixture.get("sha256", "")) if isinstance(fixture, dict) else ""
    if isinstance(fixture, dict) and not re.fullmatch(r"[0-9a-f]{64}", fixture_hash):
        problems.append("fixture.sha256 must be a 64-hex sha256")
    non_finite = _find_non_finite(artifact)
    if non_finite:
        problems.append(f"non-finite floats are not strict JSON: {non_finite[:5]}")
    claim_scope = artifact.get("claim_scope")
    if (
        isinstance(artifact, dict)
        and "claim_scope" in artifact
        and not isinstance(claim_scope, dict)
    ):
        problems.append("claim_scope must be an object")
    return problems


def _reject_constant(token: str) -> float:
    raise ValueError(f"non-finite JSON constant {token!r} is not allowed in artifacts")


_FAMILY_FIELDS = ("run_identity", "git_commit")


def validate_artifact_family(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    """Cross-artifact closure: every member of one published family must
    come from the SAME run (audit B0.4).

    ``artifacts`` maps a family role ("runtime", "quality", ...) to its
    validated artifact dict.  Checks shared identity fields (run_identity,
    git_commit, fixture hash, environment lock) and that the family set is
    exactly the expected one — a family assembled from two runs, or a
    partial family, is rejected.
    """
    problems: list[str] = []
    expected_roles = {"runtime", "quality", "fulldie", "run-config"}
    missing = expected_roles - set(artifacts)
    if missing:
        problems.append(f"incomplete family: missing {sorted(missing)}")
    roles = ["runtime", "quality", "fulldie", "run-config"]

    def _value(artifact: dict[str, Any], kind: str) -> str:
        # The run-config stores the same facts under its own layout.
        if kind == "commit":
            from_source = (artifact.get("source") or {}).get("commit") or ""
            return str(artifact.get("git_commit") or from_source)
        if kind == "fixture":
            return str(
                (artifact.get("fixture") or {}).get("sha256")
                or (artifact.get("fixtures") or {}).get("parent_gds_sha256")
                or ""
            )
        if kind == "lock":
            return str((artifact.get("environment_lock") or {}).get("lock_sha256") or "")
        return str(artifact.get(kind) or "")

    for kind in ("run_identity", "commit", "fixture", "lock"):
        values = {role: _value(artifacts[role], kind) for role in roles if role in artifacts}
        present = {r: v for r, v in values.items() if v and v != "None"}
        if len(present) >= 2 and len(set(present.values())) > 1:
            problems.append(f"family {kind} mismatch: {present}")
    return problems


RUN_CONFIG_SCHEMA = "OpenLithoHub.industrial-run-config.v1"


def build_run_identity_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Canonical identity payload FROM a published run-config (P0.3).

    Mirrors the harness's identity payload exactly: source commit + file
    hashes, environment-lock hash, fixture hashes and every semantic
    argument.  The verifier re-hashes this and compares with
    ``config["run_identity"]`` — turning the identity from a label into
    content-addressed authority.
    """
    source = config.get("source") or {}
    fixtures = config.get("fixtures") or {}
    rci = config.get("runtime_code_identity")
    if not isinstance(rci, dict):
        raise ValueError("config must include runtime_code_identity block")
    return {
        "schema": "OpenLithoHub.industrial-run-identity.v1",
        "source": {
            "commit": source.get("commit"),
            "harness_sha256": source.get("harness_sha256"),
            "industrial_core_sha256": source.get("industrial_core_sha256"),
            "claim_generator_sha256": source.get("claim_generator_sha256"),
            "run_support_sha256": source.get("run_support_sha256"),
        },
        "environment_lock_sha256": (config.get("environment_lock") or {}).get("lock_sha256"),
        "fixtures": {
            "parent_gds_sha256": fixtures.get("parent_gds_sha256"),
            "iccad": fixtures.get("iccad") or {},
        },
        "runtime_code_identity": {
            "mode": rci.get("mode"),
            "measurement_commit": rci.get("measurement_commit"),
            "source_tree_match": rci.get("source_tree_match"),
        },
        "args": config.get("args") or {},
    }


def recompute_run_identity(config: dict[str, Any]) -> str:
    """Recompute the run identity from a published run-config."""
    return hashlib.sha256(_canonical_dumps(build_run_identity_payload(config)).encode()).hexdigest()


def validate_run_config(config: dict[str, Any]) -> list[str]:
    """Structural + semantic validation of industrial-run-config.json."""
    problems: list[str] = []
    if config.get("schema") != RUN_CONFIG_SCHEMA:
        problems.append(f"schema must be {RUN_CONFIG_SCHEMA}, got {config.get('schema')!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(config.get("run_identity") or "")):
        problems.append("run_identity must be a 64-hex sha256")
    source = config.get("source") or {}
    if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit") or "")):
        problems.append("source.commit must be a full 40-hex commit")
    for key in (
        "harness_sha256",
        "industrial_core_sha256",
        "claim_generator_sha256",
        "run_support_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(source.get(key) or "")):
            problems.append(f"source.{key} must be a 64-hex sha256")
    fixtures = config.get("fixtures") or {}
    if not re.fullmatch(r"[0-9a-f]{64}", str(fixtures.get("parent_gds_sha256") or "")):
        problems.append("fixtures.parent_gds_sha256 must be a 64-hex sha256")
    problems.extend(validate_environment_lock(config.get("environment_lock") or {}))
    if not isinstance(config.get("args"), dict) or not config["args"]:
        problems.append("args must be a non-empty object of semantic arguments")
    # P0-3: runtime_code_identity block must be present and valid
    rci = config.get("runtime_code_identity")
    if not isinstance(rci, dict):
        problems.append("runtime_code_identity must be an object")
        return problems
    if rci.get("mode") != "source-tree":
        problems.append(
            f"runtime_code_identity.mode must be 'source-tree', got {rci.get('mode')!r}"
        )
    source_commit = str((config.get("source") or {}).get("commit") or "")
    if rci.get("measurement_commit") != source_commit:
        problems.append(
            f"runtime_code_identity.measurement_commit must equal source.commit ({source_commit})"
        )
    if rci.get("source_tree_match") is not True:
        problems.append("runtime_code_identity.source_tree_match must be true")
    # P0.3: the identity must be the content-addressed digest of exactly
    # this configuration.  recompute_run_identity raises ValueError if
    # the runtime_code_identity block is missing — but we've checked above.
    recomputed = recompute_run_identity(config)
    if recomputed != str(config.get("run_identity")):
        problems.append(
            "run_identity does not match the recomputed digest of the "
            "published configuration (identity is not content-addressed)"
        )
    return problems


def _canonical_dumps(payload: dict[str, Any]) -> str:
    """Canonical JSON serialization used for every recomputed digest.

    Identical to run_support.strict_dumps: sort_keys + allow_nan=False.
    """
    return json.dumps(payload, sort_keys=True, allow_nan=False)


def validate_environment_lock(lock: dict[str, Any]) -> list[str]:
    """Recompute the environment-lock hash over the lock WITHOUT
    lock_sha256 and compare (audit P0.4)."""
    problems: list[str] = []
    if not isinstance(lock, dict) or not lock:
        return ["environment_lock missing or empty"]
    stored = str(lock.get("lock_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", stored):
        problems.append("environment_lock.lock_sha256 must be a 64-hex sha256")
        return problems
    lock_body = {k: v for k, v in lock.items() if k != "lock_sha256"}
    recomputed = hashlib.sha256(_canonical_dumps(lock_body).encode()).hexdigest()
    if recomputed != stored:
        problems.append(
            "environment_lock.lock_sha256 does not match the recomputed hash of the lock body"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(lock.get("distribution_freeze_sha256") or "")):
        problems.append("environment_lock.distribution_freeze_sha256 must be a 64-hex sha256")
    return problems


def load_artifact(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate one artifact JSON; raises on structural failure.

    Parsing itself rejects NaN/Infinity tokens — artifacts are strict JSON.
    """
    with open(path, encoding="utf-8") as handle:
        artifact: dict[str, Any] = json.load(handle, parse_constant=_reject_constant)
    problems = validate_artifact(artifact)
    if problems:
        raise ValueError(f"{path}: invalid industrial benchmark artifact: {problems}")
    return artifact


def compare_runtime(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_name: str,
    candidate_name: str,
    metric_key: str = "execution_wall_s",
) -> dict[str, Any]:
    """Median-based runtime comparison between two recorded rows.

    Both rows must carry ``{metric_key: {median, p10, p90, n}}`` summaries.
    """
    base = baseline[metric_key]
    cand = candidate[metric_key]
    ratio = speedup(float(base["median"]), float(cand["median"]))
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "baseline_median_s": float(base["median"]),
        "candidate_median_s": float(cand["median"]),
        "speedup": ratio,
        "metric": metric_key,
    }
