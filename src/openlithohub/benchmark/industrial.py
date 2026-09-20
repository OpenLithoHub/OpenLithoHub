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
import os
import platform
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
STATUS_INFEASIBLE_STRUCTURAL = "INFEASIBLE_STRUCTURAL"
KNOWN_STATUSES = (
    STATUS_SUCCESS,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_UNSUPPORTED,
    STATUS_INCONCLUSIVE,
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
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file, streamed (fixtures can be tens of MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def git_commit(repo_root: Path | None = None) -> str:
    """Best-effort HEAD commit of the measuring checkout."""
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
        return "UNKNOWN"
    return out.stdout.strip() or "UNKNOWN"


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
    if commit is not None and (not isinstance(commit, str) or len(commit) < 7):
        problems.append("git_commit must be a non-trivial string")
    claim_scope = artifact.get("claim_scope")
    if (
        isinstance(artifact, dict)
        and "claim_scope" in artifact
        and not isinstance(claim_scope, dict)
    ):
        problems.append("claim_scope must be an object")
    return problems


def load_artifact(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate one artifact JSON; raises on structural failure."""
    with open(path, encoding="utf-8") as handle:
        artifact: dict[str, Any] = json.load(handle)
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
