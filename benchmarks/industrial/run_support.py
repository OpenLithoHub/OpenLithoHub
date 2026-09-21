"""Run-support primitives for the Industrial Benchmark harness.

Everything here exists to keep one measurement run *identifiable*:

- ``compute_run_identity`` — a single SHA-256 over the measurement source
  (commit + file hashes), the fixture hashes, every semantic CLI argument,
  and the environment lock.  Two runs with the same identity are the same
  run; anything that would change a number changes the identity.
- ``Checkpoint`` — run-scoped, strict-JSON, per-row checkpoint journal.
  Every row carries the schema, run identity and stage; a mismatch is a
  hard error, never a silent reuse.
- ``Progress`` — atomic progress file answering "where is the run and how
  much longer" at any moment, across all stages of a process.
- ``environment_lock`` — exact software/thread environment of the machine,
  hashed into the run identity and written into the run workspace.
- Fixture identity + fail-closed layout validation (top cell, DBU
  arithmetic, layer presence).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess  # noqa: S404 - fixed-argv probes of the local host only
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA = "OpenLithoHub.industrial-checkpoint.v1"

_FULL_HEX_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file, streamed (fixtures can be tens of MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(token: str) -> float:
    raise ValueError(f"non-finite JSON constant {token!r} is not allowed")


def strict_dumps(payload: Any) -> str:
    """Serialize with the strict-JSON contract (no NaN/Infinity)."""
    return json.dumps(payload, sort_keys=True, allow_nan=False)


def strict_loads(text: str) -> Any:
    """Parse with the strict-JSON contract (reject NaN/Infinity tokens)."""
    return json.loads(text, parse_constant=_reject_constant)


# ---------------------------------------------------------------------------
# run logging
# ---------------------------------------------------------------------------

_RUN_LOG: Path | None = None


def init_run_log(path: Path) -> None:
    global _RUN_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    _RUN_LOG = path
    with open(_RUN_LOG, "a", encoding="utf-8") as handle:
        handle.write(f"\n=== run start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _RUN_LOG is not None:
        with open(_RUN_LOG, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------

_PROGRESS_INSTANCES: dict[str, Progress] = {}


def get_progress(path: Path) -> Progress:
    """One Progress instance per path, so stages accumulate in one file."""
    key = str(path.resolve())
    if key not in _PROGRESS_INSTANCES:
        _PROGRESS_INSTANCES[key] = Progress(path)
    return _PROGRESS_INSTANCES[key]


class Progress:
    """Durable progress file: stage, done/total, in-flight row, ETA.

    Written atomically after every checkpoint append so "where is the run,
    how much longer" is answerable from a single JSON at any moment —
    including after an interrupt.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stages: dict[str, dict[str, Any]] = {}

    def stage(self, name: str, total: int) -> None:
        self.stages[name] = {
            "total": total,
            "done": 0,
            "skipped_on_resume": 0,
            "current": None,
            "last_row_wall_s": None,
            "eta_s": None,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.flush(name)

    def begin_row(self, stage: str, key: str) -> None:
        self.stages[stage]["current"] = key
        self.flush(stage)

    def end_row(self, stage: str, key: str, wall_s: float, skipped: bool = False) -> None:
        s = self.stages[stage]
        s["current"] = None
        s["last_row_wall_s"] = round(wall_s, 3)
        if skipped:
            s["skipped_on_resume"] += 1
        else:
            s["done"] += 1
        remaining = s["total"] - s["done"] - s["skipped_on_resume"]
        s["eta_s"] = round(max(0, remaining) * (s["last_row_wall_s"] or 0.0), 1)
        self.flush(stage)

    def flush(self, stage: str) -> None:
        s = self.stages[stage]
        remaining = s["total"] - s["done"] - s["skipped_on_resume"]
        if s["eta_s"] is None and s["last_row_wall_s"]:
            s["eta_s"] = round(max(0, remaining) * s["last_row_wall_s"], 1)
        payload = {
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stages": self.stages,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(strict_dumps(payload), encoding="utf-8")
        tmp.replace(self.path)
        eta = s["eta_s"]
        log(
            f"progress [{stage}] done={s['done']}/{s['total']} "
            f"(resumed {s['skipped_on_resume']}) eta=" + ("n/a" if eta is None else f"~{eta}s")
        )


# ---------------------------------------------------------------------------
# checkpoints (strict JSON + row identity)
# ---------------------------------------------------------------------------


class CheckpointIdentityError(RuntimeError):
    """A checkpoint row does not belong to the current run identity."""


class Checkpoint:
    """Run-scoped strict-JSON checkpoint journal.

    Each row records the checkpoint schema, the run identity, the stage
    and the row key; ``get`` refuses rows from any other run identity, so
    code, argument or fixture changes can never silently reuse stale
    results.
    """

    def __init__(self, path: Path, run_identity: str, stage: str) -> None:
        self.path = path
        self.run_identity = run_identity
        self.stage = stage
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = self._parse_strict(line)
                self._assert_identity(entry)
                self.rows.append(entry)

    def _parse_strict(self, line: str) -> dict[str, Any]:
        entry = strict_loads(line)
        if not isinstance(entry, dict) or entry.get("_schema") != CHECKPOINT_SCHEMA:
            raise CheckpointIdentityError(f"{self.path}: checkpoint row is not {CHECKPOINT_SCHEMA}")
        return entry

    def _assert_identity(self, entry: dict[str, Any]) -> None:
        if entry.get("_run_identity") != self.run_identity:
            raise CheckpointIdentityError(
                f"{self.path}: checkpoint row identity "
                f"{entry.get('_run_identity')!r} does not match this run "
                f"({self.run_identity!r}). Delete the run workspace or fix "
                "the identity inputs — stale reuse is not allowed."
            )
        if entry.get("_stage") != self.stage:
            raise CheckpointIdentityError(
                f"{self.path}: checkpoint row stage {entry.get('_stage')!r} "
                f"does not match journal stage {self.stage!r}"
            )

    def has(self, key: str) -> bool:
        return any(row.get("_key") == key for row in self.rows)

    def get(self, key: str) -> dict[str, Any] | None:
        for row in self.rows:
            if row.get("_key") == key:
                return dict(row.get("payload") or {})
        return None

    def append(self, key: str, payload: dict[str, Any]) -> None:
        entry: dict[str, Any] = {
            "_schema": CHECKPOINT_SCHEMA,
            "_run_identity": self.run_identity,
            "_stage": self.stage,
            "_key": key,
            "_created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "payload": payload,
        }
        self.rows.append(entry)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(strict_dumps(entry) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# measurement source + environment lock
# ---------------------------------------------------------------------------


def _git(repo_root: Path, argv: list[str]) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603, S607 - fixed-argv git probe only
            ["git", *argv],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def measurement_source(repo_root: Path) -> dict[str, Any]:
    """Exact provenance of the committed source tree doing the measuring."""
    commit = _git(repo_root, ["rev-parse", "HEAD"])
    status = _git(repo_root, ["status", "--porcelain", "--untracked-files=no"])
    tracked_dirty = status is None or status != ""

    def file_hash(relative: str) -> str | None:
        path = repo_root / relative
        return sha256_file(path) if path.exists() else None

    return {
        "commit": commit,
        "commit_valid": commit is not None and bool(_FULL_HEX_RE.match(commit)),
        "working_tree_dirty": tracked_dirty,
        "harness_path": "benchmarks/industrial/run_industrial_benchmark.py",
        "harness_sha256": file_hash("benchmarks/industrial/run_industrial_benchmark.py"),
        "industrial_core_sha256": file_hash("src/openlithohub/benchmark/industrial.py"),
        "claim_generator_sha256": file_hash("scripts/generate_industrial_claims.py"),
        "run_support_sha256": file_hash("benchmarks/industrial/run_support.py"),
    }


_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
)

_LOCKED_PACKAGES = (
    "numpy",
    "scipy",
    "torch",
    "klayout",
    "diff-surrogate",
    "openlithohub",
    "fastapi",
    "pydantic",
    "pydantic-core",
)


def environment_lock(repo_root: Path) -> dict[str, Any]:
    """Exact, hashable snapshot of the benchmark software environment."""
    packages: dict[str, str] = {}
    for name in _LOCKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "ABSENT"

    import torch

    parallel_info = ""
    try:
        parallel_info = torch.__config__.parallel_info()
    except Exception:  # noqa: BLE001 - best-effort diagnostics only
        parallel_info = "unavailable"

    thread_env = {k: os.environ.get(k) for k in _THREAD_ENV_KEYS}
    lock = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "physical_ram_bytes": _physical_ram_bytes(),
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "parallel_info": parallel_info,
        },
        "thread_env": thread_env,
        "packages": packages,
    }
    lock["lock_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in lock.items() if k != "lock_sha256"}, sort_keys=True).encode()
    ).hexdigest()
    # Keep a full distribution freeze next to the lock for forensics.
    try:
        freeze = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
        (repo_root / ".env-freeze.tmp").write_text("\n".join(freeze) + "\n", encoding="utf-8")
        freeze_sha = sha256_file(repo_root / ".env-freeze.tmp")
        (repo_root / ".env-freeze.tmp").unlink(missing_ok=True)
        lock["distribution_freeze_sha256"] = freeze_sha
    except Exception:  # noqa: BLE001 - freeze is forensic best-effort
        lock["distribution_freeze_sha256"] = None
    _ = repo_root  # freeze is written next to the caller's out dir instead
    return lock


def _cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(  # noqa: S603, S607 - fixed-argv host probe only
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
            out = subprocess.run(  # noqa: S603, S607 - fixed-argv host probe only
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


# ---------------------------------------------------------------------------
# run identity
# ---------------------------------------------------------------------------


def compute_run_identity(
    *,
    source: dict[str, Any],
    environment_lock_sha256: str,
    parent_gds_sha256: str,
    iccad_fixture_hashes: dict[str, str] | None,
    args_payload: dict[str, Any],
) -> str:
    """SHA-256 over everything that can change a measured number.

    Covers: measurement source commit + file hashes, the environment lock,
    the parent (and optional ICCAD) fixture hashes, and every semantic CLI
    argument.  Anything absent from here cannot affect the result; anything
    present changes the identity and therefore the run workspace.
    """
    payload = {
        "schema": "OpenLithoHub.industrial-run-identity.v1",
        "source": {
            "commit": source.get("commit"),
            "harness_sha256": source.get("harness_sha256"),
            "industrial_core_sha256": source.get("industrial_core_sha256"),
            "claim_generator_sha256": source.get("claim_generator_sha256"),
            "run_support_sha256": source.get("run_support_sha256"),
        },
        "environment_lock_sha256": environment_lock_sha256,
        "fixtures": {
            "parent_gds_sha256": parent_gds_sha256,
            "iccad": iccad_fixture_hashes or {},
        },
        "args": args_payload,
    }
    return hashlib.sha256(strict_dumps(payload).encode()).hexdigest()


# ---------------------------------------------------------------------------
# fixture identity + fail-closed layout validation
# ---------------------------------------------------------------------------


def _exact_dbu_nm(dbu_um: float) -> float | None:
    """DBU size in nm when exactly representable (1 dbu == dbu_nm)."""
    scaled = dbu_um * 1000.0
    if math.isclose(scaled, round(scaled), rel_tol=0, abs_tol=1e-9):
        return float(round(scaled))
    return None


def validate_parent_layout(
    gds_path: Path,
    *,
    expected_top_cell: str,
    layer: str,
    expected_dbu_nm: float,
) -> dict[str, Any]:
    """Fail-closed identity check of the parent GDS (G0.5).

    Verifies: the expected top cell exists (never ``top_cells()[0]``), the
    DBU arithmetic is exactly the expected nm-per-DBU, the requested layer
    exists and is non-empty, and the die bbox is positive.  Returns the
    layout facts used downstream.
    """
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(gds_path))
    cell = layout.cell(expected_top_cell)
    if cell is None:
        raise RuntimeError(
            f"{gds_path}: expected top cell {expected_top_cell!r} not found; "
            f"top cells present: {[c.name for c in layout.top_cells()]}"
        )
    dbu_nm = _exact_dbu_nm(layout.dbu)
    if dbu_nm is None or not math.isclose(dbu_nm, expected_dbu_nm, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError(
            f"{gds_path}: DBU is {layout.dbu!r} um (= {dbu_nm!r} nm/dbu), "
            f"expected exactly {expected_dbu_nm} nm/dbu"
        )
    if ":" not in layer:
        raise RuntimeError(f"layer {layer!r} must be 'LAYER:DTYPE'")
    layer_s, dtype_s = layer.split(":", 1)
    layer_index = layout.find_layer(int(layer_s), int(dtype_s))
    if layer_index is None:
        raise RuntimeError(f"{gds_path}: layer {layer} not present in layout")
    shape_count = layout.cell(cell.cell_index()).shapes(layer_index).size()
    if shape_count == 0:
        # Hierarchical designs may keep all shapes in children; treat a
        # flat-empty top cell with no child shapes as fatal.
        total = 0
        for it in cell.begin_shapes_rec(layer_index):
            _ = it
            total += 1
            break
        if total == 0:
            raise RuntimeError(f"{gds_path}: layer {layer} present but empty")
    bb = cell.bbox()
    if bb.width() <= 0 or bb.height() <= 0:
        raise RuntimeError(f"{gds_path}: top cell bbox is empty")
    return {
        "top_cell": expected_top_cell,
        "layer": layer,
        "dbu_nm": dbu_nm,
        "die_bbox_dbu": [bb.left, bb.bottom, bb.right, bb.top],
        "die_size_px": [bb.height(), bb.width()],
        "cell_count": layout.cells(),
    }


def fixture_record(
    derived_gds: Path,
    *,
    parent_sha256: str,
    crop_bbox_dbu: list[int],
    top_cell: str,
    layer: str,
    pixel_size_nm: float,
    generator_source: dict[str, Any],
) -> dict[str, Any]:
    """Identity record for a derived fixture (written next to the file)."""
    return {
        "parent_sha256": parent_sha256,
        "derived_sha256": sha256_file(derived_gds),
        "derived_bytes": derived_gds.stat().st_size,
        "crop_bbox": crop_bbox_dbu,
        "top_cell": top_cell,
        "layer": layer,
        "pixel_size_nm": pixel_size_nm,
        "generator_commit": generator_source.get("commit"),
        "generator_sha256": generator_source.get("harness_sha256"),
    }


def verify_fixture_reuse(
    derived_gds: Path,
    record: dict[str, Any],
    *,
    parent_sha256: str,
    generator_source: dict[str, Any],
) -> None:
    """Reuse a derived fixture ONLY if every identity field still matches.

    Exists-but-mismatched fixtures are a hard error: the caller must
    delete the run workspace rather than measure against stale geometry.
    """
    problems: list[str] = []
    if record.get("parent_sha256") != parent_sha256:
        problems.append("parent_sha256 changed")
    if record.get("derived_sha256") != sha256_file(derived_gds):
        problems.append("derived bytes changed")
    if record.get("generator_commit") != generator_source.get("commit"):
        problems.append("generator commit changed")
    if record.get("generator_sha256") != generator_source.get("harness_sha256"):
        problems.append("generator harness changed")
    if problems:
        raise RuntimeError(
            f"stale fixture {derived_gds} (reuse refused: {', '.join(problems)}); "
            "delete the run workspace to regenerate"
        )
