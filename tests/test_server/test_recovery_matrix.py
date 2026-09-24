"""PR-E hostile crash/restart matrix (roadmap §19).

Every test exercises the SQLite durable store + runtime recovery paths at
the storage/runtime layer, with the public /v1 semantics assumed frozen
(API parity is proven separately against both backends).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from openlithohub.server.config import ServerConfig
from openlithohub.server.job_store import (
    PARAMS_SCHEMA,
    JobRecord,
    SQLiteJobStore,
    serialize_params,
)
from openlithohub.server.runtime import ServerRuntime
from openlithohub.server.schemas import JobStatus


def _sqlite_config(state_dir: Path, **overrides: object) -> ServerConfig:
    return ServerConfig(job_backend="sqlite", state_dir=str(state_dir), **overrides)  # type: ignore[arg-type]


def _runtime(state_dir: Path, runner, **overrides: object) -> ServerRuntime:
    config = _sqlite_config(state_dir, **overrides)
    runtime = ServerRuntime(config, optimize_runner=runner)
    return runtime


def _make_input(state_dir: Path, job_id: str, payload: bytes = b"layout-bytes") -> str:
    job_dir = state_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "input.npy"
    path.write_bytes(payload)
    return str(path)


def _record(job_id: str, input_path: str, output_path: str) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_utc="now",
        input_path=input_path,
        input_format="npy",
        params_json=serialize_params({"input_path": input_path, "output_path": output_path}),
    )


def _trivial_runner(**params: object) -> dict[str, object]:
    output = Path(str(params.get("output_path")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"artifact-bytes")
    return {"output_path": str(output), "tiles": 1}


# ---- store unit parity ------------------------------------------------------


def test_params_envelope_round_trip_and_fail_closed() -> None:
    raw = serialize_params({"model_name": "m", "tile_size": 64})
    envelope = json.loads(raw)
    assert envelope["schema"] == PARAMS_SCHEMA
    parsed = json.loads(raw)["params"]
    assert parsed["model_name"] == "m"
    from openlithohub.server.job_store import parse_params

    assert parse_params(raw) == {"model_name": "m", "tile_size": 64}
    for bad in (None, "not json", '{"schema": "other.v9", "params": {}}', '{"schema": null}'):
        with pytest.raises(ValueError):
            parse_params(bad)  # type: ignore[arg-type]


def test_sqlite_unknown_newer_schema_fails_startup(tmp_path: Path) -> None:
    import sqlite3

    state = tmp_path / "state"
    state.mkdir()
    conn = sqlite3.connect(state / "jobs.sqlite3")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '999')")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="NEWER schema"):
        SQLiteJobStore(state)


def test_second_live_process_cannot_own_the_store(tmp_path: Path) -> None:
    state = tmp_path / "state"
    first = SQLiteJobStore(state)
    with pytest.raises(RuntimeError, match="owned by another live"):
        SQLiteJobStore(state)
    first.close()
    # After the owner releases (process death / close), the store is usable.
    second = SQLiteJobStore(state)
    second.close()


# ---- E1: queued survives restart --------------------------------------------


def test_e1_queued_survives_restart(tmp_path: Path) -> None:
    state = tmp_path / "state"
    job_id = "job-e1"
    input_path = _make_input(state, job_id)
    output_path = str(state / "jobs" / job_id / "artifact.oas")

    # Crash window: the QUEUED row commits but the (dead) worker never
    # claims it. The first runtime is never started, so nothing races the
    # simulated process death.
    first = _runtime(state, _trivial_runner)
    first._store.create_queued(_record(job_id, input_path, output_path))
    assert first._store.count_queued() == 1
    first._store.close()  # process death

    second = _runtime(state, _trivial_runner)
    assert second._store.get(job_id).status is JobStatus.QUEUED, (
        "recovery must keep a queued job with intact durable input queued"
    )
    second.start()  # the worker then executes it
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = second._store.get(job_id)
        if record.status is JobStatus.SUCCEEDED:
            break
        time.monotonic()
        time.sleep(0.05)
    assert record.status is JobStatus.SUCCEEDED, record
    second.stop()


# ---- E2: running fails honestly after crash ----------------------------------


def test_e2_stale_running_recovers_failed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    job_id = "job-e2"
    input_path = _make_input(state, job_id)
    output_path = str(state / "jobs" / job_id / "artifact.oas")

    first = _runtime(state, _trivial_runner)
    first._store.create_queued(_record(job_id, input_path, output_path))
    claimed = first._store.claim_next("owner-1")
    assert claimed is not None and claimed.status is JobStatus.RUNNING
    # Process death before the terminal commit: abandon without stop().
    first._store.close()

    second = _runtime(state, _trivial_runner)
    second.start()
    record = second._store.get(job_id)
    assert record.status is JobStatus.FAILED
    assert record.completed_utc is not None
    assert "restarted while job execution was in progress" in record.error
    second.stop()


def test_e2b_no_automatic_retry_after_restart(tmp_path: Path) -> None:
    state = tmp_path / "state"
    job_id = "job-e2b"
    input_path = _make_input(state, job_id)
    output_path = str(state / "jobs" / job_id / "artifact.oas")

    first = _runtime(state, _trivial_runner)
    first._store.create_queued(_record(job_id, input_path, output_path))
    first._store.claim_next("owner-1")
    first._store.close()  # crash while RUNNING

    calls: list[int] = []

    def counting_runner(**params: object) -> dict[str, object]:
        calls.append(1)
        return _trivial_runner(**params)

    second = _runtime(state, counting_runner)
    second.start()
    time.sleep(0.6)  # several worker poll intervals
    assert calls == [], "stale RUNNING must not be retried"
    assert second._store.get(job_id).status is JobStatus.FAILED
    second.stop()


# ---- E3: succeeded artifact survives restart ----------------------------------


def test_e3_succeeded_artifact_survives_restart(tmp_path: Path) -> None:
    state = tmp_path / "state"
    job_id = "job-e3"
    input_path = _make_input(state, job_id)
    output_path = str(state / "jobs" / job_id / "artifact.oas")

    first = _runtime(state, _trivial_runner)
    first.start()
    first._store.create_queued(_record(job_id, input_path, output_path))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if first._store.get(job_id).status is JobStatus.SUCCEEDED:
            break
        time.sleep(0.05)
    assert first._store.get(job_id).status is JobStatus.SUCCEEDED
    first.stop()

    second = _runtime(state, _trivial_runner)
    second.start()
    record = second._store.get(job_id)
    assert record.status is JobStatus.SUCCEEDED
    served = second.get_job_artifact_path(job_id)
    assert Path(served).read_bytes() == b"artifact-bytes"
    second.release_artifact_lease(job_id)
    second.stop()


# ---- E4: persisted queue counts toward capacity --------------------------------


def test_e4_persisted_queue_enforces_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openlithohub.server.runtime import JobQueueFullError

    state = tmp_path / "state"
    job_id = "job-e4"
    input_path = _make_input(state, job_id)
    output_path = str(state / "jobs" / job_id / "artifact.oas")

    first = _runtime(state, _trivial_runner, job_queue_depth=1)
    first._store.create_queued(_record(job_id, input_path, output_path))
    first._store.close()  # crash with the job queued and unclaimed

    second = _runtime(state, _trivial_runner, job_queue_depth=1)
    # Freeze the worker claim BEFORE start: the durable row must stay
    # QUEUED so the capacity gate sees it (otherwise the worker may claim
    # it between start() and the reserve probe — a race, not a gate).
    monkeypatch.setattr(second._store, "peek_next_queued", lambda: None)
    second.start()
    with pytest.raises(JobQueueFullError):
        second.try_reserve_job_slot()
    assert second._store.count_queued() == 1
    second.stop()


# ---- E5: upload crash creates no ghost job --------------------------------------


def test_e5_upload_crash_leaves_no_ghost_job(tmp_path: Path) -> None:
    state = tmp_path / "state"
    job_id = "job-e5"

    first = _runtime(state, _trivial_runner)
    # Crash window: staged upload exists but the durable job never commits.
    staging = first.staging_path(job_id, ".npy")
    staging.write_bytes(b"partial")
    orphan_input = state / "jobs" / job_id / "input.npy"
    orphan_input.parent.mkdir(parents=True, exist_ok=True)
    orphan_input.write_bytes(b"partial")
    first._store.close()  # die before create_queued

    second = _runtime(state, _trivial_runner)
    second.start()  # recovery garbage collection
    assert second._store.get(job_id) is None, "no ghost job may exist"
    assert second._store.count_queued() == 0, "no capacity may be consumed"
    assert not staging.exists(), "staging garbage must be collected"
    assert not orphan_input.parent.exists() or not any(orphan_input.parent.iterdir()), (
        "orphan job dir must be collected"
    )
    second.stop()


# ---- E6: lost wake signal cannot lose work ----------------------------------------


def test_e6_lost_wake_signal_still_executes(tmp_path: Path) -> None:
    state = tmp_path / "state"
    job_id = "job-e6"
    input_path = _make_input(state, job_id)
    output_path = str(state / "jobs" / job_id / "artifact.oas")

    runtime = _runtime(state, _trivial_runner)
    runtime.start()
    # Commit directly through the store WITHOUT the runtime wake signal —
    # the worker's poll must discover the committed row on its own.
    runtime._store.create_queued(_record(job_id, input_path, output_path))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if runtime._store.get(job_id).status is JobStatus.SUCCEEDED:
            break
        time.sleep(0.05)
    assert runtime._store.get(job_id).status is JobStatus.SUCCEEDED
    runtime.stop()


# ---- E7: output crash window ---------------------------------------------------------


def test_e7_crash_before_success_commit_never_serves_artifact(tmp_path: Path) -> None:
    from openlithohub.server.runtime import JobArtifactUnavailableError

    state = tmp_path / "state"
    job_id = "job-e7"
    input_path = _make_input(state, job_id)
    output_path = str(state / "jobs" / job_id / "artifact.oas")

    first = _runtime(state, _trivial_runner)
    first._store.create_queued(_record(job_id, input_path, output_path))
    first._store.claim_next("owner-1")
    # The artifact bytes land, but the process dies before SUCCEEDED.
    Path(output_path + ".tmp").write_bytes(b"partial-artifact")
    Path(output_path).write_bytes(b"partial-artifact")  # hostile pre-publication
    first._store.close()

    second = _runtime(state, _trivial_runner)
    second.start()
    record = second._store.get(job_id)
    assert record.status is JobStatus.FAILED, "crashed RUNNING must fail closed"
    with pytest.raises(JobArtifactUnavailableError):
        second.acquire_artifact_lease(job_id)
    # GC sweeps the uncommitted partial artifact bytes.
    assert not Path(output_path + ".tmp").exists()
    second.stop()


# ---- E8: success atomicity ---------------------------------------------------------------


def test_e8_succeeded_implies_verifiable_artifact(tmp_path: Path) -> None:
    state = tmp_path / "state"
    job_id = "job-e8"
    input_path = _make_input(state, job_id)
    output_path = str(state / "jobs" / job_id / "artifact.oas")

    runtime = _runtime(state, _trivial_runner)
    runtime.start()
    runtime._store.create_queued(_record(job_id, input_path, output_path))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = runtime._store.get(job_id)
        if record.status is JobStatus.SUCCEEDED:
            break
        time.sleep(0.05)
    assert record.status is JobStatus.SUCCEEDED
    artifact = Path(record.output_path)
    assert artifact.is_file()
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert record.artifact_sha256 == digest
    assert record.artifact_bytes == artifact.stat().st_size == len(b"artifact-bytes")
    runtime.stop()


# ---- E9: artifact lease protects download -------------------------------------------------


def test_e9_lease_protects_artifact_from_eviction(tmp_path: Path) -> None:
    from openlithohub.server.runtime import UnknownJobError

    state = tmp_path / "state"
    job_id = "job-e9"
    input_path = _make_input(state, job_id)
    output_path = str(state / "jobs" / job_id / "artifact.oas")

    runtime = _runtime(state, _trivial_runner, job_ttl_seconds=0.0)
    runtime.start()
    runtime._store.create_queued(_record(job_id, input_path, output_path))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if runtime._store.get(job_id).status is JobStatus.SUCCEEDED:
            break
        time.sleep(0.05)

    served = runtime.acquire_artifact_lease(job_id)
    try:
        # TTL=0 makes everything evictable except the leased download.
        doomed = runtime._evict_terminal()
        assert Path(served).is_file(), "leased artifact must survive eviction"
        assert all(Path(d).resolve() != Path(served).resolve() for d in doomed)
    finally:
        runtime.release_artifact_lease(job_id)
    # After release the artifact is evictable like any terminal job; the
    # public contract for an evicted job is UNKNOWN_JOB (404).
    runtime._evict_terminal()
    assert runtime._store.get(job_id) is None
    with pytest.raises(UnknownJobError):
        runtime.acquire_artifact_lease(job_id)
    runtime.stop()


# ---- E10: deletion survives restart ----------------------------------------------------------


def test_e10_deletion_survives_restart(tmp_path: Path) -> None:
    state = tmp_path / "state"
    job_id = "job-e10"
    input_path = _make_input(state, job_id)
    output_path = str(state / "jobs" / job_id / "artifact.oas")

    first = _runtime(state, _trivial_runner)
    first._store.create_queued(_record(job_id, input_path, output_path))
    cleanup = first.delete_job(job_id)
    assert first._store.get(job_id) is None
    first._store.close()
    assert cleanup, "durable footprint must be cleaned"

    second = _runtime(state, _trivial_runner)
    second.start()
    assert second._store.get(job_id) is None, "deletion must survive restart"
    second.stop()


# ---- queued with missing input cancels explicitly (§12) --------------------------------------


def test_queued_missing_input_cancels_with_explanation(tmp_path: Path) -> None:
    state = tmp_path / "state"
    job_id = "job-e12b"
    output_path = str(state / "jobs" / job_id / "artifact.oas")
    input_path = str(state / "jobs" / job_id / "input.npy")  # never written

    first = _runtime(state, _trivial_runner)
    first._store.create_queued(_record(job_id, input_path, output_path))
    first._store.close()

    second = _runtime(state, _trivial_runner)
    second.start()
    record = second._store.get(job_id)
    assert record.status is JobStatus.CANCELLED
    assert "missing" in record.error
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if second._store.count_queued() == 0:
            break
        time.sleep(0.05)
    assert second._store.count_queued() == 0
    second.stop()
