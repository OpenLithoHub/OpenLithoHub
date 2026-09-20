"""P-054 workflow governance — offline path-symmetry + preflight checks.

Re-audit post-fix (PR-3E2, H-gate "producer-only PR bypass"): the release
producer script must be covered by BOTH workflow triggers, and the
activation job must carry the preflight that spares ordinary PRs from a
guaranteed external-fetch failure while the artifact is unpublished.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import yaml

WORKFLOW = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "p054-proof-replay.yml"
)
PROTECTED = (
    "src/openlithohub/verify/**",
    "proof_artifacts/**",
    "scripts/check_verification_status.py",
    "scripts/fetch_proof_artifacts.py",
    "scripts/replay_p054_artifact.py",
    ".github/workflows/p054-proof-replay.yml",
    "tests/test_verify/test_p054_*.py",
)


def _doc() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _jobs() -> dict:
    return _doc()["jobs"]


def test_producer_script_covered_by_both_triggers():
    doc = _doc()
    push_paths = set(doc[True]["push"]["paths"])  # `on:` parses as True
    pr_paths = set(doc[True]["pull_request"]["paths"])
    for path in PROTECTED:
        assert path in push_paths, f"{path} missing from push.paths"
        assert path in pr_paths, f"{path} missing from pull_request.paths"


def test_path_filters_are_symmetric():
    doc = _doc()
    push_paths = set(doc[True]["push"]["paths"])
    pr_paths = set(doc[True]["pull_request"]["paths"])
    assert push_paths == pr_paths, "push/pull_request path filters diverged"


def test_activation_job_has_preflight_before_fetch():
    steps = _jobs()["p054-release-activation"]["steps"]
    names = [step.get("name", "") for step in steps]
    preflight = next((i for i, n in enumerate(names) if "Activation preflight" in n), None)
    fetch = next((i for i, n in enumerate(names) if "Fetch frozen artifact" in n), None)
    assert preflight is not None, "activation preflight step missing"
    assert fetch is not None
    assert preflight < fetch, "preflight must run before the external fetch"
    preflight_script = steps[preflight].get("run", "")
    assert "sha256" in preflight_script and "download_url" in preflight_script
    # PR-3E3: the neutral conclusion is recorded in the report + output
    assert "needs_replay" in preflight_script
    assert "activation-preflight.json" in preflight_script


# ---------------------------------------------------------------------------
# PR-3E3 — the preflight must actually GATE the external steps (G1–G7)
# ---------------------------------------------------------------------------

NEEDS_REPLAY_IF = "steps.activation_preflight.outputs.needs_replay == 'true'"


def _activation_steps():
    return _jobs()["p054-release-activation"]["steps"]


def _step_by_name_fragment(fragment: str) -> dict | None:
    for step in _activation_steps():
        if fragment in step.get("name", ""):
            return step
    return None


def test_g1_preflight_step_has_id():
    preflight = _step_by_name_fragment("Activation preflight")
    assert preflight is not None
    assert preflight.get("id") == "activation_preflight"
    assert "Activation preflight" in preflight["name"]


def test_g2_preflight_publishes_needs_replay():
    preflight = _step_by_name_fragment("Activation preflight")
    script = preflight.get("run", "")
    assert "GITHUB_OUTPUT" in script
    assert "needs_replay=" in script


def test_g3_fetch_step_gated_by_needs_replay():
    fetch = _step_by_name_fragment("Fetch frozen artifact")
    assert fetch is not None
    assert fetch.get("if") == NEEDS_REPLAY_IF


def test_g4_replay_step_gated_by_needs_replay():
    replay = _step_by_name_fragment("FRESH receipt")
    assert replay is not None
    assert replay.get("if") == NEEDS_REPLAY_IF


def test_g5_compare_step_gated_by_needs_replay():
    compare = _step_by_name_fragment("exact equality")
    assert compare is not None
    assert compare.get("if") == NEEDS_REPLAY_IF


def test_g6_neutral_evidence_upload_is_safe():
    upload = _step_by_name_fragment("Preserve replay evidence")
    assert upload is not None
    assert upload.get("if") == "always()"
    paths = upload["with"]["path"]
    # the deterministic preflight report is ALWAYS uploaded, so a neutral
    # run never trips if-no-files-found: error
    assert "activation-preflight.json" in paths
    assert upload["with"]["if-no-files-found"] == "error"


def test_g7_ordinary_unfetched_path_leaves_job_successful():
    # the neutral path never calls sys.exit(1): it writes needs_replay=false,
    # the gated steps skip, and the job stays green as a stable check
    preflight = _step_by_name_fragment("Activation preflight")
    run = preflight.get("run", "")
    assert "sys.exit(1)" not in run
    assert "needs_replay=" in run
    names = [s.get("name", "") for s in _activation_steps()]
    assert any("Offline status manifest" in n for n in names)


def test_pr3e4_no_activation_condition_outside_activation_job():
    # PR-3E4: steps.activation_preflight.* may only appear inside the
    # p054-release-activation job — a cross-job leak silently skips
    # steps in jobs that have no preflight.
    jobs = _jobs()
    for job_name, job in jobs.items():
        raw = json.dumps(job)
        if job_name == "p054-release-activation":
            continue
        assert "steps.activation_preflight" not in raw, (
            f"{job_name} references the activation preflight but has no preflight step"
        )


# ---------------------------------------------------------------------------
# PR-5E6 — CLI contract + fetch report preservation (audit items 1–4)
# ---------------------------------------------------------------------------


def _activation_fetch_step() -> dict:
    for step in _activation_steps():
        if "Fetch frozen artifact" in step.get("name", ""):
            return step
    raise AssertionError("activation fetch step missing")


def _full_replay_fetch_cmd() -> str:
    for step in _jobs()["p054-full-replay"]["steps"]:
        if "Fetch frozen artifact" in step.get("name", ""):
            return step.get("run", "")
    raise AssertionError("full-replay fetch step missing")


def test_cli_accepts_exact_workflow_arguments():
    # the workflow invokes: --profile p054-arf37 --fetch-report-out <path>
    # (plus --verify-only); the real parser must accept all of them.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(WORKFLOW.parents[2] / "scripts"))
    from fetch_proof_artifacts import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--profile",
            "p054-arf37",
            "--fetch-report-out",
            "/tmp/report.json",
            "--verify-only",
        ]
    )
    assert args.profile == "p054-arf37"
    assert args.fetch_report_out == Path("/tmp/report.json")
    assert args.verify_only is True


def test_full_replay_fetch_uses_fetch_report_out():
    cmd = _full_replay_fetch_cmd()
    assert "--fetch-report-out" in cmd


def test_activation_fetch_uses_fetch_report_out():
    step = _activation_fetch_step()
    assert "--fetch-report-out" in step.get("run", "")


def test_both_replay_jobs_pass_fetch_report():
    for step in _activation_steps():
        if "FRESH receipt" in step.get("name", ""):
            assert "fetch-report" in step.get("run", "")
            break
    else:
        raise AssertionError("activation replay step missing")
    full_replay = _jobs()["p054-full-replay"]
    for job_step in full_replay["steps"]:
        if "Produce the canonical replay receipt" in job_step.get("name", ""):
            assert "--fetch-report" in job_step.get("run", "")
            break
    else:
        raise AssertionError("canonical replay receipt step missing")


def test_both_evidence_uploads_include_fetch_report():
    """R129/P2-B: BOTH evidence uploads — the scheduled `p054-replay-logs`
    and the PR `p054-activation-evidence` — must carry the canonical fetch
    report, so a regression in either upload is caught."""
    doc = _doc()
    uploads: dict[str, str] = {}
    for job in doc["jobs"].values():
        for step in job.get("steps", []):
            upload = step.get("with", {})
            name = upload.get("name", "")
            if "path" in upload and name in ("p054-replay-logs", "p054-activation-evidence"):
                uploads[name] = upload["path"]
    assert set(uploads) == {"p054-replay-logs", "p054-activation-evidence"}, (
        f"expected exactly one scheduled and one activation evidence upload, got {sorted(uploads)}"
    )
    for name, path in uploads.items():
        assert "fetch-report" in path, f"{name} upload lost the canonical fetch report"


# ---------------------------------------------------------------------------
# PR-5F4 — external artifact hygiene (audit items 1–4)
# ---------------------------------------------------------------------------


def test_artifact_not_tracked():
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(  # noqa: S603 — fixed argv
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "proof_artifacts/p054/external/p054-arf37-frozen-artifact.zip",
        ],
        capture_output=True,
        text=True,
        cwd=repo,
        check=False,
    )
    assert proc.returncode != 0, "external artifact must not be tracked in git"


def test_gitignore_covers_external_dir():
    repo = Path(__file__).resolve().parents[2]
    gitignore = (repo / ".gitignore").read_text()
    assert "proof_artifacts/p054/external/" in gitignore


def test_fetch_report_request_forces_transport(tmp_path, monkeypatch):
    # PR-5F4 C: a pre-existing SHA/byte-correct destination must NOT
    # short-circuit when --fetch-report-out is supplied — the transport
    # must run and produce the report.
    import scripts.fetch_proof_artifacts as fetcher

    registry = {
        "artifacts": [
            {
                "name": "p054-arf37-frozen-artifact.zip",
                "profiles": ["p054-arf37"],
                "sha256": hashlib.sha256(b"BYTES!").hexdigest(),
                "bytes": 6,
                "download_url": "https://zenodo.org/records/1/files/x.zip",
                "destination": "proof_artifacts/p054/external",
                "source": "zenodo",
                "required_for": ["p054-full-replay"],
            }
        ]
    }
    real_registry = fetcher.REGISTRY
    real_root = fetcher.ROOT
    try:
        fetcher.REGISTRY = tmp_path / "registry.json"
        fetcher.REGISTRY.write_text(json.dumps(registry))
        fetcher.ROOT = tmp_path
        monkeypatch.setattr(fetcher, "ROOT", tmp_path)

        transport_calls = []

        def fake_curl(cmd, **kwargs):
            transport_calls.append(cmd)
            out = cmd[cmd.index("-o") + 1]
            Path(out).write_bytes(b"BYTES!")
            # curl -w emits the final effective URL on stdout
            return subprocess.CompletedProcess(
                cmd, 0, stdout="https://zenodo.org/records/1/files/x.zip\n", stderr=""
            )

        monkeypatch.setattr(fetcher.subprocess, "run", fake_curl)
        rc = fetcher.main(
            [
                "--profile",
                "p054-arf37",
                "--fetch-report-out",
                str(tmp_path / "report.json"),
            ]
        )
        assert rc == 0
        assert transport_calls, "explicit fetch report request must transport"
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["effective_url"] == "https://zenodo.org/records/1/files/x.zip"
        assert (
            report["artifact_sha256"]
            == "ca275fd3a431af5615d3119b95585d6f030a8678894bac9c06faeb162570914f"
        )
    finally:
        fetcher.REGISTRY = real_registry
        fetcher.ROOT = real_root


def test_existing_verified_destination_short_circuits_without_fetch(tmp_path, monkeypatch):
    # without an explicit fetch report request, a verified destination is
    # reused (no transport); the report is never synthesized.
    import scripts.fetch_proof_artifacts as fetcher

    registry = {
        "artifacts": [
            {
                "name": "p054-arf37-frozen-artifact.zip",
                "profiles": ["p054-arf37"],
                "sha256": hashlib.sha256(b"BYTES!").hexdigest(),
                "bytes": 6,
                "download_url": "https://zenodo.org/records/1/files/x.zip",
                "destination": "proof_artifacts/p054/external",
                "source": "zenodo",
                "required_for": ["p054-full-replay"],
            }
        ]
    }
    real_registry = fetcher.REGISTRY
    real_root = fetcher.ROOT
    try:
        fetcher.REGISTRY = tmp_path / "registry.json"
        fetcher.REGISTRY.write_text(json.dumps(registry))
        fetcher.ROOT = tmp_path
        ext = tmp_path / "proof_artifacts" / "p054" / "external"
        ext.mkdir(parents=True)
        (ext / "p054-arf37-frozen-artifact.zip").write_bytes(b"BYTES!")

        transport_calls = []

        def fake_curl(cmd, **kwargs):
            transport_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(fetcher.subprocess, "run", fake_curl)
        rc = fetcher.main(["--profile", "p054-arf37", "--verify-only"])
        assert rc == 0
        assert transport_calls == []
    finally:
        fetcher.REGISTRY = real_registry
        fetcher.ROOT = real_root


def test_fetch_curl_command_includes_download_url():
    # R127/R128 regression: a curl command without the download URL fails
    # with "no URL specified" — the URL must be part of every fetch.
    import scripts.fetch_proof_artifacts as fetcher

    entry = {
        "name": "x.zip",
        "profiles": ["p054-arf37"],
        "sha256": "a" * 64,
        "bytes": 6,
        "download_url": "https://zenodo.org/records/1/files/x.zip?download=1",
        "source": "zenodo",
        "required_for": ["p054-full-replay"],
    }
    destination = Path(tempfile.mkdtemp()) / "x.zip"
    captured = []

    def fake_curl(cmd, **kwargs):
        captured.append(list(cmd))
        out = cmd[cmd.index("-o") + 1]
        Path(out).write_bytes(b"BYTES!")
        return subprocess.CompletedProcess(
            cmd, 0, stdout="https://zenodo.org/records/1/files/x.zip\n", stderr=""
        )

    orig_run = fetcher.subprocess.run
    fetcher.subprocess.run = fake_curl
    try:
        fetcher.fetch(entry, destination, profile="p054-arf37")
    finally:
        fetcher.subprocess.run = orig_run
    assert captured, "transport must have run"
    cmd = captured[0]
    # the load-bearing regression: the download URL must be in the command
    assert entry["download_url"] in cmd, "download URL missing from curl command"
    # curl writes to destination + ".part" (atomic download, PR-5E2)
    assert str(destination) + ".part" in cmd


# ---------------------------------------------------------------------------
# PR-5F5 — action pin map + scheduled receipt equality (audit items 1–4)
# ---------------------------------------------------------------------------

EXPECTED_ACTION_PINS = {
    "actions/checkout": "d23441a48e516b6c34aea4fa41551a30e30af803",
    "actions/setup-python": "ece7cb06caefa5fff74198d8649806c4678c61a1",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
}
RETIRED_PINS = {
    "08c6903cd8c0fde910a37f88322ed31bde414eec1",
    "0a5c61591373683505ea898d09fdccdc57c2ab49",
    "6f51ac03b9356f520e9adb0b8836a71a4b6cec80",
}


def _uses_entries() -> list[str]:
    doc = _doc()
    uses = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "uses" and isinstance(v, str):
                    uses.append(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(doc)
    return uses


def test_all_uses_entries_use_expected_pins():
    for uses in _uses_entries():
        action = uses.split("@")[0]
        pin = uses.split("@")[1] if "@" in uses else None
        assert action in EXPECTED_ACTION_PINS, f"unexpected action {uses!r}"
        assert pin == EXPECTED_ACTION_PINS[action], f"{uses!r} does not match the expected pin"


def test_retired_pins_absent():
    raw = WORKFLOW.read_text()
    for retired in RETIRED_PINS:
        assert retired not in raw, f"retired pin {retired} still present"


def test_scheduled_replay_compares_receipt_equality():
    job = _jobs()["p054-full-replay"]
    names = [s.get("name", "") for s in job.get("steps", [])]
    assert any("Compare scheduled fresh receipt with committed receipt" in n for n in names), (
        "receipt equality step missing from the scheduled replay"
    )


def test_scheduled_equality_script_uses_exact_whole_document_comparison():
    job = _jobs()["p054-full-replay"]
    for step in job.get("steps", []):
        if "Compare scheduled fresh receipt" in step.get("name", ""):
            run = step.get("run", "")
            assert "if fresh != committed:" in run
            assert "raise SystemExit" in run
            return
    raise AssertionError("scheduled equality step missing")
