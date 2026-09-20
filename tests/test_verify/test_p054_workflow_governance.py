"""P-054 workflow governance — offline path-symmetry + preflight checks.

Re-audit post-fix (PR-3E2, H-gate "producer-only PR bypass"): the release
producer script must be covered by BOTH workflow triggers, and the
activation job must carry the preflight that spares ordinary PRs from a
guaranteed external-fetch failure while the artifact is unpublished.
"""

import json

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
