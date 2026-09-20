"""P-054 workflow governance — offline path-symmetry + preflight checks.

Re-audit post-fix (PR-3E2, H-gate "producer-only PR bypass"): the release
producer script must be covered by BOTH workflow triggers, and the
activation job must carry the preflight that spares ordinary PRs from a
guaranteed external-fetch failure while the artifact is unpublished.
"""

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
    preflight = next((i for i, n in enumerate(names) if "Preflight" in n), None)
    fetch = next((i for i, n in enumerate(names) if "Fetch frozen artifact" in n), None)
    assert preflight is not None, "activation preflight step missing"
    assert fetch is not None
    assert preflight < fetch, "preflight must run before the external fetch"
    preflight_script = steps[preflight].get("run", "")
    assert "sha256" in preflight_script and "download_url" in preflight_script
    assert "NEUTRAL" in preflight_script
