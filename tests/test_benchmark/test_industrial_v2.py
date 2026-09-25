"""Industrial Benchmark v2 protocol hostile tests (PR-G Phase 2A).

Roadmap §30/§31/§32/§35: identity sensitivity, manifest rejection,
timing/correctness admission, and the CPU-only publication block.
No expensive measurements run here — protocol only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from openlithohub.benchmark.industrial_v2 import (
    CANONICAL_FAMILY,
    FLOAT_TOLERANCE_BY_DTYPE,
    RESULTS_ROOT,
    RunConfigV2,
    StatusV2,
    admit_headline,
    compute_run_identity_v2,
    formal_publication_blockers,
    gpu_environment_lock,
    promote_canonical_family,
    write_strict_json,
)


def _identity(cfg: RunConfigV2, **overrides: str) -> str:
    kwargs = dict(
        measurement_commit="a" * 40,
        harness_sha256="h",
        core_sha256="c",
        claim_generator_sha256="g",
        verifier_sha256="v",
    )
    kwargs.update(overrides)
    return compute_run_identity_v2(cfg, **kwargs)


# ---- §31: identity sensitivity matrix ----------------------------------------


@pytest.mark.parametrize(
    "mutation",
    [
        {"batch_size": 16},
        {"tile_size": 2048},
        {"halo_px": 32},
        {"dtype": "bf16"},
        {"tf32_matmul": True},
        {"tf32_cudnn": True},
        {"deterministic_algorithms": False},
        {"compile_mode": "default"},
        {"device": "cuda:1"},
        {"pixel_nm": 0.5},
        {"forward_radius": 5},
        {"warmup_count": 3},
        {"repeat_count": 7},
        {"window_sizes": (4096,)},
        {"tiers": ("a", "b")},
    ],
)
def test_identity_sensitive_to_every_semantic_knob(mutation: dict) -> None:
    base = _identity(RunConfigV2())
    mutated = _identity(RunConfigV2().with_changes(**mutation))
    assert base != mutated, f"identity ignored semantic change {mutation}"


@pytest.mark.parametrize(
    "field,value",
    [
        ("measurement_commit", "b" * 40),
        ("harness_sha256", "h2"),
        ("core_sha256", "c2"),
        ("claim_generator_sha256", "g2"),
        ("verifier_sha256", "v2"),
    ],
)
def test_identity_sensitive_to_source_bytes(field: str, value: str) -> None:
    base = _identity(RunConfigV2())
    assert base != _identity(RunConfigV2(), **{field: value})


def test_identity_sensitive_to_fixture_hash() -> None:
    base = _identity(RunConfigV2())
    assert base != _identity(RunConfigV2().with_changes(fixture_sha256="f" * 64))


# ---- §23/§9: status vocabulary + GPU environment lock --------------------------


def test_status_vocabulary_is_frozen() -> None:
    assert {s.value for s in StatusV2} == {
        "SUCCESS",
        "FAILED",
        "INCONCLUSIVE",
        "UNSUPPORTED",
        "NOT_RUN_ENVIRONMENT",
        "NOT_RUN_MEMORY_POLICY",
        "INFEASIBLE_ON_REFERENCE_MACHINE",
    }


def test_gpu_environment_lock_completeness_on_cpu_host() -> None:
    lock = gpu_environment_lock()
    for key in (
        "available",
        "count",
        "devices",
        "torch_cuda_version",
        "torch_version",
        "tf32_matmul",
        "tf32_cudnn",
    ):
        assert key in lock, f"GPU environment lock missing {key!r} (B2-D)"
    if not lock["available"]:
        assert lock["devices"] == [] and lock["count"] == 0


# ---- CPU-only publication block (§35, red-team Q1) ------------------------------


def test_cpu_only_host_cannot_publish_canonical_v2(tmp_path: Path) -> None:
    lock = gpu_environment_lock()  # CUDA-less CI/dev host
    if lock["available"]:  # pragma: no cover
        pytest.skip("host has CUDA")
    blockers = formal_publication_blockers(
        env_lock=lock,
        tier_rows={
            "a": {"status": "SUCCESS", "correctness_witness_pass": True},
            "b": {"status": "NOT_RUN_ENVIRONMENT", "correctness_witness_pass": False},
            "c": {"status": "NOT_RUN_ENVIRONMENT", "correctness_witness_pass": False},
        },
        git_clean=True,
        provisional=False,
    )
    assert any("FORMAL_PUBLICATION_BLOCKED" in b for b in blockers)
    assert any("tier B" in b for b in blockers)
    assert any("tier C" in b for b in blockers)

    leftover = promote_canonical_family(
        workspace_dir=tmp_path,
        canonical_root=tmp_path / "canonical",
        env_lock=lock,
        tier_rows={
            "a": {"status": "SUCCESS", "correctness_witness_pass": True},
            "b": {"status": "NOT_RUN_ENVIRONMENT", "correctness_witness_pass": False},
            "c": {"status": "NOT_RUN_ENVIRONMENT", "correctness_witness_pass": False},
        },
        git_clean=True,
        provisional=False,
    )
    assert leftover == blockers
    assert not (tmp_path / "canonical").exists(), "nothing may be promoted"


def test_incomplete_tiers_or_dirty_tree_block_publication(tmp_path: Path) -> None:
    good_row = {"status": "SUCCESS", "correctness_witness_pass": True}
    env = {"available": True, "count": 1}

    assert (
        formal_publication_blockers(
            env_lock=env,
            tier_rows={"a": good_row, "b": good_row, "c": good_row},
            git_clean=True,
            provisional=False,
        )
        == []
    )

    dirty = formal_publication_blockers(
        env_lock=env,
        tier_rows={"a": good_row, "b": good_row, "c": good_row},
        git_clean=False,
        provisional=False,
    )
    assert any("dirty tracked tree" in b for b in dirty)

    failed_correctness = formal_publication_blockers(
        env_lock=env,
        tier_rows={
            "a": {"status": "SUCCESS", "correctness_witness_pass": False},
            "b": good_row,
            "c": good_row,
        },
        git_clean=True,
        provisional=False,
    )
    assert any("correctness witness" in b for b in failed_correctness)


# ---- §26: headline firewall ------------------------------------------------------


def test_headline_firewall_thresholds_and_gates() -> None:
    ok, _ = admit_headline(
        claim_id="IB2-GPU-E2E-4096",
        correctness_pass=True,
        repeat_count=5,
        runtime_speedup=1.10,
        scope="Tier B single-GPU streaming execution",
    )
    assert ok
    below_speedup, why = admit_headline(
        claim_id="IB2-GPU-E2E-4096",
        correctness_pass=True,
        repeat_count=5,
        runtime_speedup=1.05,
        scope="Tier B",
    )
    assert not below_speedup and "1.05" in why
    below_memory, why2 = admit_headline(
        claim_id="IB2-GPU-MEM-4096",
        correctness_pass=True,
        repeat_count=5,
        memory_reduction=0.15,
        scope="Tier B",
    )
    assert not below_memory and "memory" in why2
    no_correctness, why3 = admit_headline(
        claim_id="IB2-X",
        correctness_pass=False,
        repeat_count=5,
        runtime_speedup=9.9,
        scope="Tier B",
    )
    assert not no_correctness and "correctness" in why3
    few_repeats, why4 = admit_headline(
        claim_id="IB2-X",
        correctness_pass=True,
        repeat_count=3,
        runtime_speedup=9.9,
        scope="Tier B",
    )
    assert not few_repeats and "repeat" in why4
    no_scope, why5 = admit_headline(
        claim_id="IB2-X",
        correctness_pass=True,
        repeat_count=5,
        runtime_speedup=9.9,
        scope="",
    )
    assert not no_scope and "scope" in why5


def test_dtype_tolerances_are_frozen_before_measurement() -> None:
    assert FLOAT_TOLERANCE_BY_DTYPE == {"fp32": 1e-5}
    assert "bf16" not in FLOAT_TOLERANCE_BY_DTYPE, "bf16 is a separate, unmerged claim"


# ---- §32/§28: manifest + verifier hostile table ----------------------------------


class FamilyBuilder:
    """Build a synthetic canonical family in a temp dir, then corrupt it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_config = RunConfigV2(tiers=("a", "b", "c"))
        self.commit = "a" * 40
        # keys match the verifier's recomputation contract
        self.source = {
            "harness": "h",
            "core": "c",
            "claim_generator": "g",
            "verifier": "v",
        }
        self.identity = compute_run_identity_v2(
            self.run_config,
            measurement_commit=self.commit,
            harness_sha256=self.source["harness"],
            core_sha256=self.source["core"],
            claim_generator_sha256=self.source["claim_generator"],
            verifier_sha256=self.source["verifier"],
        )
        self.tier_row = {
            "status": "SUCCESS",
            "correctness_witness_pass": True,
            "timing_method": "cuda_synchronized",
            "device": "cuda:0",
            "dtype": "fp32",
            "max_memory_allocated": 1,
            "max_memory_reserved": 2,
            "device_requires_cuda": False,
            # claim-bearing repeat statistics (verifier-locked on hopkins)
            "aggregate_n": 5,
            "aggregate_median_s": 1.0,
            "aggregate_p10_s": 0.9,
            "aggregate_p90_s": 1.1,
            "timing_observations": 1,
        }

    def write(self) -> Path:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from scripts.verify_industrial_v2_artifacts import VerifyError

        self.verify_error = VerifyError
        for name in (
            "industrial-v2-index.json",
            "industrial-v2-gpu-runtime.json",
            "industrial-v2-hopkins.json",
        ):
            write_strict_json(
                self.root / name,
                {
                    "schema": "OpenLithoHub.industrial-benchmark.v2",
                    "run_identity": self.identity,
                    "correctness_witness_pass": True,
                    "claim_level": "REPRODUCED_INTERNAL",
                    "rows": [dict(self.tier_row)],
                },
            )
        write_strict_json(
            self.root / "industrial-v2-run-config.json",
            {
                "schema": "OpenLithoHub.industrial-run-config.v2",
                "measurement_commit": self.commit,
                "tracked_tree_clean": True,
                "provisional": False,
                "source_hashes": self.source,
                "run_identity": self.identity,
                "run_config": self.run_config.to_payload(),
            },
        )
        write_strict_json(
            self.root / "industrial-v2-distribution-freeze.txt",
            {
                "gpu": {
                    "available": True,
                    "count": 1,
                    "devices": [
                        {
                            "index": 0,
                            "name": "TestGPU",
                            "total_memory_bytes": 1,
                            "compute_capability": "8.0",
                        }
                    ],
                    "driver_version": "550",
                    "torch_cuda_version": "12.4",
                    "cudnn_version": 90100,
                    "tf32_matmul": False,
                    "tf32_cudnn": False,
                    "torch_version": "2.14.0",
                }
            },
        )
        # manifest FIRST (SHA256SUMS covers it); the manifest lists every
        # canonical member except itself and SHA256SUMS.txt (it cannot
        # carry its own byte count).
        members = sorted(CANONICAL_FAMILY - {"manifest.json", "SHA256SUMS.txt"})
        manifest = {
            "run_identity": self.identity,
            "members": [
                {"name": name, "bytes": (self.root / name).stat().st_size} for name in members
            ],
        }
        write_strict_json(self.root / "manifest.json", manifest)
        lines = []
        for name in members:
            digest = hashlib.sha256((self.root / name).read_bytes()).hexdigest()
            lines.append(f"{digest}  {name}")
        (self.root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
        return self.root

    def _refresh_closures(self) -> None:
        # after any payload edit, refresh manifest bytes + SHA256SUMS
        members = sorted(CANONICAL_FAMILY - {"SHA256SUMS.txt"})
        manifest = json.loads((self.root / "manifest.json").read_text())
        for entry in manifest["members"]:
            entry["bytes"] = (self.root / entry["name"]).stat().st_size
        write_strict_json(self.root / "manifest.json", manifest)
        lines = [
            f"{hashlib.sha256((self.root / name).read_bytes()).hexdigest()}  {name}"
            for name in members
        ]
        (self.root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")

    def verify(self):
        import scripts.verify_industrial_v2_artifacts as verifier

        try:
            verifier.verify(self.root)
        except verifier.VerifyError as exc:
            return False, str(exc)
        return True, "closed"

    # corruptions ----------------------------------------------------------

    def corrupt_remove_member(self, name: str = "industrial-v2-hopkins.json") -> None:
        (self.root / name).unlink()

    def corrupt_extra_member(self) -> None:
        (self.root / "rogue.json").write_text("{}")

    def corrupt_wrong_sha(self) -> None:
        """SHA256SUMS carries a well-formed but WRONG digest: the recorded
        hash no longer matches the member bytes (bytes themselves valid)."""
        sums = self.root / "SHA256SUMS.txt"
        lines = []
        for line in sums.read_text().splitlines():
            if line.endswith("  industrial-v2-index.json"):
                line = ("f" * 64) + "  industrial-v2-index.json"
            lines.append(line)
        sums.write_text("\n".join(lines) + "\n")

    def corrupt_stale_sums(self) -> None:
        target = self.root / "industrial-v2-index.json"
        payload = json.loads(target.read_text())
        payload["rows"][0]["owned_run_count"] = 999
        target.write_text(json.dumps(payload))  # SHA256SUMS NOT refreshed -> stale

    def corrupt_manifest_bytes(self) -> None:
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["members"][0]["bytes"] += 1
        (self.root / "manifest.json").write_text(json.dumps(manifest))

    def corrupt_unsynchronized_gpu_timing(self) -> None:
        target = self.root / "industrial-v2-gpu-runtime.json"
        payload = json.loads(target.read_text())
        payload["rows"][0]["timing_method"] = "host_perf_counter"
        write_strict_json(target, payload)
        self._refresh_closures()

    def corrupt_missing_gpu_memory_stat(self) -> None:
        target = self.root / "industrial-v2-gpu-runtime.json"
        payload = json.loads(target.read_text())
        del payload["rows"][0]["max_memory_reserved"]
        write_strict_json(target, payload)
        self._refresh_closures()

    def corrupt_not_run_environment_row(self) -> None:
        target = self.root / "industrial-v2-gpu-runtime.json"
        payload = json.loads(target.read_text())
        payload["rows"][0]["status"] = "NOT_RUN_ENVIRONMENT"
        write_strict_json(target, payload)
        self._refresh_closures()

    def corrupt_failed_correctness(self) -> None:
        target = self.root / "industrial-v2-index.json"
        payload = json.loads(target.read_text())
        payload["rows"][0]["correctness_witness_pass"] = False
        payload["correctness_witness_pass"] = False
        write_strict_json(target, payload)
        self._refresh_closures()

    def corrupt_identity(self) -> None:
        target = self.root / "industrial-v2-run-config.json"
        payload = json.loads(target.read_text())
        payload["run_config"]["batch_size"] = 999  # semantic change w/o identity change
        write_strict_json(target, payload)
        self._refresh_closures()

    def corrupt_nan(self) -> None:
        target = self.root / "industrial-v2-index.json"
        text = target.read_text().replace('"rows"', '"rows"', 1)
        target.write_text(text + "")  # keep valid, then inject NaN textually
        raw = target.read_text()
        target.write_text(raw.replace('"claim_level": "REPRODUCED_INTERNAL"', '"claim_level": NaN'))
        self._refresh_closures()


@pytest.fixture
def family(tmp_path: Path) -> FamilyBuilder:
    builder = FamilyBuilder(tmp_path / "v2")
    builder.write()
    ok, why = builder.verify()
    assert ok, f"synthetic family must close before corruption: {why}"
    return builder


CORRUPTIONS = [
    "corrupt_remove_member",
    "corrupt_extra_member",
    "corrupt_wrong_sha",
    "corrupt_stale_sums",
    "corrupt_manifest_bytes",
    "corrupt_unsynchronized_gpu_timing",
    "corrupt_missing_gpu_memory_stat",
    "corrupt_not_run_environment_row",
    "corrupt_failed_correctness",
    "corrupt_identity",
    "corrupt_nan",
]


@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_b2_manifest_hostile_table(tmp_path: Path, corruption: str) -> None:
    """Every corruption of the closed family must be REJECTED — a fresh
    family is built per case, corrupted once, then verified."""
    builder = FamilyBuilder(tmp_path / f"v2-{corruption}")
    builder.write()
    getattr(builder, corruption)()
    ok, why = builder.verify()
    assert not ok, f"{corruption} must be rejected by the verifier"
    assert why


def test_b2_partial_canonical_root_rejected(tmp_path: Path) -> None:
    family = FamilyBuilder(tmp_path / "v2")
    family.write()
    (family.root / "industrial-v2-hopkins.json").unlink()
    ok, why = family.verify()
    assert not ok and "incomplete" in why


def test_b2_v11_root_is_never_the_v2_root() -> None:
    assert RESULTS_ROOT == "benchmarks/results/industrial-v2"
    assert "results/industrial/" not in RESULTS_ROOT


# ---- verifier CLI + claim generator -----------------------------------------------


def test_verifier_cli_fails_on_missing_root() -> None:
    import subprocess
    import sys

    proc = subprocess.run(  # noqa: S603
        [sys.executable, "scripts/verify_industrial_v2_artifacts.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1
    assert "V2 VERIFIER: FAIL" in proc.stderr


def test_b2j_readme_ib2_reference_without_claims_fails(tmp_path: Path, monkeypatch) -> None:
    import scripts.generate_industrial_v2_claims as generator

    monkeypatch.setattr(generator, "GENERATED_JSON", tmp_path / "none.json")
    monkeypatch.setattr(generator, "README", tmp_path / "README.md")
    readme = tmp_path / "README.md"
    readme.write_text("Our GPU speedup is IB2-GPU-E2E-4096 = 3.2x!")
    assert generator._extract_ib2_ids(readme.read_text()) == ["IB2-GPU-E2E-4096"]
