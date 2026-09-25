"""Industrial Scale verifier + claim firewall hostile tests (S10/S11).

Builds a real provisional family by running the harness on a tiny
synthetic fixture, then corrupts it in every way the verifier must
reject.  Also proves the headline firewall: no ISC-* claim may reach a
README while the scale track is provisional.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "benchmarks" / "industrial-scale" / "run_scale_benchmark.py"
PREPARE_SCRIPT = REPO / "scripts" / "prepare_industrial_scale_fixture.py"
VERIFIER = REPO / "scripts" / "verify_industrial_scale_artifacts.py"
CLAIMS = REPO / "scripts" / "generate_industrial_scale_claims.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def provisional_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete provisional family built by the REAL harness on a
    synthetic routed fixture."""
    tmp_path = tmp_path_factory.mktemp("scale-verifier")
    source_root = tmp_path / "PDB"
    (source_root / "layout" / "sky130hd" / "ibex").mkdir(parents=True)

    import klayout.db as db

    ly = db.Layout()
    ly.dbu = 0.001
    li = ly.layer(66, 44)
    cell = ly.create_cell("ibex_core")
    cell.shapes(li).insert(db.Box(0, 0, 256, 256))
    ly.write(str(source_root / "layout" / "sky130hd" / "ibex" / "ibex.gds"))

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(source_root), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "f@e.com")
    git("config", "user.name", "f")
    git("add", "-A")
    git("commit", "-q", "-m", "pdb")
    commit = git("rev-parse", "HEAD")

    prepare = _load("prepare_s4", PREPARE_SCRIPT)
    fixtures = tmp_path / "fixtures" / "ibex"
    prepare.prepare_fixture(
        source_root=source_root,
        design="ibex",
        selected_layer="66:44",
        pixel_nm=1.0,
        output_dir=fixtures,
        expected_commit=commit,
    )
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(HARNESS),
            "--fixture-manifest",
            str(fixtures / "fixture-manifest.json"),
            "--gds",
            str(fixtures / "ibex.gds"),
            "--lanes",
            "a",
            "--windows",
            "128",
            "--tile",
            "64",
            "--repeats",
            "2",
            "--warmup",
            "0",
            "--device",
            "cpu",
            "--out-root",
            str(tmp_path / "scale"),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr
    return Path(json.loads(proc.stdout)["workspace"]) / "family"


def _verify(root: Path, *flags: str):
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(VERIFIER), "--root", str(root), *flags],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
    )
    return proc


def test_provisional_family_passes_structural_tier(provisional_workspace: Path) -> None:
    proc = _verify(provisional_workspace)
    assert proc.returncode == 0, proc.stderr
    assert "structural tier" in proc.stdout


def test_provisional_family_fails_formal_tier(provisional_workspace: Path) -> None:
    proc = _verify(provisional_workspace, "--require-formal")
    assert proc.returncode == 1
    assert "provisional" in proc.stderr


def _corrupted_family(provisional_workspace: Path, tmp_path: Path, corrupt) -> Path:
    """Copy the family, apply a corruption, refresh closures unless the
    corruption is meant to break them."""
    target = tmp_path / "family-copy"
    if target.exists():
        import shutil

        shutil.rmtree(target)
        shutil.rmtree(target, ignore_errors=True)
    import shutil

    shutil.copytree(provisional_workspace, target)
    corrupt(target)
    return target


def test_missing_member_rejected(provisional_workspace: Path, tmp_path: Path) -> None:
    target = _corrupted_family(
        provisional_workspace, tmp_path, lambda r: (r / "manifest.json").unlink()
    )
    proc = _verify(target)
    assert proc.returncode == 1 and "incomplete" in proc.stderr


def test_unknown_file_rejected(provisional_workspace: Path, tmp_path: Path) -> None:
    def add_rogue(root: Path) -> None:
        (root / "rogue.json").write_text("{}")

    target = _corrupted_family(provisional_workspace, tmp_path, add_rogue)
    proc = _verify(target)
    assert proc.returncode == 1 and "unknown file" in proc.stderr


def test_tampered_member_rejected(provisional_workspace: Path, tmp_path: Path) -> None:
    def tamper(root: Path) -> None:
        member = root / "industrial-scale-index.json"
        payload = json.loads(member.read_text())
        payload["rows"][0]["aggregate_median_s"] = 0.0001  # too good to be true
        member.write_text(json.dumps(payload))

    target = _corrupted_family(provisional_workspace, tmp_path, tamper)
    proc = _verify(target)
    assert proc.returncode == 1 and "SHA256 mismatch" in proc.stderr


def test_identity_drift_rejected(provisional_workspace: Path, tmp_path: Path) -> None:
    def drift(root: Path) -> None:
        member = root / "industrial-scale-run-config.json"
        payload = json.loads(member.read_text())
        payload["run_config"]["tile_size"] = 9999  # semantic change, same identity
        member.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    target = _corrupted_family(provisional_workspace, tmp_path, drift)
    # refresh sums AND manifest byte counts so ONLY the identity
    # recomputation can catch the tampered semantic knob
    manifest = json.loads((target / "manifest.json").read_text())
    for entry in manifest["members"]:
        entry["bytes"] = (target / entry["name"]).stat().st_size
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    members = sorted(set(p.name for p in target.iterdir() if p.is_file()) - {"SHA256SUMS.txt"})
    lines = [
        f"{hashlib.sha256((target / name).read_bytes()).hexdigest()}  {name}" for name in members
    ]
    (target / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    proc = _verify(target)
    assert proc.returncode == 1 and "identity drift" in proc.stderr


def test_nan_injected_rejected(provisional_workspace: Path, tmp_path: Path) -> None:
    def inject_nan(root: Path) -> None:
        member = root / "industrial-scale-index.json"
        text = member.read_text().replace('"aggregate_median_s"', '"aggregate_median_X"')
        member.write_text(text)

    target = _corrupted_family(provisional_workspace, tmp_path, inject_nan)
    proc = _verify(target)
    assert proc.returncode == 1


# ---- claim firewall (S11) ----------------------------------------------------------


def test_claim_generator_emits_only_provisional(
    provisional_workspace: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "derived"
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(CLAIMS),
            "--root",
            str(provisional_workspace),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr
    claims = json.loads((out_dir / "industrial-scale-claims.json").read_text())
    assert claims["level"] == "PROVISIONAL_INTERNAL"
    assert claims["note"].startswith("PROVISIONAL / INTERNAL")
    assert all(claim["admitted"] is False for claim in claims["claims"])
    assert all(claim["headline_eligible"] is False for claim in claims["claims"])
    assert "NOT PERFORMANCE AUTHORITY" in (out_dir / "industrial-scale-claims.md").read_text()


def test_headline_firewall_blocks_isc_in_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _load("scale_claims_check", CLAIMS)
    readme = tmp_path / "README.md"
    readme.write_text("Our scale number is ISC-A-4096 = 1.2 GPx/s!")
    monkeypatch.setattr(generator, "README", readme)
    monkeypatch.setattr(generator, "README_ZH", tmp_path / "README_zh.md")
    assert generator.check_readme_firewall() == 1


def test_headline_firewall_passes_clean_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _load("scale_claims_check2", CLAIMS)
    readme = tmp_path / "README.md"
    readme.write_text("No scale claims here yet.")
    (tmp_path / "README_zh.md").write_text("尚无 scale 数字。")
    monkeypatch.setattr(generator, "README", readme)
    monkeypatch.setattr(generator, "README_ZH", tmp_path / "README_zh.md")
    assert generator.check_readme_firewall() == 0
