"""GPU authority evidence packaging (GPU Authority Repair §25).

Builds a DETERMINISTIC authority bundle for one scale run — small JSON
evidence only, never the tens-of-GB raw sink buffers:

* run config, environment lock, fixture member, row JSONs, run-summary;
* the canonical family (all members + SHA256SUMS);
* preflight output and verifier output (operator-attached logs);
* git HEAD + clean-tree evidence (recomputed at packaging time);
* the environment identity capture (driver/device identity + source);
* the Lane-A campaign manifest when the run is part of a campaign.

Raw sink buffers (``*.npy``) are RETAINED LOCALLY and recorded in the
bundle only as SHA-256 + byte count.

Determinism: identical inputs produce byte-identical archives
(sorted members, zeroed mtimes/uids, gzip mtime=0).  Output:
``gpu-authority-evidence-<run-id>.tar.gz`` plus the archive SHA-256.

Usage::

    python scripts/package_gpu_authority_evidence.py \\
        --workspace benchmarks/results/industrial-scale/runs/<run-id> \\
        --preflight-log measurement-logs/scale-preflight.txt \\
        --verifier-log measurement-logs/verifier.txt \\
        --out-dir evidence/
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openlithohub.benchmark.industrial_scale import (  # noqa: E402
    SCALE_CANONICAL_FAMILY,
)

EVIDENCE_SCHEMA = "OpenLithoHub.gpu-authority-evidence.v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_authority_evidence() -> dict[str, object]:
    """git HEAD + clean-tree evidence, recomputed AT PACKAGING TIME via
    the shared cross-platform helper (GPU Authority Repair §8/§21)."""
    from openlithohub.benchmark.measurement_support import measurement_git_state

    commit, clean = measurement_git_state(REPO)
    return {"git_head": commit, "tracked_tree_clean": clean}


def environment_identity_capture(family_dir: Path) -> dict[str, object]:
    """The environment identity capture from the family freeze — driver
    version, identity SOURCE, stable device identifier and their types
    (GPU Authority Repair §10)."""
    freeze = json.loads((family_dir / "industrial-scale-distribution-freeze.txt").read_text())
    lock = freeze.get("gpu") or {}
    return {
        "platform": lock.get("platform"),
        "driver_version": lock.get("driver_version"),
        "driver_identity_source": lock.get("driver_identity_source"),
        "device_identifier": lock.get("device_identifier"),
        "device_identifier_type": lock.get("device_identifier_type"),
        "devices": lock.get("devices", []),
        "torch_version": lock.get("torch_version"),
        "torch_cuda_version": lock.get("torch_cuda_version"),
        "cudnn_version": lock.get("cudnn_version"),
        "tf32_matmul": lock.get("tf32_matmul"),
        "tf32_cudnn": lock.get("tf32_cudnn"),
    }


def sink_buffer_records(workspace: Path) -> list[dict[str, object]]:
    """Raw sink buffers are retained LOCALLY and recorded by SHA-256 +
    byte count only (§25: never upload tens of GB of raw .npy)."""
    records: list[dict[str, object]] = []
    for path in sorted(workspace.glob("*.npy")):
        records.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "retention": "local-only",
            }
        )
    return records


def collect_evidence_members(
    *,
    workspace: Path,
    family_dir: Path,
    preflight_log: Path | None,
    verifier_log: Path | None,
    campaign_manifest: Path | None,
) -> dict[str, bytes]:
    """Collect every bundle member as (archive path -> bytes)."""
    members: dict[str, bytes] = {}

    def add(name: str, path: Path) -> None:
        members[name] = path.read_bytes()

    # run config / environment lock / fixture / row JSONs / run-summary
    for path in sorted(workspace.iterdir()):
        if path.is_file() and path.suffix == ".json":
            add(f"workspace/{path.name}", path)
    # canonical family (all members incl. SHA256SUMS)
    for name in sorted(SCALE_CANONICAL_FAMILY):
        member_path = family_dir / name
        if member_path.is_file():
            add(f"family/{name}", member_path)
    # operator-attached preflight / verifier outputs
    if preflight_log is not None:
        add("logs/scale-preflight.txt", preflight_log)
    if verifier_log is not None:
        add("logs/verifier.txt", verifier_log)
    # Lane-A campaign manifest when the run belongs to a campaign
    if campaign_manifest is not None:
        add("campaign/lane-a-campaign.json", campaign_manifest)

    run_config = json.loads((family_dir / "industrial-scale-run-config.json").read_text())
    evidence_manifest = {
        "schema": EVIDENCE_SCHEMA,
        "run_identity": run_config.get("run_identity"),
        "measurement_commit": run_config.get("measurement_commit"),
        "git_authority": git_authority_evidence(),
        "environment_identity": environment_identity_capture(family_dir),
        "sink_buffers": sink_buffer_records(workspace),
        "campaign_manifest": campaign_manifest.name if campaign_manifest else None,
        "members": sorted(members),
    }
    members["evidence-manifest.json"] = (
        json.dumps(evidence_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    return members


def package_evidence(members: dict[str, bytes], run_id: str, out_dir: Path) -> tuple[Path, str]:
    """Write the deterministic tar.gz and return (path, archive sha256)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"gpu-authority-evidence-{run_id}.tar.gz"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(members):
            info = tarfile.TarInfo(name=f"gpu-authority-evidence-{run_id}/{name}")
            payload = members[name]
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tar.addfile(info, io.BytesIO(payload))
    with (
        open(archive_path, "wb") as handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as gz,
    ):
        gz.write(raw.getvalue())
    return archive_path, sha256_file(archive_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="run workspace dir")
    parser.add_argument(
        "--family", default=None, help="canonical family dir (default: <workspace>/family)"
    )
    parser.add_argument("--preflight-log", default=None)
    parser.add_argument("--verifier-log", default=None)
    parser.add_argument("--campaign-manifest", default=None)
    parser.add_argument("--out-dir", default="evidence")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    family_dir = Path(args.family) if args.family else workspace / "family"
    if not workspace.is_dir():
        print(f"EVIDENCE PACKAGING: FAIL — workspace missing: {workspace}", file=sys.stderr)
        return 1
    if not (family_dir / "SHA256SUMS.txt").is_file():
        print(
            f"EVIDENCE PACKAGING: FAIL — family not closed (no SHA256SUMS): {family_dir}",
            file=sys.stderr,
        )
        return 1
    run_config = json.loads((family_dir / "industrial-scale-run-config.json").read_text())
    run_id = str(run_config.get("run_identity") or workspace.name)[:16]

    members = collect_evidence_members(
        workspace=workspace,
        family_dir=family_dir,
        preflight_log=Path(args.preflight_log) if args.preflight_log else None,
        verifier_log=Path(args.verifier_log) if args.verifier_log else None,
        campaign_manifest=Path(args.campaign_manifest) if args.campaign_manifest else None,
    )
    # packaging-time git authority: a dirty tracked tree at packaging is
    # recorded honestly (it cannot invalidate the already-frozen family,
    # but it is evidence the packaging host was not clean)
    git_state = json.loads(members["evidence-manifest.json"])["git_authority"]
    if not git_state["tracked_tree_clean"]:
        print(
            "EVIDENCE PACKAGING: NOTE — tracked tree dirty at packaging time "
            "(recorded in evidence-manifest.json)",
            file=sys.stderr,
        )
    archive_path, archive_sha = package_evidence(members, run_id, Path(args.out_dir))
    print(f"EVIDENCE PACKAGING: BUILT — {archive_path}")
    print(f"archive sha256: {archive_sha}")
    print(f"members: {len(members)} (deterministic: same inputs -> same archive bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
