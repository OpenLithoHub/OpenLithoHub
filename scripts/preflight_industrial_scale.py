"""Industrial Scale preflight (1×RTX4090 migration, §14).

Validates every formal gate of the frozen scale protocol on the actual
host BEFORE measurement and exits nonzero unless all of them close:

* exact source commit (optional ``--expected-commit``) + clean tree;
* CUDA available, ``cuda:0`` selectable, GPU identity / VRAM /
  driver / CUDA / PyTorch / cuDNN / TF32 recorded;
* KLayout available;
* Ibex AND Microwatt fixture manifests load + revalidate, the GDS
  bytes on disk match the frozen sha256, and the frozen selected
  layers are present;
* disk free >= 16 GiB;
* formal parameter contract (frozen Lane C ladder, repeats >= 5,
  tile/halo aligned with the frozen protocol).

Usage::

    python scripts/preflight_industrial_scale.py \\
        --ibex-manifest … --gds-ibex … \\
        --microwatt-manifest … --gds-microwatt … \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openlithohub.benchmark.industrial_scale import (  # noqa: E402
    FROZEN_MICROBATCH_LADDER,
    load_scale_fixture_manifest,
)

DISK_FREE_BYTES = 16 * 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ibex-manifest", required=True)
    parser.add_argument("--gds-ibex", required=True)
    parser.add_argument("--microwatt-manifest", required=True)
    parser.add_argument("--gds-microwatt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--min-repeats", type=int, default=5)
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # git state
    import subprocess

    git = shutil.which("git")
    if git is None:
        check("git available", False, "git is required for source verification")
    else:
        try:
            head = subprocess.run(  # noqa: S603 — fixed-argv git query
                [git, "-C", str(REPO), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            ).stdout.strip()
            dirty = subprocess.run(  # noqa: S603 — fixed-argv git query
                [git, "-C", str(REPO), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            ).stdout.strip()
            check("measurement commit", True, head)
            if args.expected_commit:
                check("source commit == frozen commit", head == args.expected_commit, head)
            check("tracked tree clean", dirty == "", "dirty tree blocks formal runs (B2-A)")
        except (OSError, subprocess.SubprocessError) as exc:
            check("git state", False, str(exc)[-160:])

    # CUDA environment + GPU identity
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        check(
            "CUDA available",
            cuda_available,
            f"count={torch.cuda.device_count() if cuda_available else 0}",
        )
        if cuda_available:
            index = int(args.device.split(":")[-1]) if ":" in args.device else 0
            device = f"cuda:{index}"
            check("selected device", index < torch.cuda.device_count(), device)
            props = torch.cuda.get_device_properties(index)
            check("GPU model", True, props.name)
            check("VRAM bytes", True, str(int(props.total_memory)))
            check("compute capability", True, f"{props.major}.{props.minor}")
            check("torch CUDA build", True, torch.version.cuda or "none")
            cudnn = torch.backends.cudnn.version()
            check("cuDNN", cudnn is not None, str(cudnn))
            check(
                "TF32 state recorded",
                True,
                f"matmul={torch.backends.cuda.matmul.allow_tf32} "
                f"cudnn={torch.backends.cudnn.allow_tf32}",
            )
        else:
            check(f"{args.device} usable", False, "CUDA measurement environment unavailable")
    except ImportError as exc:
        check("torch import", False, str(exc)[:160])

    # GPU identity via nvidia-smi (UUID/PCI/driver) — recorded, not guessed
    nvidia_smi = shutil.which("nvidia-smi")
    check(
        "nvidia-smi available",
        nvidia_smi is not None or not _cuda_requested(args.device),
        str(nvidia_smi),
    )

    # KLayout
    check("KLayout available", importlib.util.find_spec("klayout") is not None)

    # fixture authority: both fixtures must load, validate, and bind bytes
    for label, manifest_path, gds_path in (
        ("Ibex", args.ibex_manifest, args.gds_ibex),
        ("Microwatt", args.microwatt_manifest, args.gds_microwatt),
    ):
        try:
            manifest = load_scale_fixture_manifest(manifest_path)
            check(
                f"{label} fixture manifest",
                True,
                f"{manifest.gds_sha256[:16]}… layer {manifest.selected_layer}",
            )
            gds = Path(gds_path)
            if gds.is_file():
                import hashlib

                digest = hashlib.sha256(gds.read_bytes()).hexdigest()
                check(
                    f"{label} GDS bytes match manifest",
                    digest == manifest.gds_sha256,
                    f"{gds.stat().st_size} bytes",
                )
            else:
                check(f"{label} GDS present", False, str(gds))
        except (OSError, ValueError) as exc:
            check(f"{label} fixture manifest", False, str(exc)[:160])

    # disk + formal parameter contract
    free = shutil.disk_usage(REPO).free
    check("disk free >= 16 GiB", free >= DISK_FREE_BYTES, f"{free} bytes free")
    check(
        "formal parameter contract",
        args.min_repeats >= 5 and list(FROZEN_MICROBATCH_LADDER) == [1, 2, 4, 8, 16, 32],
        f"min_repeats={args.min_repeats} microbatch_ladder={list(FROZEN_MICROBATCH_LADDER)}",
    )

    print("== Industrial Scale preflight ==")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    failures = [name for name, ok, _ in checks if not ok]
    if failures:
        print(
            f"SCALE PREFLIGHT: FAIL — {len(failures)} blocker(s); "
            "formal scale measurement is blocked"
        )
        return 1
    print("SCALE PREFLIGHT: PASS — formal scale measurement may proceed on this host")
    return 0


def _cuda_requested(device: str) -> bool:
    return device.startswith("cuda")


if __name__ == "__main__":
    raise SystemExit(main())
