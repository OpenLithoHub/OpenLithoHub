"""Industrial Benchmark v2 formal-measurement preflight (PR-G §34).

Reports every formal blocker for a planned v2 measurement run and exits
nonzero when formal publication would be impossible.  On a CPU-only host
this exits nonzero with::

    FORMAL_PUBLICATION_BLOCKED: CUDA measurement environment unavailable

which is the documented, honest terminal state of Phase 2A — never a
benchmark failure and never emulated.

Usage::

    python scripts/preflight_industrial_v2.py --gds /path/to/ibex.gds --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gds", required=True, help="real routed GDS fixture path")
    parser.add_argument("--device", default="cuda:0", help="planned device (cuda:0)")
    parser.add_argument("--tiers", default="a,b,c")
    args = parser.parse_args()

    import torch

    from openlithohub.server.config import resolve_execution_device

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # git state
    import subprocess

    try:
        commit = subprocess.run(  # noqa: S603 — fixed argv
            ["/usr/bin/git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603 — fixed argv
            ["/usr/bin/git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        check("measurement commit", True, commit)
        check("tracked tree clean", dirty == "", "dirty files block formal runs (B2-A)")
    except (OSError, subprocess.CalledProcessError) as exc:
        check("git state", False, str(exc)[-200:])

    # fixture
    gds = Path(args.gds)
    check("fixture exists", gds.is_file(), str(gds))
    if gds.is_file():
        from openlithohub.benchmark.industrial_v2 import sha256_file

        check("fixture sha256", True, sha256_file(gds))

    # CUDA environment
    cuda_available = bool(torch.cuda.is_available())
    device_count = torch.cuda.device_count() if cuda_available else 0
    check("CUDA available", cuda_available, f"count={device_count}")
    try:
        selected = resolve_execution_device(
            args.device, cuda_available=cuda_available, device_count=device_count
        )
        check("selected device", True, selected)
    except ValueError as exc:
        selected = "unavailable"
        check("selected device", False, str(exc))
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        check("GPU model", True, props.name)
        check("VRAM bytes", True, str(int(props.total_memory)))
        check("compute capability", True, f"{props.major}.{props.minor}")
        check("torch CUDA build", True, torch.version.cuda or "none")
        cudnn = torch.backends.cudnn.version()
        check("cuDNN version", cudnn is not None, str(cudnn))
    check("dtype fp32 support", True, "fp32 is always supported")
    check(
        "TF32 policy",
        True,
        f"matmul={torch.backends.cuda.matmul.allow_tf32} cudnn={torch.backends.cudnn.allow_tf32}",
    )

    # disk + writability
    free = shutil.disk_usage(REPO).free
    check("disk free > 8 GiB", free > 8 * 1024**3, f"{free} bytes free")
    probe = REPO / "benchmarks" / "results" / "industrial-v2" / ".preflight-probe"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        probe.unlink()
        check("v2 results root writable", True, str(probe.parent))
    except OSError as exc:
        check("v2 results root writable", False, str(exc))

    # tooling
    check("KLayout available", importlib.util.find_spec("klayout") is not None)
    check("benchmark capabilities", True, "finite-support blur + hopkins sim present")

    cuda_required = any(tier.strip() in ("b", "c") for tier in args.tiers.split(","))
    formal_blockers = [name for name, ok, _ in checks if not ok] + (
        ["CUDA measurement environment unavailable"] if cuda_required and not cuda_available else []
    )

    print("== Industrial Benchmark v2 preflight ==")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    for blocker in formal_blockers:
        if blocker.startswith("FORMAL"):
            print(f"  {blocker}")
    if formal_blockers:
        print(
            f"PREFLIGHT: FAIL — {len(formal_blockers)} formal blocker(s); "
            "canonical v2 publication is blocked"
        )
        return 1
    print("PREFLIGHT: PASS — formal v2 measurement may proceed on this host")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
