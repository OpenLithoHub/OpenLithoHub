"""Industrial Scale preflight (GPU Authority Repair §23 — fail-closed).

Validates every formal gate of the frozen scale protocol on the actual
host BEFORE measurement and exits nonzero unless all of them close:

* exact source commit (optional ``--expected-commit``) + clean tree;
* supported OS (Windows / Linux — §22) and CUDA available with
  ``cuda:0`` selectable;
* the FROZEN formal host policy: supported GPU model (RTX 4090 class),
  VRAM >= 15 GiB, compute capability 8.9, device count >= 1;
* driver/device identity NONEMPTY with an EXPLICIT source-owned source
  (nvidia-smi, or the Windows registry fallback — executable presence
  is never identity);
* CUDA build / PyTorch build / cuDNN / TF32 state recorded;
* KLayout available;
* Ibex AND Microwatt fixture manifests load + revalidate, the GDS bytes
  on disk match the frozen sha256, and the frozen selected layer 66:44
  is bound;
* disk free >= 16 GiB;
* formal parameter contract (frozen Lane C ladder, repeats >= 5,
  tile/halo aligned with the frozen protocol);
* a BOUNDED ACTUAL-CUDA forward probe through the same shared helpers
  as the measured path: CUDA init → peak-stat reset → one finite-
  support forward with input AND output devices on CUDA before D2H.
  CUDA-enabled PyTorch + a CPU forward can never PASS this preflight.

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
    FORMAL_SCALE_COMPUTE_CAPABILITY,
    FORMAL_SCALE_MIN_VRAM_BYTES,
    FROZEN_MICROBATCH_LADDER,
    SUPPORTED_SCALE_GPU_MODELS,
    SUPPORTED_SCALE_OS,
    load_scale_fixture_manifest,
)

DISK_FREE_BYTES = 16 * 1024**3
FROZEN_SELECTED_LAYER = "66:44"


def scale_forward_probe(device: str) -> tuple[bool, str]:
    """Bounded actual-CUDA forward probe (GPU Authority Repair §23) — the
    SAME shared helpers as the measured path: initialize CUDA, reset
    peak stats, run one finite-support forward, and require the input
    AND output tensors to have been on CUDA before the D2H copy.  A
    CPU tensor path under a CUDA-enabled PyTorch fails this probe."""
    import torch

    from openlithohub.benchmark.industrial_v2 import BatchedFiniteSupportBlur
    from openlithohub.benchmark.measurement_support import (
        ForwardExecutionWitness,
        initialize_cuda_measurement_device,
    )

    initialize_cuda_measurement_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    blur = BatchedFiniteSupportBlur(radius=4, sigma=1.6)
    blur.to(device)
    tile = torch.rand((64, 64), dtype=torch.float32)
    gpu_in = tile.to(device)
    out = blur.window_forward(gpu_in)
    witness = ForwardExecutionWitness(device)
    witness.record(gpu_in, out)
    summary = witness.summary()
    ok = bool(summary["cuda_execution_witness_pass"] and torch.isfinite(out).all())
    detail = (
        f"requested={summary['requested_device']} "
        f"input={summary['forward_input_device']} "
        f"output_before_d2h={summary['forward_output_device_before_d2h']}"
    )
    return ok, detail


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
    check = lambda name, ok, detail="": checks.append((name, bool(ok), detail))  # noqa: E731

    # git state — GPU Authority Repair §21: the shared cross-platform
    # helper; no hard-coded /usr/bin/git, no Windows git shim.
    import platform
    import subprocess

    try:
        from openlithohub.benchmark.measurement_support import measurement_git_state

        head, clean = measurement_git_state(REPO)
        check("measurement commit", True, head)
        if args.expected_commit:
            check("source commit == frozen commit", head == args.expected_commit, head)
        check("tracked tree clean", clean, "dirty tree blocks formal runs (B2-A)")
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        check("git state", False, str(exc)[-160:])

    # supported OS (§22)
    system = platform.system()
    check(
        "supported OS",
        system in SUPPORTED_SCALE_OS,
        f"{system} (supported: {sorted(SUPPORTED_SCALE_OS)})",
    )

    # CUDA environment + the FROZEN formal host policy (§22)
    from openlithohub.benchmark.measurement_support import gpu_driver_identity

    identity: dict[str, str] = {
        "driver_version": "",
        "driver_identity_source": "unavailable",
        "device_identifier": "",
        "device_identifier_type": "none",
    }
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
            check(
                "supported GPU model",
                props.name in SUPPORTED_SCALE_GPU_MODELS,
                f"{props.name} (supported: {sorted(SUPPORTED_SCALE_GPU_MODELS)})",
            )
            check(
                "VRAM >= frozen minimum",
                int(props.total_memory) >= FORMAL_SCALE_MIN_VRAM_BYTES,
                f"{int(props.total_memory)} bytes (min {FORMAL_SCALE_MIN_VRAM_BYTES})",
            )
            check(
                "compute capability == frozen",
                f"{props.major}.{props.minor}" == FORMAL_SCALE_COMPUTE_CAPABILITY,
                f"{props.major}.{props.minor}",
            )
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
        cuda_available = False
        check("torch import", False, str(exc)[:160])

    # GPU driver/device identity — NONEMPTY with an EXPLICIT source-owned
    # source (GPU Authority Repair §10/§23).  nvidia-smi presence alone
    # is NOT identity: on the #74/#75 host the executable exists while
    # NVML fails, so the Windows registry fallback applies and the
    # identity SOURCE is recorded.  An unavailable identity blocks the
    # formal run here, fail-closed.
    if cuda_available:
        identity = gpu_driver_identity()
        check(
            "driver identity nonempty",
            bool(identity["driver_version"]),
            f"version={identity['driver_version']!r}",
        )
        check(
            "driver identity source explicit",
            identity["driver_identity_source"] not in ("", "unavailable"),
            identity["driver_identity_source"],
        )
        check(
            "stable device identifier",
            bool(identity["device_identifier"]),
            f"{identity['device_identifier_type']}={identity['device_identifier'][:24]}",
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
            check(
                f"{label} frozen selected layer",
                manifest.selected_layer == FROZEN_SELECTED_LAYER,
                manifest.selected_layer,
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

    # bounded ACTUAL-CUDA forward probe (§23) — CUDA-enabled PyTorch with
    # a CPU forward must NOT pass this preflight
    if cuda_available:
        try:
            ok, detail = scale_forward_probe(args.device)
            check("bounded actual-CUDA forward probe", ok, detail)
        except Exception as exc:  # noqa: BLE001 — capability probe
            check("bounded actual-CUDA forward probe", False, repr(exc)[:200])

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


if __name__ == "__main__":
    raise SystemExit(main())
