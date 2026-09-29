"""Source-owned cross-platform measurement support (GPU Authority Repair).

One implementation for every cross-platform measurement primitive the
formal benchmark harnesses share: CUDA context initialization, git-state
authority, host peak RSS and GPU driver/device identity.  A formal run
must need NO operator shims on any host — Windows included
(``C:\\usr\\bin\\git.exe``, a ``resource.py`` copy into site-packages, a
nested measurement-logs ``.gitignore``; issues #56/#74/#75).  Each helper
here is the single authority for its concern and fails closed instead of
guessing.
"""

from __future__ import annotations

import shutil
import subprocess  # noqa: S404 — fixed-argv git query only
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "ForwardExecutionWitness",
    "gpu_driver_identity",
    "host_peak_rss_bytes",
    "initialize_cuda_measurement_device",
    "measurement_git_state",
]


# ---- CUDA initialization (GPU Authority Repair §4 / issue #56 defect 1) ---------


def initialize_cuda_measurement_device(device: str) -> str:
    """Initialize the CUDA context on the measurement device BEFORE any
    peak-stat reset or claim-bearing measurement.

    Historical defect (issue #56, defect 1): the Tier B worker executed a
    CPU reference pass first and then called
    ``torch.cuda.reset_peak_memory_stats(device)`` — with no CUDA context
    initialized on the host, that raises ``RuntimeError: Invalid device
    argument``.  This helper validates the CUDA target, selects the
    device, performs a minimal device touch/allocation and synchronizes;
    it returns only after the context is initialized and RAISES on any
    failure.  It is never catch-and-ignore: a claim-bearing GPU
    measurement must not proceed on an uninitialized or wrong device.
    """
    import torch

    if not device.startswith("cuda"):
        raise ValueError(
            f"initialize_cuda_measurement_device requires a cuda device, got {device!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA measurement environment unavailable — formal GPU "
            "measurement must not proceed on a CPU-only host"
        )
    index = 0
    if ":" in device:
        suffix = device.split(":", 1)[1]
        if suffix:
            index = int(suffix)
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(f"requested CUDA device {device!r} is not selectable on this host")
    torch.cuda.set_device(index)
    touch = torch.zeros(1, dtype=torch.float32, device=device)
    touch.add_(0.0)
    torch.cuda.synchronize(device)
    return str(touch.device)


# ---- actual-CUDA execution witness (GPU Authority Repair §7/§15) -----------------


class ForwardExecutionWitness:
    """The tensor-path authority for an actual-CUDA claim.

    A CUDA claim is proven by the devices the forward tensors actually
    occupied — never by ``torch.cuda.is_available()``, a device argument
    or a synchronize call.  The harness records every forward
    execution's input device (after H2D) and output device BEFORE the
    D2H copy; the summary passes only when at least one forward executed
    and EVERY execution ran on the requested CUDA device.
    """

    def __init__(self, requested_device: str) -> None:
        self.requested_device = requested_device
        self._last_input = ""
        self._last_output = ""
        self._cuda_executions = 0
        self._cpu_executions = 0

    def record(self, input_tensor: Any, output_before_d2h: Any) -> None:
        """Record one forward execution: the input after its H2D move and
        the output before the D2H copy."""
        input_device = str(input_tensor.device)
        output_device = str(output_before_d2h.device)
        self._last_input = input_device
        self._last_output = output_device
        if input_device.startswith("cuda") and output_device.startswith("cuda"):
            self._cuda_executions += 1
        else:
            self._cpu_executions += 1

    def summary(self) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "forward_input_device": self._last_input,
            "forward_output_device_before_d2h": self._last_output,
            "cuda_forward_executions": self._cuda_executions,
            "cpu_forward_executions": self._cpu_executions,
            "cuda_execution_witness_pass": bool(
                self.requested_device.startswith("cuda")
                and self._cuda_executions > 0
                and self._cpu_executions == 0
                and self._last_input.startswith("cuda")
                and self._last_output.startswith("cuda")
            ),
        }


# ---- cross-platform git-state authority (GPU Authority Repair §8) ----------------


def measurement_git_state(repo: str | Path) -> tuple[str, bool]:
    """``(commit, tracked_tree_clean)`` — the cross-platform git-state
    authority for formal measurement.

    Resolves git with ``shutil.which("git")`` and invokes it with fixed
    argv (``rev-parse HEAD``, ``status --porcelain``).  No hard-coded
    ``/usr/bin/git`` (unresolvable from Windows Python; issue #56
    accommodation) and no ``C:\\usr\\bin\\git.exe`` shim (issue #75
    accommodation 1): every host uses its own git like any other tool.
    Fails closed — raises — when git is unavailable; a measurement must
    never proceed without git-state authority.
    """
    git_exec = shutil.which("git")
    if git_exec is None:
        raise RuntimeError(
            "git executable not found on PATH — measurement git-state "
            "authority requires git and fails closed without it"
        )

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603 — fixed-argv git query
            [git_exec, "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    return commit, dirty == ""


# ---- cross-platform host peak RSS (GPU Authority Repair §9) ----------------------


def host_peak_rss_bytes() -> int:
    """Platform-normalized host peak RSS in bytes.

    Linux/macOS keep the native ``resource`` semantics (``ru_maxrss`` is
    bytes on macOS, KiB on Linux — normalized here); Windows uses the
    source-owned ``GetProcessMemoryInfo`` implementation below.  No
    ``.venv/Lib/site-packages/resource.py`` shim is ever required
    (issue #75 accommodation 2).
    """
    if sys.platform == "win32":
        return _windows_peak_rss_bytes()
    return _posix_peak_rss_bytes()


def _posix_peak_rss_bytes() -> int:
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _windows_peak_rss_bytes() -> int:
    """Peak working set in bytes via ``psapi.GetProcessMemoryInfo`` — a
    source-owned Windows implementation, no operator shim."""
    import ctypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    current_process = ctypes.c_void_p(-1)  # GetCurrentProcess() pseudo-handle
    ok = ctypes.CDLL("psapi.dll").GetProcessMemoryInfo(
        current_process, ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise OSError("GetProcessMemoryInfo failed — host peak RSS unavailable")
    return int(counters.PeakWorkingSetSize)


# ---- GPU driver/device identity (GPU Authority Repair §10) -----------------------


def gpu_driver_identity(device_index: int = 0) -> dict[str, str]:
    """Driver/device identity for a formal environment lock.

    Policy: try nvidia-smi/NVML first; on success record the driver
    version with source ``nvidia-smi``.  On Windows, when the nvidia-smi
    query fails (issue #74/#75: the executable is present but NVML
    fails with ``Failed to initialize NVML``), use the source-owned
    registry fallback and record driver version + a stable device
    identifier with source ``windows-fallback``.  Executable presence is
    never success: a failed query records ``unavailable`` and formal
    authority must reject the empty identity — never silently accept it.
    """
    version = _nvidia_smi_driver_version()
    if version:
        return {
            "driver_version": version,
            "driver_identity_source": "nvidia-smi",
            **_stable_device_identifier(device_index),
        }
    if sys.platform == "win32":
        fallback = _windows_display_driver_identity()
        if fallback["driver_version"]:
            return {
                "driver_version": fallback["driver_version"],
                "driver_identity_source": "windows-fallback",
                **_stable_device_identifier(device_index, fallback["device_identifier"]),
            }
    return {
        "driver_version": "",
        "driver_identity_source": "unavailable",
        "device_identifier": "",
        "device_identifier_type": "none",
    }


def _nvidia_smi_driver_version() -> str:
    import subprocess

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return ""
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no user input
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


_WINDOWS_DISPLAY_CLASS_KEY = (
    r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
)


def _windows_display_driver_identity() -> dict[str, str]:
    """Source-owned Windows fallback: the display-class registry stores
    the installed display adapter's description, driver version and PnP
    matching device id — readable without NVML, which is exactly what
    fails on the issue #74/#75 host.  Returns empty strings when the
    registry cannot identify an NVIDIA adapter."""
    import importlib

    winreg: Any = importlib.import_module("winreg")  # Windows-only by design

    for subkey_index in range(64):
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                f"{_WINDOWS_DISPLAY_CLASS_KEY}\\{subkey_index:04d}",
            ) as key:
                description = str(winreg.QueryValueEx(key, "DriverDesc")[0])
                if "NVIDIA" not in description.upper():
                    continue
                version = ""
                device_id = ""
                for value_name in ("DriverVersion", "MatchingDeviceId"):
                    try:
                        value = str(winreg.QueryValueEx(key, value_name)[0]).strip()
                    except OSError:
                        value = ""
                    if value_name == "DriverVersion":
                        version = value
                    else:
                        device_id = value
                if version:
                    return {
                        "driver_description": description,
                        "driver_version": version,
                        "device_identifier": device_id,
                    }
        except OSError:
            continue
    return {"driver_description": "", "driver_version": "", "device_identifier": ""}


def _stable_device_identifier(device_index: int, fallback: str = "") -> dict[str, str]:
    """A stable device identifier: the CUDA driver UUID when torch
    exposes it, else the host-provided stable identifier — always
    recorded WITH its type, never an unnamed guess."""
    try:
        import torch

        props = torch.cuda.get_device_properties(device_index)
        uuid = getattr(props, "uuid", None)
        if uuid is not None:
            text = uuid.hex() if isinstance(uuid, bytes) else str(uuid)
            text = text.strip()
            if text and text != "None":
                return {"device_identifier": text, "device_identifier_type": "cuda-uuid"}
    except Exception:  # noqa: BLE001,S110 — best-effort identifier on any host
        pass
    if fallback:
        return {
            "device_identifier": fallback,
            "device_identifier_type": "windows-pnp-matching-device-id",
        }
    return {"device_identifier": "", "device_identifier_type": "none"}
