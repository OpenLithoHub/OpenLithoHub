# ruff: noqa: S603
"""Commercial simulator adapter protocol and shared utilities.

Defines :class:`CommercialSimulatorAdapter`, the protocol that every
vendor-specific adapter (Tachyon, Calibre, …) must satisfy. Provides
common machinery for pre-flight checks, file I/O, and mock mode so each
concrete adapter stays thin.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from openlithohub.simulators.base import SimulatorResult


class ToolchainError(RuntimeError):
    """Raised when a required commercial toolchain is not available."""


@dataclass(frozen=True)
class PreflightStatus:
    ok: bool
    tool_found: bool = False
    license_ok: bool = False
    messages: tuple[str, ...] = ()


@runtime_checkable
class CommercialSimulatorAdapter(Protocol):
    """Protocol shared by all commercial simulator adapters.

    Concrete adapters (Tachyon, Calibre) must provide:
    - :meth:`preflight` — verify toolchain + license
    - :meth:`simulate`  — forward simulation
    - :attr:`mock_mode`  — whether the adapter runs without the real tool
    """

    mock_mode: bool

    def preflight(self) -> PreflightStatus:
        """Check that the toolchain binary exists and a license is reachable."""
        ...

    def simulate(self, mask: torch.Tensor) -> SimulatorResult: ...


def _find_binary(name: str, search_dirs: tuple[str, ...] = ()) -> str | None:
    """Return the absolute path to *name* if found on PATH or in *search_dirs*."""
    path = shutil.which(name)
    if path is not None:
        return path
    for d in search_dirs:
        candidate = Path(d) / name
        if candidate.is_file():
            return str(candidate)
    return None


def _check_license_env(var_names: tuple[str, ...]) -> bool:
    """Return True if at least one of the license env-vars is set and non-empty."""
    return any(os.environ.get(v, "").strip() for v in var_names)


def write_mask_gdsii(mask: torch.Tensor, path: str | Path, pixel_size_um: float = 0.004) -> Path:
    """Write a binary mask tensor to a trivial GDSII-like text file.

    Real adapters would write actual GDSII/OASIS via KLayout or a vendor
    library. This stub writes a compact run-length representation that
    concrete adapters can translate or swap out.

    Args:
        mask: 2-D tensor with values in [0, 1].
        path: Destination file path.
        pixel_size_um: Physical pixel pitch in micrometers.

    Returns:
        The written file path.
    """
    mask_np = mask.detach().cpu().numpy()
    binary = (mask_np > 0.5).astype("uint8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# OpenLithoHub mask export  pixel_size_um={pixel_size_um}\n")
        f.write(f"# shape={binary.shape[0]} {binary.shape[1]}\n")
        for row in binary:
            f.write("".join(str(v) for v in row) + "\n")
    return path


def read_aerial_image(path: str | Path, shape: tuple[int, int]) -> torch.Tensor:
    """Read a plain-text aerial image written by mock simulators.

    Lines starting with ``#`` are ignored; remaining lines contain
    whitespace-separated float values.
    """
    rows: list[list[float]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(v) for v in line.split()])
    if not rows:
        return torch.zeros(shape)
    return torch.tensor(rows, dtype=torch.float32)


def run_subprocess(
    cmd: list[str],
    timeout: int = 600,
    cwd: str | Path | None = None,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an external simulator binary with standard error handling."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            check=True,
        )
    except FileNotFoundError as exc:
        raise ToolchainError(f"Binary not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolchainError(f"Simulator timed out after {timeout}s: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise ToolchainError(f"Simulator exited with code {exc.returncode}:\n{exc.stderr}") from exc
