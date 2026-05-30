"""ASML Brion Tachyon simulator adapter.

Real Tachyon integration requires the Tachyon toolchain on ``PATH`` plus a
license. The adapter supports a ``mock_mode`` that produces plausible
synthetic results for testing and CI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from openlithohub.simulators.base import BaseSimulator, SimulatorConfig, SimulatorResult
from openlithohub.simulators.commercial import (
    PreflightStatus,
    ToolchainError,
    _check_license_env,
    _find_binary,
    read_aerial_image,
    run_subprocess,
    write_mask_gdsii,
)

_TACHYON_BIN = "tachyon_lm"
_LICENSE_VARS = ("TACHYON_LICENSE", "LM_LICENSE_FILE")


class TachyonSimulator(BaseSimulator):
    """Adapter for ASML Brion Tachyon lithography simulator.

    ``config.extra`` should carry:

    * ``tachyon_home`` (str) -- install root containing the Tachyon binary.
    * ``recipe`` (str) -- path to the ``.tcl`` recipe to execute.
    * ``layer_map`` (dict[str, int], optional) -- OpenLithoHub layer to
      Tachyon layer mapping.
    * ``mock_mode`` (bool, optional) -- run in mock mode without the real
      toolchain. Defaults to ``False``.
    """

    name = "tachyon"
    differentiable = False

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        super().__init__(config)
        self.mock_mode: bool = bool(self.config.extra.get("mock_mode", False))
        if not self.mock_mode:
            self._validate_config()

    def _validate_config(self) -> None:
        extra = self.config.extra
        for key in ("tachyon_home", "recipe"):
            if key not in extra:
                raise ValueError(
                    f"TachyonSimulator requires config.extra[{key!r}]; "
                    f"got keys={sorted(extra.keys())}"
                )

    def preflight(self) -> PreflightStatus:
        """Check Tachyon binary and license availability."""
        if self.mock_mode:
            return PreflightStatus(ok=True, tool_found=True, license_ok=True)

        extra = self.config.extra
        home = extra.get("tachyon_home", "")
        search = (str(Path(home) / "bin"),) if home else ()
        binary = _find_binary(_TACHYON_BIN, search)
        tool_found = binary is not None
        license_ok = _check_license_env(_LICENSE_VARS)
        messages: list[str] = []
        if not tool_found:
            messages.append(
                f"Cannot find {_TACHYON_BIN} on PATH or in {home}/bin. "
                "Install Tachyon or set tachyon_home in config.extra."
            )
        if not license_ok:
            messages.append(
                "No Tachyon license detected. Set TACHYON_LICENSE or "
                "LM_LICENSE_FILE environment variable."
            )
        return PreflightStatus(
            ok=tool_found and license_ok,
            tool_found=tool_found,
            license_ok=license_ok,
            messages=tuple(messages),
        )

    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        """Run Tachyon forward simulation.

        In mock mode, produces a Gaussian-blurred approximation of the mask
        that mimics a low-fidelity aerial image without requiring the
        commercial tool.
        """
        spatial_shape = mask.shape[-2:]
        if self.mock_mode:
            return self._mock_simulate(mask)

        status = self.preflight()
        if not status.ok:
            raise ToolchainError("Tachyon preflight failed: " + "; ".join(status.messages))

        with tempfile.TemporaryDirectory(prefix="tachyon_") as tmpdir:
            mask_file = Path(tmpdir) / "mask.txt"
            write_mask_gdsii(mask, mask_file, self.config.pixel_size_nm / 1000.0)
            output_file = Path(tmpdir) / "aerial.txt"
            cmd = self._build_command(mask_file, output_file)
            run_subprocess(cmd, cwd=tmpdir)
            if not output_file.exists():
                raise ToolchainError(
                    "Tachyon completed but produced no output file. Check recipe configuration."
                )
            aerial = read_aerial_image(output_file, (int(spatial_shape[0]), int(spatial_shape[1])))

        return SimulatorResult(
            aerial=aerial,
            resist=(aerial > self.config.threshold).float(),
            backend=self.name,
            metadata={"mock": False, "recipe": self.config.extra.get("recipe", "")},
        )

    def _build_command(self, mask_file: Path, output_file: Path) -> list[str]:
        extra = self.config.extra
        home = extra["tachyon_home"]
        binary = _find_binary(_TACHYON_BIN, (str(Path(home) / "bin"),))
        if binary is None:
            binary = _TACHYON_BIN
        recipe = extra["recipe"]
        return [
            binary,
            "-recipe",
            recipe,
            "-mask",
            str(mask_file),
            "-output",
            str(output_file),
        ]

    def _mock_simulate(self, mask: torch.Tensor) -> SimulatorResult:
        """Produce a synthetic aerial image using a simple Gaussian blur."""
        import torch.nn.functional as functional

        mask_2d = mask.detach().cpu().float()
        if mask_2d.ndim == 4:
            mask_2d = mask_2d.squeeze(0).squeeze(0)
        elif mask_2d.ndim == 3:
            mask_2d = mask_2d.squeeze(0)

        kernel_size = 9
        sigma = 2.0
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = gauss_1d[:, None] * gauss_1d[None, :]
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        padded = mask_2d.unsqueeze(0).unsqueeze(0)
        aerial = functional.conv2d(padded, kernel, padding=kernel_size // 2).squeeze(0).squeeze(0)

        # Clamp to [0, 1] and apply dose
        aerial = aerial * self.config.dose
        aerial = aerial.clamp(0.0, 1.0)

        # Restore batch dimensions if input was batched
        if mask.ndim == 4:
            aerial = aerial.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            aerial = aerial.unsqueeze(0)

        resist = (aerial > self.config.threshold).float()
        return SimulatorResult(
            aerial=aerial,
            resist=resist,
            backend=self.name,
            metadata={"mock": True},
        )
