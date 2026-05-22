"""``openlithohub simulate`` — run a registered simulator on a mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer

from openlithohub.simulators import (
    SimulatorConfig,
    describe_simulators,
    get_simulator,
    list_simulators,
)

simulate_app = typer.Typer(help="Run a forward simulator on a mask.", no_args_is_help=True)


@simulate_app.command()
def run(
    mask_path: Path = typer.Argument(..., help="Path to mask .npy or grayscale image."),
    backend: str = typer.Option("hopkins", "--backend", "-b", help="Simulator backend."),
    out_path: Path = typer.Option(
        Path("aerial.npy"), "--out", "-o", help="Where to write the aerial image."
    ),
    pixel_size_nm: float = typer.Option(1.0, help="Pixel size in nm."),
    wavelength_nm: float = typer.Option(193.0, help="Exposure wavelength in nm."),
    na: float = typer.Option(1.35, help="Numerical aperture."),
    sigma: float = typer.Option(0.7, help="Outer partial-coherence factor."),
    threshold: float = typer.Option(0.225, help="Resist threshold."),
    dose: float = typer.Option(1.0, help="Dose multiplier."),
) -> None:
    """Forward-simulate ``mask_path`` with the chosen ``backend``."""

    if not mask_path.exists():
        raise typer.BadParameter(f"mask not found: {mask_path}")

    mask = _load_mask(mask_path)
    config = SimulatorConfig(
        wavelength_nm=wavelength_nm,
        na=na,
        sigma=sigma,
        pixel_size_nm=pixel_size_nm,
        dose=dose,
        threshold=threshold,
    )
    sim = get_simulator(backend, config)
    sim.prepare()
    result = sim.simulate(mask)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, result.aerial.detach().cpu().numpy())
    typer.echo(f"backend={result.backend} aerial={tuple(result.aerial.shape)} -> {out_path}")


@simulate_app.command("list-backends")
def list_backends_cmd(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Also print the backing simulator class."
    ),
) -> None:
    """Print registered simulator backends.

    Without ``--verbose`` this prints one name per line (script-friendly).
    With ``--verbose`` it adds the implementing class so users can locate
    the source without reading ``simulators/registry.py`` directly.
    """

    names = list_simulators()
    if not verbose:
        for name in names:
            typer.echo(name)
        return
    width = max((len(n) for n in names), default=0)
    for name, cls in describe_simulators():
        typer.echo(f"{name.ljust(width)}  {cls.__module__}.{cls.__qualname__}")


def _load_mask(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        from PIL import Image

        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.astype(np.float32))
