"""``openlithohub synth`` — generate synthetic PDK-aware layouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from openlithohub.synth import (
    PDK_PRESETS,
    PatternKind,
    generate_synthetic_batch,
)

synth_app = typer.Typer(help="Generate synthetic PDK-aware layouts.", no_args_is_help=True)


@synth_app.command()
def run(
    out_dir: Path = typer.Option(Path("synth_out"), "--out", "-o", help="Output directory."),
    pdk: str = typer.Option("freepdk45", "--pdk", help=f"PDK preset: {sorted(PDK_PRESETS)}."),
    pattern: PatternKind = typer.Option(
        PatternKind.RANDOM_LOGIC, "--pattern", "-p", help="Pattern type."
    ),
    n: int = typer.Option(10, "--n", "-n", help="Number of layouts."),
    size: int = typer.Option(256, "--size", "-s", help="Edge length in pixels."),
    seed: int = typer.Option(0, "--seed", help="PRNG seed."),
) -> None:
    """Generate ``n`` synthetic layouts and write them as ``.npy``."""

    batch = generate_synthetic_batch(pattern, n, pdk, size=size, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(batch.masks):
        path = out_dir / f"{batch.pattern.value}_{batch.pdk.name}_{batch.seeds[i]}.npy"
        np.save(path, mask.numpy())
    typer.echo(
        f"Wrote {n} {batch.pattern.value} layouts ({batch.pdk.name}, {size}x{size}) to {out_dir}"
    )


@synth_app.command("list-pdks")
def list_pdks() -> None:
    """Print registered PDK presets and key rules."""

    for name in sorted(PDK_PRESETS):
        rules = PDK_PRESETS[name]
        typer.echo(
            f"{name}: pixel={rules.pixel_size_nm}nm "
            f"min_width={rules.min_width_nm}nm "
            f"min_spacing={rules.min_spacing_nm}nm "
            f"pitch={rules.pitch_nm}nm"
        )
