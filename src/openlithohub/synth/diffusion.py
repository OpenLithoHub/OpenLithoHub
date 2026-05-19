"""Diffusion-based synthetic layout generator (stub).

A full diffusion path — train a denoising U-Net on rasterised PDK
layouts, sample with classifier-free guidance, post-process with the
rule-based DRC pass — is on the roadmap but is not part of v0.1. It
needs (a) a curated training set, (b) GPU budget, and (c) an evaluation
protocol that proves the samples are *useful* for downstream OPC/ILT
training, not merely visually plausible.

The detailed plan is in
``docs/rfcs/0001-base-model.md`` (the layout-MAE base model doubles as
the generator initialisation) and
``docs/rfcs/0002-layout-tokens.md`` (tokenisation lets a transformer
generate sequences instead of rasters).

This stub exists so that:

1. Downstream code can target ``DiffusionLayoutGenerator`` symbolically.
2. The expected interface is locked in: ``__init__(pdk, weights_path)``
   then ``sample(n, size, seed) -> SyntheticBatch``.

A user with weights of their own can subclass this and override
:meth:`sample`.
"""

from __future__ import annotations

from pathlib import Path

from openlithohub.synth.pdk import PdkRules, get_pdk
from openlithohub.synth.rule_based import PatternKind, SyntheticBatch


class DiffusionLayoutGenerator:
    """Stub for a diffusion-model-backed synthetic-layout sampler."""

    def __init__(
        self,
        pdk: PdkRules | str = "freepdk45",
        weights_path: Path | str | None = None,
    ) -> None:
        self.pdk = pdk if isinstance(pdk, PdkRules) else get_pdk(pdk)
        self.weights_path = Path(weights_path) if weights_path is not None else None

    def sample(
        self,
        n: int,
        size: int = 256,
        seed: int = 0,
        pattern: PatternKind | str = PatternKind.RANDOM_LOGIC,
    ) -> SyntheticBatch:
        del n, size, seed, pattern
        raise NotImplementedError(
            "DiffusionLayoutGenerator is a v0.1 stub. The diffusion path "
            "requires training a denoising U-Net on rasterised PDK "
            "layouts; see docs/rfcs/0001-base-model.md and "
            "docs/rfcs/0002-layout-tokens.md. Use "
            "openlithohub.synth.generate_synthetic_batch for the "
            "rule-based path that ships in v0.1."
        )
