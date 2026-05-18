# Benchmarks

This page tracks the canonical baseline numbers for OpenLithoHub's bundled
ILT models, plus the differentiable forward models that drive them.

## Headline baseline (synthetic-8)

Numbers are produced by

```bash
python scripts/generate_baselines.py --synthetic --limit 8 --output baselines/
```

against eight hand-rolled 64×64 layouts (square, h-line, line/space, T,
L, cross, contacts, dense lines). The synthetic suite is dataset-free and
runs in seconds, which is why it is the published reference. Real-dataset
numbers can be regenerated locally with `--data-root <path>`.

| Model | Samples | EPE mean (nm) | EPE max (nm) | PVB mean (nm) | MRC pass |
|---|---|---|---|---|---|
| `dummy-identity` | 8 | 0.000 | 0.000 | 2.140 | 0% |
| `rule-based-opc` | 8 | 0.530 | 1.414 | 2.487 | 0% |
| `levelset-ilt` (Gaussian PSF) | 8 | 0.036 | 0.250 | 2.128 | 0% |
| `neural-ilt` (untrained U-Net) | 8 | 15.074 | 24.637 | 2.497 | 100% |

Things worth knowing about these numbers:

- **`dummy-identity`** copies the design straight through. Its EPE is zero
  by construction on a synthetic suite where `design == target_mask`. It
  exists as a smoke test of the metric pipeline, not as a real model.
- **`rule-based-opc`** applies analytic per-edge bias OPC. It is the
  cheapest non-trivial baseline and a fair starting point when comparing
  AI methods.
- **`levelset-ilt`** runs 200 iterations of gradient-descent ILT under the
  default Gaussian PSF forward model (`sigma_px=2.0`). The MRC failure
  count reflects the small synthetic patterns being narrower than the
  default `min_width_nm=40`; this is expected on a 64-pixel-wide canvas
  and not an indictment of the optimizer. Run with a real LithoBench
  layout or relaxed MRC thresholds for production-grade numbers.
- **`neural-ilt`** uses a randomly-initialized U-Net unless you load
  pretrained weights via the model hub. The EPE shown here is the
  honest "no-weights" baseline. It will improve substantially after
  training; we report it as-is rather than hide it.

## Reproducing on real data

Once you have LithoBench cached locally:

```bash
python scripts/generate_baselines.py \
  --data-root /path/to/lithobench \
  --limit 16 \
  --pixel-nm 1.0 \
  --output baselines/lithobench/
```

The same `results.json` / `results.md` artifacts land under the chosen
output directory. Submit them to the public leaderboard with
`openlithohub leaderboard submit --file <results.json>`.

## Differentiable forward models

Two forward models ship in `openlithohub._utils`. Both are pure PyTorch
and auto-differentiable, so they slot directly into ILT optimization
loops, AI-OPC training, or any downstream gradient-based pipeline.

### Gaussian PSF (default)

`simulate_aerial_image(mask, sigma_px, dose=1.0)` — a single Gaussian
point spread function convolved with the mask. Fast, faithful enough for
unit tests and small synthetic patterns, and used as the default in
`LevelSetILTModel`.

### Hopkins partial-coherent imaging (SOCS)

`simulate_aerial_image_hopkins(mask, params)` implements the Sum-of-Coherent-
Systems decomposition of the Hopkins transmission cross coefficient. It
captures partial coherence, off-axis illumination, and defocus, which
the Gaussian model cannot.

Configurable via `HopkinsParams`:

| Field | Default | Meaning |
|---|---|---|
| `wavelength_nm` | 193.0 | Exposure wavelength (193 nm = ArF, 13.5 nm = EUV) |
| `na` | 1.35 | Numerical aperture (image-side) |
| `sigma` | 0.7 | Partial-coherence factor; outer sigma for annular/dipole/quasar |
| `sigma_inner` | 0.0 | Inner sigma for off-axis illumination |
| `pixel_size_nm` | 1.0 | Physical mask pixel size |
| `num_kernels` | 24 | SOCS truncation order |
| `illumination` | `"circular"` | One of `circular`, `annular`, `dipole`, `quasar` |
| `dipole_angle_deg` | 0.0 | Pole-pair orientation for dipole/quasar |
| `defocus_nm` | 0.0 | Defocus offset (parabolic phase) |

Switch `LevelSetILTModel` to Hopkins:

```python
from openlithohub._utils import HopkinsParams
from openlithohub.models.levelset_ilt import LevelSetILTModel

model = LevelSetILTModel(
    iterations=200,
    forward_model="hopkins",
    hopkins_params=HopkinsParams(
        wavelength_nm=193.0,
        na=1.35,
        sigma=0.7,
        num_kernels=24,
        pixel_size_nm=2.0,
    ),
)
result = model.predict(design)  # standard PredictionResult
```

The kernels are computed once and cached per `(params, grid_size, device)`,
so iterative ILT loops pay the SVD cost a single time.
