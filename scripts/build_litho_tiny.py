"""Build the OpenLithoHub/litho-tiny dataset (100 mask/aerial pairs).

Why this exists
---------------
P0-4 of the strategic plan: ship a placeholder dataset on HuggingFace so
that ``load_dataset("OpenLithoHub/litho-tiny")`` resolves and so research
teams have a concrete schema to extend. The full Litho-1M predecessor is
explicitly deferred — generating 100 deterministic samples is enough to
prove the pipeline.

What it produces
----------------
A directory with:

* ``data/train-00000-of-00001.parquet`` — 100 rows. Columns:
    - ``id``                : str, e.g. ``"sram-000"``.
    - ``pattern``           : str, one of ``sram | contact_array | random_logic``.
    - ``pdk``               : str, e.g. ``"freepdk45"``.
    - ``size_px``           : int (mask edge length).
    - ``mask``              : binary uint8 PNG bytes, ``(H, W)`` in ``{0, 255}``.
    - ``aerial``            : float32 PNG-like binary (8-bit normalized) bytes,
                              ``(H, W)`` aerial intensity ∈ [0, ~1].
    - ``mask_npy``          : raw float32 ``(H, W)`` little-endian bytes.
    - ``aerial_npy``        : raw float32 ``(H, W)`` little-endian bytes.
* ``README.md``  — HF dataset card.

How to use
----------

::

    python scripts/build_litho_tiny.py --out out/litho-tiny

To upload (after ``hf auth login``)::

    hf upload OpenLithoHub/litho-tiny out/litho-tiny --repo-type dataset

Or via Python::

    from huggingface_hub import HfApi
    HfApi().upload_folder(
        folder_path="out/litho-tiny",
        repo_id="OpenLithoHub/litho-tiny",
        repo_type="dataset",
    )
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import torch

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub.synth import generate_layout

PATTERNS = ("sram", "contact_array", "random_logic")
PDK = "freepdk45"
SIZE = 512  # SRAM cells need >=512 px to tile inside the canvas — see synth.rule_based.
PER_PATTERN = 34  # 34 + 33 + 33 = 100
SIGMA_PX = 4.0
DOSE = 1.0


def _png_bytes(arr: np.ndarray, mode: str) -> bytes:
    """Encode a 2D ndarray to PNG bytes. ``mode`` is ``"L"`` (uint8 mask)
    or ``"L"`` for normalized aerial; both go through 8-bit grayscale."""
    from PIL import Image

    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_records(out_dir: Path) -> list[dict[str, object]]:
    counts = {"sram": PER_PATTERN, "contact_array": 33, "random_logic": 33}
    records: list[dict[str, object]] = []
    seed = 0
    for pattern in PATTERNS:
        for i in range(counts[pattern]):
            mask = generate_layout(pattern, PDK, size=SIZE, seed=seed)
            aerial = simulate_aerial_image(mask, sigma_px=SIGMA_PX, dose=DOSE)

            mask_np = mask.detach().cpu().numpy().astype(np.float32)
            aerial_np = aerial.detach().cpu().numpy().astype(np.float32)

            mask_u8 = (mask_np * 255.0).clip(0, 255).astype(np.uint8)
            # Normalize aerial to [0, 255] by clipping at 1.0 (open-frame
            # calibration in our forward model).
            aerial_u8 = (aerial_np.clip(0.0, 1.0) * 255.0).astype(np.uint8)

            records.append(
                {
                    "id": f"{pattern}-{i:03d}",
                    "pattern": pattern,
                    "pdk": PDK,
                    "size_px": SIZE,
                    "seed": seed,
                    "mask": _png_bytes(mask_u8, "L"),
                    "aerial": _png_bytes(aerial_u8, "L"),
                    "mask_npy": mask_np.tobytes(),
                    "aerial_npy": aerial_np.tobytes(),
                }
            )
            seed += 1
    return records


def _write_parquet(records: list[dict[str, object]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {k: [r[k] for r in records] for k in records[0]}
    table = pa.table(cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")


DATASET_CARD = """---
license: apache-2.0
task_categories:
- image-segmentation
- image-to-image
language:
- en
tags:
- lithography
- semiconductor
- EUV
- OPC
- mask-optimization
- inverse-lithography
size_categories:
- n<1K
pretty_name: Litho-Tiny-100
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Litho-Tiny-100

A tiny placeholder lithography dataset shipped by
[OpenLithoHub](https://github.com/OpenLithoHub/OpenLithoHub) — **100 deterministic
(mask, aerial-image) pairs** generated from the project's rule-based PDK-aware
synthesizer and Gaussian-PSF forward model.

> ⚠️ **This dataset is intentionally tiny.** It exists to nail down the schema
> and prove `load_dataset("OpenLithoHub/litho-tiny")` works end-to-end. If you
> need a real research-scale dataset, see the strategic plan's discussion of the
> upcoming `Litho-1M` pre-training set, or use one of the upstream sources
> referenced in OpenLithoHub's data layer.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("OpenLithoHub/litho-tiny", split="train")
print(ds)
print(ds.column_names)

# Decode the first mask
import io, numpy as np
from PIL import Image
mask = np.array(Image.open(io.BytesIO(ds[0]["mask"])))    # (512, 512) uint8 in {0, 255}
aerial = np.array(Image.open(io.BytesIO(ds[0]["aerial"])))  # (512, 512) uint8

# Or use the raw float32 bytes for full precision
mask_f32 = np.frombuffer(ds[0]["mask_npy"], dtype=np.float32).reshape(512, 512)
aerial_f32 = np.frombuffer(ds[0]["aerial_npy"], dtype=np.float32).reshape(512, 512)
```

## Schema

| Column        | Type         | Description                                                    |
|---------------|--------------|----------------------------------------------------------------|
| `id`          | string       | Stable identifier `<pattern>-<index>`, e.g. `sram-007`.        |
| `pattern`     | string       | One of `sram`, `contact_array`, `random_logic`.                |
| `pdk`         | string       | PDK preset (always `freepdk45` for this tiny version).         |
| `size_px`     | int32        | Mask edge length (512).                                        |
| `seed`        | int32        | PRNG seed used by the synthesizer.                             |
| `mask`        | bytes (PNG)  | Binary mask, 8-bit grayscale, `{0, 255}`.                      |
| `aerial`      | bytes (PNG)  | Aerial-image intensity normalized to `[0, 255]` (clipped at 1).|
| `mask_npy`    | bytes        | Raw `float32` mask, row-major `(H, W)`.                        |
| `aerial_npy`  | bytes        | Raw `float32` aerial intensity (un-clipped), row-major `(H, W)`. |

Counts: 34 SRAM + 33 contact-array + 33 random-logic = **100 rows**, all in `train`.

## Reproducing

```bash
git clone https://github.com/OpenLithoHub/OpenLithoHub.git
cd OpenLithoHub
pip install -e '.[data]'
python scripts/build_litho_tiny.py --out out/litho-tiny
```

The script is fully deterministic; identical commits produce identical bytes.

## Forward model

The aerial images come from `openlithohub._utils.forward_model.simulate_aerial_image`
— a Gaussian-PSF approximation (`sigma_px=4.0`, `dose=1.0`) of the Hopkins forward
model, with circular padding. For a research-grade SOCS Hopkins simulation, see
`openlithohub._utils.hopkins.simulate_aerial_image_hopkins`.

## License

Apache-2.0 — same as OpenLithoHub itself. Patterns are synthetic and free of any
real fab IP.

## Citation

If this dataset helped your work, please ⭐ the
[OpenLithoHub repo](https://github.com/OpenLithoHub/OpenLithoHub) and cite the project.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("out/litho-tiny"),
        help="Output directory (will be created).",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    print(f"Generating {sum([34, 33, 33])} layout/aerial pairs...")
    records = _build_records(out_dir)
    print(f"  ok — {len(records)} records.")

    parquet_path = out_dir / "data" / "train-00000-of-00001.parquet"
    print(f"Writing {parquet_path} ...")
    _write_parquet(records, parquet_path)

    readme_path = out_dir / "README.md"
    readme_path.write_text(DATASET_CARD, encoding="utf-8")
    print(f"Wrote {readme_path}")

    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"\nDataset built at {out_dir} ({total_bytes / 1024:.1f} KiB total).")
    print("Upload with:")
    print(f"  hf upload OpenLithoHub/litho-tiny {out_dir} --repo-type dataset")


if __name__ == "__main__":
    main()
