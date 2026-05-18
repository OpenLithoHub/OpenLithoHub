"""GAN-OPC training-data adapter.

GAN-OPC (Yang et al., TCAD 2020 — "GAN-OPC: Mask Optimization with
Lithography-guided Generative Adversarial Nets") ships its training set
as ~4875 paired binary PNGs at 2048×2048 resolution. The public mirror
is https://github.com/phdyang007/GAN-OPC, distributed as a 30-volume
7z archive (``ganopc-data.7z.001`` … ``.030``).

Once unpacked, the directory layout is::

    ganopc-data/
      artitgt/
        1.glp.png         # target design layout (binary)
        2.glp.png
        ...
        map.txt           # filename index (ignored by this loader)
      artimsk/
        1.glpOPC.png      # OPC-output mask paired with the target
        2.glpOPC.png
        ...

The two trees share sample IDs verbatim (``N.glp.png`` ↔
``N.glpOPC.png``). Pixel pitch is not stored alongside the data; the
loader defaults to 1.0 nm/px (configurable). Both PNGs are 8-bit
grayscale with strictly {0, 255} content, which the loader thresholds
into a {0., 1.} float32 tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from openlithohub.data.base import DatasetAdapter, LithoSample, natural_sort_key


class GanOpcDataset(DatasetAdapter):
    """Adapter for the GAN-OPC paired-PNG training set.

    Args:
        root: Either the directory containing ``artitgt/`` and
            ``artimsk/`` (typically ``ganopc-data/``), or the parent
            directory holding ``ganopc-data/`` — both are accepted.
        sample_ids: Optional explicit list of sample IDs to expose
            (e.g. ``["1", "2", "100"]``). Defaults to every ID present
            in both ``artitgt/`` and ``artimsk/``, sorted numerically
            when possible.
        pixel_nm: Raster pixel size in nm. Defaults to 1.0; this is the
            convention OpenLithoHub uses elsewhere and matches the
            ~2 µm patch sizes typical of GAN-OPC layouts. Override via
            constructor if your downstream pipeline assumes a different
            scale.
        threshold: Grayscale cutoff (0–255) above which a pixel is
            considered "on". Defaults to 127. The published PNGs are
            already strict binary, so the threshold only matters if a
            user supplies non-canonical files.
    """

    def __init__(
        self,
        root: str | Path,
        sample_ids: list[str] | None = None,
        pixel_nm: float = 1.0,
        threshold: int = 127,
    ) -> None:
        root = Path(root)
        if (root / "artitgt").is_dir() and (root / "artimsk").is_dir():
            data_root = root
        elif (root / "ganopc-data" / "artitgt").is_dir():
            data_root = root / "ganopc-data"
        else:
            raise FileNotFoundError(
                f"Could not find artitgt/ + artimsk/ under {root}. Pass "
                "either the ganopc-data directory itself or its parent."
            )
        self.root = data_root
        self.pixel_nm = float(pixel_nm)
        self.threshold = int(threshold)
        self._tgt_dir = data_root / "artitgt"
        self._msk_dir = data_root / "artimsk"

        if sample_ids is None:
            tgt_ids = {p.stem.removesuffix(".glp") for p in self._tgt_dir.glob("*.glp.png")}
            msk_ids = {p.stem.removesuffix(".glpOPC") for p in self._msk_dir.glob("*.glpOPC.png")}
            paired = sorted(tgt_ids & msk_ids, key=natural_sort_key)
            sample_ids = paired
        if not sample_ids:
            raise FileNotFoundError(
                f"No paired samples found under {data_root}/{{artitgt,artimsk}}/"
            )
        self._ids = list(sample_ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> LithoSample:
        if index < 0 or index >= len(self._ids):
            raise IndexError(f"Index {index} out of range [0, {len(self._ids)})")
        sample_id = self._ids[index]
        tgt_path = self._tgt_dir / f"{sample_id}.glp.png"
        msk_path = self._msk_dir / f"{sample_id}.glpOPC.png"
        if not tgt_path.exists():
            raise FileNotFoundError(f"Missing target PNG: {tgt_path}")
        if not msk_path.exists():
            raise FileNotFoundError(f"Missing mask PNG: {msk_path}")

        design_arr = self._load_png(tgt_path)
        mask_arr = self._load_png(msk_path)

        metadata: dict[str, Any] = {
            "dataset": "ganopc",
            "sample_id": sample_id,
            "source_target_png": str(tgt_path),
            "source_mask_png": str(msk_path),
            "pixel_nm": self.pixel_nm,
        }

        return LithoSample(
            design=torch.from_numpy(design_arr).float(),
            mask=torch.from_numpy(mask_arr).float(),
            resist=None,
            metadata=metadata,
        )

    def _load_png(self, path: Path) -> np.ndarray:
        # Imported lazily so importing the data package does not require
        # Pillow for users who only touch other adapters.
        from PIL import Image

        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.uint8)
        return (arr > self.threshold).astype(np.float32)

    def download(self, root: str) -> None:
        raise NotImplementedError(
            "GAN-OPC auto-download is not implemented. Clone the data "
            "repository manually:\n"
            "    git clone https://github.com/phdyang007/GAN-OPC <root>\n"
            "then unpack the multi-volume 7z archive (e.g. with py7zr + "
            "multivolumefile) so that <root>/ganopc-data/{artitgt,artimsk}/ "
            "exists, and pass <root> to GanOpcDataset()."
        )

    @property
    def sample_ids(self) -> list[str]:
        return list(self._ids)
