"""LithoBench dataset adapter (.npy format).

LithoBench (NeurIPS'23) organizes data as paired .npy arrays per sample:
    root/
      design/
        sample_0000.npy   # binary design layout (H, W)
        sample_0001.npy
        ...
      mask/
        sample_0000.npy   # optimized mask (H, W), may not exist for all samples
        ...
      resist/
        sample_0000.npy   # simulated resist contour (H, W), optional
        ...
      metadata.json       # optional: per-sample process parameters

Alternatively, a flat layout is supported:
    root/
      sample_0000_design.npy
      sample_0000_mask.npy
      sample_0000_resist.npy
      ...
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openlithohub.data.base import DatasetAdapter, LithoSample

_FILENAME_RE = re.compile(r"^(?P<sample_id>.+?)_(?P<kind>design|mask|resist)\.npy$")


class LithoBenchDataset(DatasetAdapter):
    """Adapter for the LithoBench dataset (NeurIPS'23, 45nm baseline).

    Supports two directory layouts:
    1. Subdirectory layout: root/{design,mask,resist}/sample_XXXX.npy
    2. Flat layout: root/sample_XXXX_{design,mask,resist}.npy

    Args:
        root: Path to the dataset directory.
        split: Optional split name (e.g. 'train', 'test'). If set, looks for root/split/.
        pixel_nm: Pixel resolution in nanometers (default 1.0 for LithoBench 45nm node).
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        pixel_nm: float = 1.0,
    ) -> None:
        self.root = Path(root)
        if split:
            self.root = self.root / split
        self.pixel_nm = pixel_nm
        self._index: list[str] = []
        self._layout: str = "unknown"
        self._metadata: dict[str, Any] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        design_dir = self.root / "design"
        if design_dir.is_dir():
            self._layout = "subdirectory"
            self._index = sorted(p.stem for p in design_dir.glob("*.npy"))
        else:
            self._layout = "flat"
            seen: set[str] = set()
            for p in self.root.glob("*.npy"):
                m = _FILENAME_RE.match(p.name)
                if m and m.group("kind") == "design":
                    seen.add(m.group("sample_id"))
            self._index = sorted(seen)

        meta_path = self.root / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> LithoSample:
        if index < 0 or index >= len(self._index):
            raise IndexError(f"Index {index} out of range [0, {len(self._index)})")

        sample_id = self._index[index]
        design = self._load_array(sample_id, "design")
        mask = self._try_load_array(sample_id, "mask")
        resist = self._try_load_array(sample_id, "resist")

        metadata: dict[str, Any] = {
            "dataset": "lithobench",
            "sample_id": sample_id,
            "pixel_nm": self.pixel_nm,
        }
        if sample_id in self._metadata:
            metadata.update(self._metadata[sample_id])

        return LithoSample(
            design=torch.from_numpy(design).float(),
            mask=torch.from_numpy(mask).float() if mask is not None else None,
            resist=torch.from_numpy(resist).float() if resist is not None else None,
            metadata=metadata,
        )

    def _resolve_path(self, sample_id: str, kind: str) -> Path:
        if self._layout == "subdirectory":
            return self.root / kind / f"{sample_id}.npy"
        return self.root / f"{sample_id}_{kind}.npy"

    def _load_array(self, sample_id: str, kind: str) -> np.ndarray:
        path = self._resolve_path(sample_id, kind)
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return np.load(path)

    def _try_load_array(self, sample_id: str, kind: str) -> np.ndarray | None:
        path = self._resolve_path(sample_id, kind)
        if path.exists():
            return np.load(path)
        return None

    def download(self, root: str) -> None:
        raise NotImplementedError(
            "LithoBench auto-download not yet implemented. "
            "Please download manually from: https://github.com/phdyang007/lithobench"
        )

    @property
    def sample_ids(self) -> list[str]:
        return list(self._index)
