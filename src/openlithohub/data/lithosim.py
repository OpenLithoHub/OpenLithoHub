"""LithoSim dataset adapter (HuggingFace Parquet format).

LithoSim is a sub-28nm industrial lithography simulation dataset hosted on
HuggingFace Hub. It stores design/mask/resist image pairs as Parquet rows
with image columns and process metadata.

Requires: pip install openlithohub[data]  (adds `datasets` and `pyarrow`)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from openlithohub.data.base import DatasetAdapter, LithoSample

_HF_DATASET_NAME = "OpenLithoHub/LithoSim"


def _ensure_datasets_available() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The `datasets` package is required for LithoSim. "
            "Install it with: pip install openlithohub[data]"
        ) from e


class LithoSimDataset(DatasetAdapter):
    """Adapter for the LithoSim dataset (sub-28nm industrial benchmark).

    Loads data from HuggingFace Hub using the `datasets` library.
    Images are stored as columns in Parquet format and decoded to tensors on access.

    Args:
        split: Dataset split ('train', 'test', or 'all').
        dataset_name: HuggingFace dataset identifier. Override for custom forks.
        cache_dir: Local cache directory for downloaded data.
        pixel_nm: Pixel resolution in nanometers.
        streaming: If True, use streaming mode (no full download).
    """

    def __init__(
        self,
        split: str = "test",
        dataset_name: str = _HF_DATASET_NAME,
        cache_dir: str | None = None,
        pixel_nm: float = 0.5,
        streaming: bool = False,
    ) -> None:
        _ensure_datasets_available()
        self.split = split
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.pixel_nm = pixel_nm
        self.streaming = streaming
        self._ds: Any = None
        self._len: int | None = None

    def _load_dataset(self) -> Any:
        if self._ds is None:
            from datasets import load_dataset

            self._ds = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                streaming=self.streaming,
            )
        return self._ds

    def __len__(self) -> int:
        if self._len is not None:
            return self._len
        ds = self._load_dataset()
        self._len = len(ds)
        return self._len

    def __getitem__(self, index: int) -> LithoSample:
        ds = self._load_dataset()

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        row = ds[index]
        design = self._decode_image(row, "design")
        mask = self._try_decode_image(row, "mask")
        resist = self._try_decode_image(row, "resist")

        metadata: dict[str, Any] = {
            "dataset": "lithosim",
            "pixel_nm": self.pixel_nm,
            "split": self.split,
        }
        for key in ("process_node", "pitch_nm", "dose", "focus", "sample_id", "feature_type"):
            if key in row:
                metadata[key] = row[key]

        return LithoSample(
            design=design,
            mask=mask,
            resist=resist,
            metadata=metadata,
        )

    def _decode_image(self, row: dict[str, Any], column: str) -> torch.Tensor:
        if column not in row:
            raise KeyError(f"Required column '{column}' not found in dataset row")
        return self._to_tensor(row[column])

    def _try_decode_image(self, row: dict[str, Any], column: str) -> torch.Tensor | None:
        if column not in row or row[column] is None:
            return None
        return self._to_tensor(row[column])

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            arr = value.astype(np.float32)
            if arr.size > 0 and arr.max() > 1.0:
                arr = arr / 255.0
            return torch.from_numpy(arr)

        if isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=torch.float32)

        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Pillow is required for image decoding. Install with: pip install Pillow"
            ) from e

        if isinstance(value, Image.Image):
            arr = np.array(value, dtype=np.float32)
            if arr.size > 0 and arr.max() > 1.0:
                arr = arr / 255.0
            return torch.from_numpy(arr)

        if isinstance(value, dict) and "bytes" in value:
            import io

            img = Image.open(io.BytesIO(value["bytes"]))
            arr = np.array(img, dtype=np.float32)
            if arr.size > 0 and arr.max() > 1.0:
                arr = arr / 255.0
            return torch.from_numpy(arr)

        raise TypeError(f"Cannot convert {type(value)} to tensor")

    def download(self, root: str) -> None:
        from datasets import load_dataset

        load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=root,
        )

    @property
    def columns(self) -> list[str]:
        ds = self._load_dataset()
        return ds.column_names
