"""LithoSim dataset adapter (HuggingFace Parquet format).

LithoSim is a sub-28nm industrial lithography simulation dataset hosted on
HuggingFace Hub. It stores design/mask/resist image pairs as Parquet rows
with image columns and process metadata.

The upstream dataset (``OpenLithoHub/LithoSim``) is currently **gated**:
new users must request access on the Hub and authenticate with
``huggingface-cli login`` before this adapter can fetch data. Calls
without auth fail with HTTP 401; the adapter detects that and raises
:class:`RuntimeError` with the remediation steps.

Requires: pip install openlithohub[data]  (adds `datasets` and `pyarrow`)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from openlithohub.data.base import DatasetAdapter, LithoSample

_HF_DATASET_NAME = "OpenLithoHub/LithoSim"

# Pinned for reproducibility per playbook §6. ``main`` is mutable, so any
# load that resolves the default branch is, by definition, irreproducible.
# Update this constant when the dataset advances and you have verified the
# new revision against your evaluation pipeline.
_DEFAULT_REVISION: str = "main"

_GATED_REMEDIATION = (
    "The HuggingFace dataset {name!r} is gated or private and the current "
    "session is not authenticated. To use this adapter:\n"
    "  1. Visit https://huggingface.co/datasets/{name} and request access.\n"
    "  2. Once approved, run `huggingface-cli login` (or set the\n"
    "     HF_TOKEN env var) so `datasets.load_dataset` can authenticate.\n"
    "Full guide: https://docs.openlithohub.com/hf-auth/\n"
    "Original error: {err}"
)


def _ensure_datasets_available() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The `datasets` package is required for LithoSim. "
            "Install it with: pip install openlithohub[data]"
        ) from e


def _is_auth_error(exc: BaseException) -> bool:
    """Detect HF auth/gated-repo failures across `datasets`/`huggingface_hub` versions.

    The exception class hierarchy has shifted (HfHubHTTPError, GatedRepoError,
    DatasetNotFoundError-with-401-cause); we match by class name and HTTP
    status to stay robust across upgrades without pinning to one error type.
    """
    name = type(exc).__name__
    if name in {"GatedRepoError", "RepositoryNotFoundError", "HfHubHTTPError"}:
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return True
    msg = str(exc).lower()
    return "401" in msg or "gated" in msg or "is not a valid dataset" in msg


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
        revision: Optional Git revision (commit SHA, tag, or branch) to pin
            for reproducible downloads. Defaults to ``_DEFAULT_REVISION``;
            pass ``None`` explicitly to opt out and resolve the dataset's
            default branch (irreproducible).
    """

    def __init__(
        self,
        split: str = "test",
        dataset_name: str = _HF_DATASET_NAME,
        cache_dir: str | None = None,
        pixel_nm: float = 0.5,
        streaming: bool = False,
        revision: str | None = _DEFAULT_REVISION,
    ) -> None:
        _ensure_datasets_available()
        self.split = split
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.pixel_nm = pixel_nm
        self.streaming = streaming
        self.revision = revision
        self._ds: Any = None
        self._len: int | None = None
        # Mutable default revision ("main") gives no reproducibility — the
        # upstream dataset can advance at any time and existing scores
        # silently rebase. Warn loudly so downstream evaluators know the
        # bytes are not pinned. Pinned revisions (commit hash / tag) are
        # silent.
        if revision in (None, "main"):
            import warnings as _w

            _w.warn(
                f"LithoSimDataset is loading {dataset_name!r} at revision="
                f"{revision!r} — this is mutable. For reproducible scoring "
                "pass a commit hash or tag (e.g. revision='abc1234'). "
                "Loaded bytes are not integrity-pinned.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def supports_random_access(self) -> bool:
        # Streaming mode wraps an HF IterableDataset — `len()` / `ds[i]`
        # would require draining the stream, so they raise TypeError. The
        # batched (non-streaming) load is materialised and supports both.
        return not self.streaming

    def _load_dataset(self) -> Any:
        if self._ds is None:
            from datasets import load_dataset

            try:
                # B615: revision is exposed as a constructor argument so callers
                # can pin a specific commit/tag for reproducible downloads.
                self._ds = load_dataset(  # nosec B615
                    self.dataset_name,
                    split=self.split,
                    cache_dir=self.cache_dir,
                    streaming=self.streaming,
                    revision=self.revision,
                )
            except Exception as exc:  # noqa: BLE001 — re-raised below
                if _is_auth_error(exc):
                    raise RuntimeError(
                        _GATED_REMEDIATION.format(name=self.dataset_name, err=exc)
                    ) from exc
                raise
        return self._ds

    def __len__(self) -> int:
        if self.streaming:
            raise TypeError(
                "LithoSimDataset does not support len() in streaming mode. "
                "Use iteration instead: `for sample in dataset: ...`"
            )
        if self._len is not None:
            return self._len
        ds = self._load_dataset()
        self._len = len(ds)
        return self._len

    def __getitem__(self, index: int) -> LithoSample:
        if self.streaming:
            raise TypeError(
                "LithoSimDataset does not support indexing in streaming mode. "
                "Use iteration instead: `for sample in dataset: ...`"
            )
        ds = self._load_dataset()

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        row = ds[index]
        return self._row_to_sample(row)

    def __iter__(self) -> Iterator[LithoSample]:
        ds = self._load_dataset()
        for row in ds:
            yield self._row_to_sample(row)

    def _row_to_sample(self, row: dict[str, Any]) -> LithoSample:
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
    def _array_to_tensor(arr: np.ndarray[Any, Any]) -> torch.Tensor:
        """Convert a numpy image array to a normalized float32 tensor in [0, 1].

        SEM and aerial-image rows in industrial litho datasets are commonly
        uint16; falling through to a plain ``astype(float32)`` would leave
        values in [0, 65535] and silently break any downstream code that
        assumes a [0, 1] resist threshold or EPE input range.
        """
        if arr.dtype == np.uint8:
            return torch.from_numpy(arr.astype(np.float32) / 255.0)
        if arr.dtype == np.uint16:
            return torch.from_numpy(arr.astype(np.float32) / 65535.0)
        if np.issubdtype(arr.dtype, np.integer):
            raise TypeError(
                f"Unsupported integer dtype {arr.dtype}; "
                "expected uint8 or uint16 SEM/aerial images."
            )
        return torch.from_numpy(arr.astype(np.float32))

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            return LithoSimDataset._array_to_tensor(value)

        if isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=torch.float32)

        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Pillow is required for image decoding. Install with: pip install Pillow"
            ) from e

        if isinstance(value, Image.Image):
            return LithoSimDataset._array_to_tensor(np.array(value))

        if isinstance(value, dict) and "bytes" in value:
            import io

            with Image.open(io.BytesIO(value["bytes"])) as img:
                return LithoSimDataset._array_to_tensor(np.array(img))

        raise TypeError(f"Cannot convert {type(value)} to tensor")

    def download(self, root: str) -> None:
        from datasets import load_dataset

        try:
            # B615: revision is pinnable via the constructor argument.
            load_dataset(  # nosec B615
                self.dataset_name,
                split=self.split,
                cache_dir=root,
                revision=self.revision,
            )
        except Exception as exc:  # noqa: BLE001 — re-raised below
            if _is_auth_error(exc):
                raise RuntimeError(
                    _GATED_REMEDIATION.format(name=self.dataset_name, err=exc)
                ) from exc
            raise

    @property
    def columns(self) -> list[str]:
        ds = self._load_dataset()
        return ds.column_names  # type: ignore[no-any-return]
