"""LithoSim dataset adapter (HuggingFace Parquet format)."""

from __future__ import annotations

from openlithohub.data.base import DatasetAdapter, LithoSample


class LithoSimDataset(DatasetAdapter):
    """Adapter for the LithoSim dataset (NeurIPS'25, sub-28nm industrial).

    Uses HuggingFace datasets library to load Parquet-formatted data.
    """

    def __init__(self, split: str = "test") -> None:
        self.split = split

    def __len__(self) -> int:
        raise NotImplementedError(
            "LithoSim loader not yet implemented. "
            "Planned: use `datasets.load_dataset('lithosim')` to get sample count. "
            "Requires: pip install openlithohub[data]"
        )

    def __getitem__(self, index: int) -> LithoSample:
        raise NotImplementedError(
            "LithoSim sample loading not yet implemented. "
            "Planned: load Parquet row, decode image columns to tensors, "
            "align resolution metadata with LithoSample schema."
        )

    def download(self, root: str) -> None:
        raise NotImplementedError(
            "LithoSim auto-download not yet implemented. "
            "Planned: use HuggingFace datasets `load_dataset` with cache_dir=root."
        )
