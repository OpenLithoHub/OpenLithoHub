"""LithoBench dataset adapter (.npy format)."""

from __future__ import annotations

from pathlib import Path

from openlithohub.data.base import DatasetAdapter, LithoSample


class LithoBenchDataset(DatasetAdapter):
    """Adapter for the LithoBench dataset (NeurIPS'23, 45nm baseline).

    Expected format: directory of .npy files with design/mask/resist arrays.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __len__(self) -> int:
        raise NotImplementedError(
            "LithoBench loader not yet implemented. "
            "Planned: scan root directory for .npy files, "
            "index by sample ID. Reference: LithoBench GitHub repo."
        )

    def __getitem__(self, index: int) -> LithoSample:
        raise NotImplementedError(
            "LithoBench sample loading not yet implemented. "
            "Planned: np.load() design/mask/resist .npy files, "
            "convert to torch.Tensor, populate metadata from filename convention."
        )

    def download(self, root: str) -> None:
        raise NotImplementedError(
            "LithoBench auto-download not yet implemented. "
            "Manual download available at the LithoBench GitHub repository."
        )
