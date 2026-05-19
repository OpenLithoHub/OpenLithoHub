"""Tests for the FreePDK45 + NanGate OCL PDK adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# klayout is in the [workflow] extra; CI runs with [dev] only, so skip
# the whole module when klayout is unavailable.
pytest.importorskip("klayout.db")

from openlithohub.data import FreePdk45Dataset, LithoSample  # noqa: E402
from openlithohub.data.freepdk45 import (  # noqa: E402
    CANONICAL_CELLS,
    DEFAULT_DESIGN_LAYER,
    FREEPDK45_LICENSE,
    FREEPDK45_LICENSE_URL,
    NANGATE_LICENSE_URL,
)


def _build_synthetic_freepdk45_gds(target: Path) -> Path:
    """Write a tiny GDS with the canonical NanGate cell names so the adapter can load.

    Each cell gets two horizontal rectangles on metal1 (11/0). Sized to mimic
    NanGate X1 cell footprints. The FreePDK45 dbu is 0.001 µm = 1 nm, but we
    use 0.0001 µm (0.1 nm) here to exercise the dbu-aware rasterizer.
    """
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.dbu = 0.0001  # 0.1 nm dbu (FreePDK45 is actually 1 nm; we test finer)
    metal_layer = layout.layer(DEFAULT_DESIGN_LAYER[0], DEFAULT_DESIGN_LAYER[1])

    # 1 dbu = 0.1 nm, so 1900 dbu = 190 nm. Mirrors NanGate X1 cell sizes.
    sizes_dbu = {
        "INV_X1": (1900, 14000),
        "NAND2_X1": (2850, 14000),
        "NOR2_X1": (2850, 14000),
        "DFF_X1": (9500, 14000),
    }
    for name, (w, h) in sizes_dbu.items():
        cell = layout.create_cell(name)
        cell.shapes(metal_layer).insert(kdb.Box(0, 0, w, 500))
        cell.shapes(metal_layer).insert(kdb.Box(0, h - 500, w, h))

    target.parent.mkdir(parents=True, exist_ok=True)
    layout.write(str(target))
    return target


@pytest.fixture
def freepdk45_root(tmp_path) -> Path:
    """Build a fake FreePDK45 tree with stdcells.gds at the root."""
    gds_path = tmp_path / "stdcells.gds"
    _build_synthetic_freepdk45_gds(gds_path)
    return tmp_path


class TestFreePdk45Dataset:
    def test_canonical_cells_constant(self):
        assert len(CANONICAL_CELLS) >= 4
        assert all(name.endswith("_X1") for name in CANONICAL_CELLS)

    def test_length_default_cells(self, freepdk45_root):
        ds = FreePdk45Dataset(root=freepdk45_root)
        assert len(ds) == len(CANONICAL_CELLS)

    def test_getitem_returns_lithosample(self, freepdk45_root):
        ds = FreePdk45Dataset(root=freepdk45_root)
        sample = ds[0]
        assert isinstance(sample, LithoSample)
        assert isinstance(sample.design, torch.Tensor)
        assert sample.design.dtype == torch.float32
        assert sample.design.ndim == 2
        assert sample.design.sum().item() > 0
        assert sample.mask is None
        assert sample.resist is None

    def test_metadata_fields(self, freepdk45_root):
        ds = FreePdk45Dataset(root=freepdk45_root, pixel_nm=1.0)
        sample = ds[0]
        md = sample.metadata
        assert md["dataset"] == "freepdk45"
        assert md["pdk"] == "freepdk45"
        assert md["pdk_variant"] == "nangate-openlib"
        assert md["cell_name"] == CANONICAL_CELLS[0]
        assert md["pixel_nm"] == 1.0
        assert md["design_layer"] == list(DEFAULT_DESIGN_LAYER)
        assert md["license"] == FREEPDK45_LICENSE
        assert md["license_url"] == FREEPDK45_LICENSE_URL
        assert md["secondary_license_url"] == NANGATE_LICENSE_URL

    def test_caching_returns_same_object(self, freepdk45_root):
        ds = FreePdk45Dataset(root=freepdk45_root)
        a = ds[0]
        b = ds[0]
        assert a is b

    def test_index_out_of_range(self, freepdk45_root):
        ds = FreePdk45Dataset(root=freepdk45_root)
        with pytest.raises(IndexError):
            ds[len(CANONICAL_CELLS)]

    def test_unknown_cell_raises_keyerror(self, freepdk45_root):
        ds = FreePdk45Dataset(root=freepdk45_root, cells=["NotARealCell_X1"])
        with pytest.raises(KeyError, match="not found"):
            ds[0]

    def test_custom_cell_subset(self, freepdk45_root):
        only = ["INV_X1", "NAND2_X1"]
        ds = FreePdk45Dataset(root=freepdk45_root, cells=only)
        assert len(ds) == 2
        assert ds[0].metadata["cell_name"] == only[0]
        assert ds[1].metadata["cell_name"] == only[1]

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="root not found"):
            FreePdk45Dataset(root=tmp_path / "does_not_exist")

    def test_missing_gds_raises(self, tmp_path):
        # Root exists but stdcells.gds was never cloned.
        with pytest.raises(FileNotFoundError, match="GDS not found"):
            FreePdk45Dataset(root=tmp_path)

    def test_explicit_gds_path(self, freepdk45_root):
        gds = freepdk45_root / "stdcells.gds"
        ds = FreePdk45Dataset(root=freepdk45_root, gds_path=gds)
        assert len(ds) == len(CANONICAL_CELLS)

    def test_iter(self, freepdk45_root):
        ds = FreePdk45Dataset(root=freepdk45_root, cells=["INV_X1"])
        samples = list(ds)
        assert len(samples) == 1
        assert samples[0].metadata["cell_name"] == "INV_X1"


class TestFreePdk45DownloadGate:
    def test_download_method_always_rejects(self, freepdk45_root):
        ds = FreePdk45Dataset(root=freepdk45_root)
        with pytest.raises(RuntimeError, match="FreePdk45Dataset.fetch"):
            ds.download(str(freepdk45_root / "elsewhere"))

    def test_fetch_without_accept_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="stacked license"):
            FreePdk45Dataset.fetch(tmp_path / "clone", accept_license=False)

    def test_fetch_message_mentions_both_license_urls(self, tmp_path):
        with pytest.raises(RuntimeError) as exc:
            FreePdk45Dataset.fetch(tmp_path / "clone")
        msg = str(exc.value)
        assert FREEPDK45_LICENSE_URL in msg
        assert NANGATE_LICENSE_URL in msg
