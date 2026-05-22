"""Tests for the FreePDK45 SRAM-cell adapter (OpenRAM-bundled GDS)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# klayout is in [workflow]; CI runs with [dev] only.
pytest.importorskip("klayout.db")

from openlithohub.data import FreePdk45SramDataset, LithoSample  # noqa: E402
from openlithohub.data.freepdk45_sram import (  # noqa: E402
    CANONICAL_CELLS,
    DEFAULT_DESIGN_LAYER,
    OPENRAM_LICENSE,
    OPENRAM_LICENSE_URL,
)


def _write_one_cell_gds(target: Path, cell_name: str, *, w_dbu: int, h_dbu: int) -> None:
    """Mimic OpenRAM's bundle convention: one GDS per cell, top-cell name == file stem.

    FreePDK45 uses 0.5 nm dbu in OpenRAM's bundle; we replicate that to keep
    the rasterizer code path on the same dbu scaling factor as production.
    """
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.dbu = 0.0005  # 0.5 nm
    metal_layer = layout.layer(DEFAULT_DESIGN_LAYER[0], DEFAULT_DESIGN_LAYER[1])
    cell = layout.create_cell(cell_name)
    # Two horizontal stripes — top and bottom rails — like a real cell.
    rail = max(1, h_dbu // 20)
    cell.shapes(metal_layer).insert(kdb.Box(0, 0, w_dbu, rail))
    cell.shapes(metal_layer).insert(kdb.Box(0, h_dbu - rail, w_dbu, h_dbu))
    target.parent.mkdir(parents=True, exist_ok=True)
    layout.write(str(target))


@pytest.fixture
def synthetic_gds_lib(tmp_path) -> Path:
    """Build a fake OpenRAM-style ``gds_lib/`` with all 10 canonical cells."""
    lib = tmp_path / "gds_lib"
    sizes = {
        "cell_1rw": (1800, 3000),
        "cell_2rw": (2400, 3000),
        "dff": (4000, 3000),
        "sense_amp": (3000, 3000),
        "write_driver": (3000, 3000),
        "tri_gate": (2000, 3000),
        "replica_cell_1rw": (1800, 3000),
        "replica_cell_2rw": (2400, 3000),
        "dummy_cell_1rw": (1800, 3000),
        "dummy_cell_2rw": (2400, 3000),
    }
    for name, (w, h) in sizes.items():
        _write_one_cell_gds(lib / f"{name}.gds", name, w_dbu=w, h_dbu=h)
    return lib


class TestFreePdk45SramDataset:
    def test_canonical_cells_constant(self):
        # The bundle has 10 cells; the bitcell must be first because it's
        # the headline data product.
        assert CANONICAL_CELLS[0] == "cell_1rw"
        assert "dff" in CANONICAL_CELLS
        assert "sense_amp" in CANONICAL_CELLS
        assert len(CANONICAL_CELLS) == 10

    def test_default_length(self, synthetic_gds_lib):
        ds = FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib)
        assert len(ds) == len(CANONICAL_CELLS)

    def test_returns_lithosample(self, synthetic_gds_lib):
        ds = FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib)
        sample = ds[0]
        assert isinstance(sample, LithoSample)
        assert isinstance(sample.design, torch.Tensor)
        assert sample.design.dtype == torch.float32
        assert sample.design.ndim == 2
        assert sample.design.sum().item() > 0
        assert sample.mask is None
        assert sample.resist is None

    def test_metadata_fields(self, synthetic_gds_lib):
        ds = FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib, pixel_nm=1.0)
        sample = ds[0]
        md = sample.metadata
        assert md["dataset"] == "freepdk45-sram"
        assert md["pdk"] == "freepdk45"
        assert md["pdk_variant"] == "openram-bundled"
        assert md["cell_name"] == "cell_1rw"
        assert md["pixel_nm"] == 1.0
        assert md["design_layer"] == list(DEFAULT_DESIGN_LAYER)
        assert md["dbu_nm"] == pytest.approx(0.5)
        assert md["tooling_license"] == OPENRAM_LICENSE
        assert md["tooling_license_url"] == OPENRAM_LICENSE_URL
        assert "license" in md
        assert "license_url" in md

    def test_caching_returns_same_object(self, synthetic_gds_lib):
        ds = FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib)
        a = ds[0]
        b = ds[0]
        assert a is b

    def test_custom_cell_subset(self, synthetic_gds_lib):
        only = ["cell_1rw", "sense_amp"]
        ds = FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib, cells=only)
        assert len(ds) == 2
        assert ds[0].metadata["cell_name"] == "cell_1rw"
        assert ds[1].metadata["cell_name"] == "sense_amp"

    def test_unknown_cell_raises_keyerror(self, synthetic_gds_lib):
        ds = FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib, cells=["not_a_real_cell"])
        with pytest.raises(KeyError, match="not found"):
            ds[0]

    def test_index_out_of_range(self, synthetic_gds_lib):
        ds = FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib)
        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_invalid_pixel_nm(self, synthetic_gds_lib):
        with pytest.raises(ValueError, match="pixel_nm must be positive"):
            FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib, pixel_nm=0)

    def test_missing_gds_lib_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="gds_lib not found"):
            FreePdk45SramDataset(gds_lib_path=tmp_path / "does_not_exist")

    def test_iter(self, synthetic_gds_lib):
        ds = FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib, cells=["cell_1rw", "dff"])
        samples = list(ds)
        assert len(samples) == 2
        assert [s.metadata["cell_name"] for s in samples] == ["cell_1rw", "dff"]

    def test_pixel_size_scales_image(self, synthetic_gds_lib):
        # cell_1rw is 1800 dbu × 3000 dbu @ 0.5 nm/dbu = 900 × 1500 nm.
        # At 1 nm/pixel → 1500 rows × 900 cols. At 5 nm/pixel → 300 × 180.
        ds_fine = FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib, pixel_nm=1.0)
        ds_coarse = FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib, pixel_nm=5.0)
        # cell_1rw is at index 0
        h_fine, w_fine = ds_fine[0].design.shape
        h_coarse, w_coarse = ds_coarse[0].design.shape
        assert h_fine > h_coarse and w_fine > w_coarse
        # Roughly 5× downscaling on each axis.
        assert h_fine / h_coarse == pytest.approx(5.0, rel=0.1)


class TestFreePdk45SramDownloadGuard:
    def test_download_always_rejects(self, synthetic_gds_lib, tmp_path):
        ds = FreePdk45SramDataset(gds_lib_path=synthetic_gds_lib)
        with pytest.raises(RuntimeError, match="pip install"):
            ds.download(str(tmp_path / "anywhere"))


class TestOpenramAutoLocate:
    """Exercise the importlib.resources path only when openram is installed."""

    def test_locate_or_skip(self):
        # If openram is not installed (the common CI case), we should get
        # a clean ImportError with installation instructions, NOT a
        # generic AttributeError or a silent zero-length dataset.
        try:
            import openram  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="openram"):
                FreePdk45SramDataset()
            return

        # When openram appears importable but its bundled gds_lib is
        # missing (e.g. a broken half-uninstall left a namespace package
        # behind), we should get a clear FileNotFoundError pointing at
        # the missing bundle directory — not a confusing downstream error.
        try:
            ds = FreePdk45SramDataset()
        except FileNotFoundError as exc:
            assert "gds_lib" in str(exc)
            return

        # Healthy install: bundle is present, dataset rooted on cell_1rw.
        assert len(ds) == len(CANONICAL_CELLS)
        assert ds[0].metadata["cell_name"] == "cell_1rw"
        assert ds[0].design.sum().item() > 0
