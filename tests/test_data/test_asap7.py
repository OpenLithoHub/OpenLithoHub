"""Tests for the ASAP7 PDK adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from openlithohub.data import Asap7Dataset, LithoSample
from openlithohub.data.asap7 import (
    ASAP7_LICENSE,
    ASAP7_LICENSE_URL,
    CANONICAL_CELLS,
    DEFAULT_DESIGN_LAYER,
    rasterize_cell_layer,
)


def _build_synthetic_asap7_gds(target: Path) -> Path:
    """Write a tiny GDS with the canonical cell names so the adapter can load.

    Each cell gets a single rectangle on the default design layer (10/0).
    Sized loosely to mimic ASAP7's 7.5T cells (~210 nm × 314 nm) in 0.25 nm
    dbu. Only INV/NAND2/NOR2/DFFHQN names are populated.
    """
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.dbu = 0.00025  # 0.25 nm dbu, matches real ASAP7
    metal_layer = layout.layer(DEFAULT_DESIGN_LAYER[0], DEFAULT_DESIGN_LAYER[1])

    # 1 dbu = 0.25 nm, so 800 dbu = 200 nm. Sized to mirror real cell footprints.
    sizes_dbu = {
        "INVx1_ASAP7_75t_R": (824, 1256),
        "NAND2x1_ASAP7_75t_R": (1452, 1256),
        "NOR2x1_ASAP7_75t_R": (1464, 1256),
        "DFFHQNx1_ASAP7_75t_R": (4464, 1256),
    }
    for name, (w, h) in sizes_dbu.items():
        cell = layout.create_cell(name)
        # Two horizontal rectangles to give the rasterizer something with
        # internal structure (mimics M1 routing tracks).
        cell.shapes(metal_layer).insert(kdb.Box(0, 0, w, 200))
        cell.shapes(metal_layer).insert(kdb.Box(0, h - 200, w, h))

    target.parent.mkdir(parents=True, exist_ok=True)
    layout.write(str(target))
    return target


@pytest.fixture
def asap7_root(tmp_path) -> Path:
    """Build a fake ASAP7 tree with the canonical cells under the right path."""
    gds_path = tmp_path / "asap7sc7p5t_27" / "GDS" / "asap7sc7p5t_27_R_999999.gds"
    _build_synthetic_asap7_gds(gds_path)
    return tmp_path


class TestAsap7Dataset:
    def test_canonical_cells_constant(self):
        # Sanity: the public list is non-empty and the entries look like cell names.
        assert len(CANONICAL_CELLS) >= 4
        assert all(name.endswith("_ASAP7_75t_R") for name in CANONICAL_CELLS)

    def test_length_default_cells(self, asap7_root):
        ds = Asap7Dataset(root=asap7_root)
        assert len(ds) == len(CANONICAL_CELLS)

    def test_getitem_returns_lithosample(self, asap7_root):
        ds = Asap7Dataset(root=asap7_root)
        sample = ds[0]
        assert isinstance(sample, LithoSample)
        assert isinstance(sample.design, torch.Tensor)
        assert sample.design.dtype == torch.float32
        assert sample.design.ndim == 2
        # The synthetic INVx1 has two horizontal stripes on M1, so the
        # rasterized design must contain *some* foreground pixels.
        assert sample.design.sum().item() > 0
        assert sample.mask is None
        assert sample.resist is None

    def test_metadata_fields(self, asap7_root):
        ds = Asap7Dataset(root=asap7_root, pixel_nm=1.0)
        sample = ds[0]
        md = sample.metadata
        assert md["dataset"] == "asap7"
        assert md["pdk"] == "asap7"
        assert md["cell_name"] == CANONICAL_CELLS[0]
        assert md["pixel_nm"] == 1.0
        assert md["design_layer"] == list(DEFAULT_DESIGN_LAYER)
        assert md["license"] == ASAP7_LICENSE
        assert md["license_url"] == ASAP7_LICENSE_URL

    def test_caching_returns_same_object(self, asap7_root):
        ds = Asap7Dataset(root=asap7_root)
        a = ds[0]
        b = ds[0]
        assert a is b

    def test_index_out_of_range(self, asap7_root):
        ds = Asap7Dataset(root=asap7_root)
        with pytest.raises(IndexError):
            ds[len(CANONICAL_CELLS)]

    def test_unknown_cell_raises_keyerror(self, asap7_root):
        ds = Asap7Dataset(root=asap7_root, cells=["NotARealCell_ASAP7_75t_R"])
        with pytest.raises(KeyError, match="not found"):
            ds[0]

    def test_custom_cell_subset(self, asap7_root):
        only = ["INVx1_ASAP7_75t_R", "NAND2x1_ASAP7_75t_R"]
        ds = Asap7Dataset(root=asap7_root, cells=only)
        assert len(ds) == 2
        assert ds[0].metadata["cell_name"] == only[0]
        assert ds[1].metadata["cell_name"] == only[1]

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="root not found"):
            Asap7Dataset(root=tmp_path / "does_not_exist")

    def test_missing_gds_raises(self, tmp_path):
        # Root exists but submodule was never initialised.
        (tmp_path / "asap7sc7p5t_27").mkdir()
        with pytest.raises(FileNotFoundError, match="No GDS matching"):
            Asap7Dataset(root=tmp_path)

    def test_explicit_gds_path(self, asap7_root):
        gds = next((asap7_root / "asap7sc7p5t_27" / "GDS").glob("*.gds"))
        ds = Asap7Dataset(root=asap7_root, gds_path=gds)
        assert len(ds) == len(CANONICAL_CELLS)

    def test_iter(self, asap7_root):
        ds = Asap7Dataset(root=asap7_root, cells=["INVx1_ASAP7_75t_R"])
        samples = list(ds)
        assert len(samples) == 1
        assert samples[0].metadata["cell_name"] == "INVx1_ASAP7_75t_R"


class TestAsap7DownloadGate:
    def test_download_method_always_rejects(self, tmp_path):
        # The instance method exists only to satisfy the abstract contract;
        # callers must use Asap7Dataset.fetch() instead.
        gds = tmp_path / "asap7sc7p5t_27" / "GDS" / "asap7sc7p5t_27_R_x.gds"
        _build_synthetic_asap7_gds(gds)
        ds = Asap7Dataset(root=tmp_path)
        with pytest.raises(RuntimeError, match="Asap7Dataset.fetch"):
            ds.download(str(tmp_path / "elsewhere"))

    def test_fetch_without_accept_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match=ASAP7_LICENSE):
            Asap7Dataset.fetch(tmp_path / "clone", accept_license=False)

    def test_fetch_message_mentions_license_url(self, tmp_path):
        with pytest.raises(RuntimeError) as exc:
            Asap7Dataset.fetch(tmp_path / "clone")
        assert ASAP7_LICENSE_URL in str(exc.value)


class TestRasterizeHelper:
    def test_rasterize_empty_layer_returns_zeros(self, asap7_root):
        import klayout.db as kdb

        gds = next((asap7_root / "asap7sc7p5t_27" / "GDS").glob("*.gds"))
        layout = kdb.Layout()
        layout.read(str(gds))
        cell = layout.cell("INVx1_ASAP7_75t_R")
        # Layer 999/0 has no shapes — must return zeros, not crash.
        arr, origin = rasterize_cell_layer(layout, cell, (999, 0), pixel_nm=1.0)
        assert arr.sum() == 0.0
        assert arr.ndim == 2

    def test_rasterize_design_layer_has_foreground(self, asap7_root):
        import klayout.db as kdb

        gds = next((asap7_root / "asap7sc7p5t_27" / "GDS").glob("*.gds"))
        layout = kdb.Layout()
        layout.read(str(gds))
        cell = layout.cell("INVx1_ASAP7_75t_R")
        arr, _ = rasterize_cell_layer(layout, cell, DEFAULT_DESIGN_LAYER, pixel_nm=1.0)
        assert arr.sum() > 0
