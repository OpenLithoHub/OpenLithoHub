"""Tests for the ORFS artifact adapter and tile cutter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

# klayout is in [workflow]; CI runs with [dev] only.
pytest.importorskip("klayout.db")

from openlithohub.data import LithoSample, OrfsArtifactDataset  # noqa: E402
from openlithohub.data.orfs import (  # noqa: E402
    DEFAULT_DESIGN_LAYER,
    DEFAULT_TILE_NM,
    tile_design_tensor,
)


def _build_synthetic_orfs_gds(target: Path, *, cell_name: str = "mock_alu") -> Path:
    """Write a synthetic ORFS-style routed GDS — one big top cell with M1 stripes.

    Mimics a small routed block: ~5 µm × 5 µm with horizontal M1 wires every
    140 nm (a coarse routing pitch). Enough geometry for tile-cutting tests
    to find non-empty tiles without needing the real ORFS toolchain.
    """
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.dbu = 0.00025  # 0.25 nm dbu, matches real ASAP7
    metal_layer = layout.layer(DEFAULT_DESIGN_LAYER[0], DEFAULT_DESIGN_LAYER[1])
    cell = layout.create_cell(cell_name)

    # 5 µm = 20000 dbu; pitch 560 dbu (140 nm); wire width 200 dbu (50 nm).
    extent_dbu = 20000
    pitch_dbu = 560
    width_dbu = 200
    for y0 in range(0, extent_dbu, pitch_dbu):
        cell.shapes(metal_layer).insert(kdb.Box(0, y0, extent_dbu, y0 + width_dbu))

    target.parent.mkdir(parents=True, exist_ok=True)
    layout.write(str(target))
    return target


@pytest.fixture
def orfs_gds(tmp_path) -> Path:
    return _build_synthetic_orfs_gds(tmp_path / "mock_alu.gds")


class TestTileDesignTensor:
    def test_basic_grid(self):
        # 100×100 array, 25-pixel tiles, non-overlapping → 16 tiles if all non-empty.
        arr = np.ones((100, 100), dtype=np.float32)
        tiles = tile_design_tensor(arr, tile_nm=25.0, pixel_nm=1.0)
        assert len(tiles) == 16
        for tile, _ in tiles:
            assert tile.shape == (25, 25)

    def test_drop_empty_default(self):
        # Half-zero array: tiles in the zero region should be dropped.
        arr = np.zeros((100, 100), dtype=np.float32)
        arr[:50, :] = 1.0
        tiles = tile_design_tensor(arr, tile_nm=25.0, pixel_nm=1.0)
        # Top half (y in [0, 50)) has 2 rows × 4 cols = 8 non-empty tiles.
        assert len(tiles) == 8
        for _tile, (_, y) in tiles:
            assert y < 50  # all kept tiles are in the populated half

    def test_keep_empty_when_disabled(self):
        arr = np.zeros((50, 50), dtype=np.float32)
        tiles = tile_design_tensor(arr, tile_nm=25.0, pixel_nm=1.0, drop_empty=False)
        assert len(tiles) == 4
        assert all(not tile.any() for tile, _ in tiles)

    def test_stride_overrides_default(self):
        arr = np.ones((50, 50), dtype=np.float32)
        # 25-pixel tile with 10-pixel stride → ceil((50-25)/10)+1 = 3 positions per axis.
        tiles = tile_design_tensor(arr, tile_nm=25.0, pixel_nm=1.0, stride_nm=10.0)
        assert len(tiles) == 9

    def test_ragged_edge_dropped(self):
        # 30×30 array, 25×25 tile, non-overlapping: only the (0,0) tile fits.
        arr = np.ones((30, 30), dtype=np.float32)
        tiles = tile_design_tensor(arr, tile_nm=25.0, pixel_nm=1.0)
        assert len(tiles) == 1
        _, coord = tiles[0]
        assert coord == (0, 0)

    def test_pixel_nm_scaling(self):
        # tile_nm=50, pixel_nm=2 → 25-pixel tiles. 100×100 array → 16 tiles.
        arr = np.ones((100, 100), dtype=np.float32)
        tiles = tile_design_tensor(arr, tile_nm=50.0, pixel_nm=2.0)
        assert len(tiles) == 16
        assert tiles[0][0].shape == (25, 25)


class TestOrfsArtifactDataset:
    def test_default_tile_size_yields_multiple_samples(self, orfs_gds):
        # 5 µm block, 2 µm tiles, non-overlapping → 2x2 = 4 tiles.
        ds = OrfsArtifactDataset(gds_path=orfs_gds, pixel_nm=1.0)
        # Synthetic block has wires across full extent so all 4 tiles are non-empty.
        assert len(ds) == 4

    def test_returns_lithosample(self, orfs_gds):
        ds = OrfsArtifactDataset(gds_path=orfs_gds, pixel_nm=1.0)
        sample = ds[0]
        assert isinstance(sample, LithoSample)
        assert isinstance(sample.design, torch.Tensor)
        assert sample.design.dtype == torch.float32
        assert sample.design.ndim == 2
        # 2 µm at 1 nm/pixel = 2000 pixels per side.
        assert sample.design.shape == (2000, 2000)
        assert sample.design.sum().item() > 0
        assert sample.mask is None

    def test_metadata_fields(self, orfs_gds):
        ds = OrfsArtifactDataset(
            gds_path=orfs_gds,
            pixel_nm=1.0,
            tile_nm=2000.0,
            design_name="mock_alu",
            orfs_revision="abc123",
        )
        sample = ds[0]
        md = sample.metadata
        assert md["dataset"] == "orfs"
        assert md["pdk"] == "asap7"
        assert md["design_name"] == "mock_alu"
        assert md["cell_name"] == "mock_alu"
        assert md["pixel_nm"] == 1.0
        assert md["tile_nm"] == 2000.0
        assert md["tile_index"] == 0
        assert md["orfs_revision"] == "abc123"
        assert md["design_layer"] == list(DEFAULT_DESIGN_LAYER)
        assert "license" in md

    def test_no_tiling_returns_one_sample(self, orfs_gds):
        ds = OrfsArtifactDataset(gds_path=orfs_gds, pixel_nm=1.0, tile_nm=None)
        assert len(ds) == 1
        # Whole-block sample is roughly 5 µm × 5 µm at 1 nm pixels;
        # exact size depends on the synthetic wire layout's bbox.
        h, w = ds[0].design.shape
        assert 4500 <= h <= 5100
        assert 4500 <= w <= 5100

    def test_5um_tile_yields_one_sample(self, orfs_gds):
        # Block is ~5 µm; if it falls slightly short of a full 5 µm extent,
        # the strict tile cutter drops the only candidate. 4500 nm is small
        # enough to definitely fit one full tile.
        ds = OrfsArtifactDataset(gds_path=orfs_gds, pixel_nm=1.0, tile_nm=4500.0)
        assert len(ds) == 1
        assert ds[0].design.shape == (4500, 4500)

    def test_tile_origin_offsets(self, orfs_gds):
        # 4 tiles at 2 µm each: lower-left corners at (0,0), (2000,0), (0,2000), (2000,2000) nm.
        ds = OrfsArtifactDataset(gds_path=orfs_gds, pixel_nm=1.0)
        origins = sorted(tuple(ds[i].metadata["tile_origin_nm"]) for i in range(len(ds)))
        assert origins == sorted([(0, 0), (2000, 0), (0, 2000), (2000, 2000)])

    def test_index_out_of_range(self, orfs_gds):
        ds = OrfsArtifactDataset(gds_path=orfs_gds, pixel_nm=1.0)
        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_explicit_cell_name(self, orfs_gds):
        ds = OrfsArtifactDataset(gds_path=orfs_gds, pixel_nm=1.0, cell_name="mock_alu")
        assert len(ds) == 4

    def test_riscv32i_mock_sram_design_name(self, tmp_path):
        # Issue #4 Phase 3 promises a SRAM-instantiated RISC-V test design.
        # ORFS already ships flow/designs/asap7/riscv32i-mock-sram/, and the
        # adapter is generic over design name. This test pins that contract
        # by feeding a synthetic GDS through with the SRAM design name and
        # asserting metadata round-trips cleanly. Cell names use underscores
        # because ORFS converts dashes when emitting top cells.
        gds = _build_synthetic_orfs_gds(
            tmp_path / "riscv32i_mock_sram.gds", cell_name="riscv32i_mock_sram"
        )
        ds = OrfsArtifactDataset(
            gds_path=gds,
            pixel_nm=1.0,
            tile_nm=2000.0,
            design_name="riscv32i-mock-sram",
        )
        assert len(ds) == 4
        md = ds[0].metadata
        assert md["design_name"] == "riscv32i-mock-sram"
        assert md["cell_name"] == "riscv32i_mock_sram"
        assert md["pdk"] == "asap7"

    def test_unknown_cell_raises(self, orfs_gds):
        ds = OrfsArtifactDataset(gds_path=orfs_gds, pixel_nm=1.0, cell_name="not_a_cell")
        with pytest.raises(KeyError, match="not found"):
            ds[0]

    def test_missing_gds_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ORFS GDS not found"):
            OrfsArtifactDataset(gds_path=tmp_path / "nope.gds")

    def test_invalid_tile_nm(self, orfs_gds):
        with pytest.raises(ValueError, match="tile_nm must be positive"):
            OrfsArtifactDataset(gds_path=orfs_gds, tile_nm=0)

    def test_iter(self, orfs_gds):
        ds = OrfsArtifactDataset(gds_path=orfs_gds, pixel_nm=1.0)
        samples = list(ds)
        assert len(samples) == 4
        assert all(isinstance(s, LithoSample) for s in samples)


class TestOrfsDownloadGuard:
    def test_download_always_rejects(self, orfs_gds, tmp_path):
        ds = OrfsArtifactDataset(gds_path=orfs_gds)
        with pytest.raises(RuntimeError, match="build-asap7-mock-alu"):
            ds.download(str(tmp_path / "anywhere"))


class TestDefaultTileNmConstant:
    def test_canonical_value(self):
        # 2 µm is the AI-OPC inference window the issue spec calls out.
        assert DEFAULT_TILE_NM == 2000.0
