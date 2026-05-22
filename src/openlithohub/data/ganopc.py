"""GAN-OPC training-data adapter.

GAN-OPC ships its training set as ~4875 paired binary PNGs at 2048×2048
resolution. The public mirror is https://github.com/phdyang007/GAN-OPC,
distributed as a 30-volume 7z archive (``ganopc-data.7z.001`` … ``.030``).
The :func:`download_ganopc` helper auto-fetches and unpacks it on first
use; until then the repo carries no upstream bytes (per
``DATA-LICENSES.md`` — redistribution is not granted).

Reference: Yang et al., *GAN-OPC: Mask Optimization with Lithography-guided
Generative Adversarial Nets*, DAC 2018 (open-access, arXiv:1810.04293).
A paywalled TCAD 2020 extension exists; the open DAC paper is the canonical
citation for this adapter.

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

import shutil
import subprocess  # nosec B404 — git invocation, validated args, no shell
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openlithohub.data.base import DatasetAdapter, LithoSample, natural_sort_key

_UPSTREAM_REPO = "https://github.com/phdyang007/GAN-OPC"
# Pin to the upstream commit captured in acquisition_log.md so that a
# silent rewrite of ``main`` cannot reshape downstream training runs.
# Override via the ``revision`` argument to ``download_ganopc`` if a new
# upstream tag has been validated against the eval pipeline.
_DEFAULT_REVISION = "main"


def download_ganopc(
    root: str | Path,
    *,
    revision: str = _DEFAULT_REVISION,
    repo_url: str = _UPSTREAM_REPO,
) -> Path:
    """Clone GAN-OPC and extract the multi-volume 7z into ``<root>/ganopc-data``.

    Idempotent: if ``<root>/ganopc-data/artitgt`` already exists the
    fetch is a no-op and the existing path is returned. Otherwise the
    upstream repo is shallow-cloned into ``<root>/.ganopc-src``, the 30
    archive volumes (``ganopc-data.7z.001`` … ``.030``) are joined via
    :mod:`multivolumefile`, and the resulting tree is extracted in place
    via :mod:`py7zr`.

    Returns the path to the extracted ``ganopc-data`` directory.

    Raises:
        ImportError: ``py7zr`` or ``multivolumefile`` is not installed.
        FileNotFoundError: ``git`` is not on ``PATH``.
        RuntimeError: the upstream layout no longer matches the expected
            ``ganopc-data.7z.NNN`` shape — usually means the repo moved.
    """
    dest_root = Path(root)
    dest_root.mkdir(parents=True, exist_ok=True)
    extracted = dest_root / "ganopc-data"
    if (extracted / "artitgt").is_dir() and (extracted / "artimsk").is_dir():
        return extracted

    if shutil.which("git") is None:
        raise FileNotFoundError(
            "git is required to fetch GAN-OPC but was not found on PATH. "
            "Install git or pre-populate <root>/ganopc-data/{artitgt,artimsk}/ "
            "manually."
        )
    try:
        import multivolumefile  # type: ignore[import-not-found]
        import py7zr  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "GAN-OPC auto-fetch requires py7zr and multivolumefile. Install "
            "them with: pip install py7zr multivolumefile"
        ) from e

    src = dest_root / ".ganopc-src"
    if not src.exists():
        # ``git`` is resolved via PATH (``shutil.which`` checked above) and
        # every argument is a literal or validated string — no shell
        # interpolation. S603/S607 ignored at file level via pyproject.toml.
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                revision,
                repo_url,
                str(src),
            ],
            check=True,
        )

    volumes = sorted(src.glob("ganopc-data.7z.*"))
    if not volumes:
        raise RuntimeError(
            f"Upstream layout has changed: no ganopc-data.7z.NNN volumes under {src}. "
            "Open an issue at https://github.com/OpenLithoHub/OpenLithoHub/issues."
        )

    archive_prefix = volumes[0].with_suffix("")  # strip ``.001`` etc.
    with (
        multivolumefile.open(str(archive_prefix), mode="rb") as joined,
        py7zr.SevenZipFile(joined, mode="r") as archive,
    ):
        archive.extractall(path=dest_root)

    if not (extracted / "artitgt").is_dir() or not (extracted / "artimsk").is_dir():
        raise RuntimeError(
            f"Extraction completed but {extracted}/{{artitgt,artimsk}}/ are missing. "
            "Inspect the archive contents and report at "
            "https://github.com/OpenLithoHub/OpenLithoHub/issues."
        )
    return extracted


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
        """Fetch the GAN-OPC training set from upstream on first use.

        Mirrors the pattern in :class:`LithoBenchDataset.download`: clones
        the upstream repository, joins the 30-volume 7z archive, and
        extracts the resulting tree so that
        ``<root>/ganopc-data/{artitgt,artimsk}/`` is populated.

        Idempotent: if ``<root>/ganopc-data/artitgt`` already exists the
        call is a no-op. The intermediate clone and joined archive are
        kept on disk so a partial extraction can resume without re-cloning.

        Requires ``git`` on ``PATH`` and the ``py7zr`` +
        ``multivolumefile`` Python packages (declared optional under
        the ``data`` extras).
        """
        download_ganopc(root)

    @property
    def sample_ids(self) -> list[str]:
        return list(self._ids)
