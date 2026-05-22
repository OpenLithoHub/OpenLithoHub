"""GAN-OPC training-data adapter.

GAN-OPC ships its training set as ~4875 paired binary PNGs at 2048×2048
resolution. The public mirror is https://github.com/phdyang007/GAN-OPC,
distributed as a 30-volume 7z archive (``ganopc-data.7z.001`` … ``.030``).
The :func:`download_ganopc` helper auto-fetches and unpacks it on first
use; until then the repo carries no upstream bytes (per
``DATA-LICENSES.md`` — redistribution is not granted).

Reference: Yang et al., *GAN-OPC: Mask Optimization with Lithography-guided
Generative Adversarial Nets*, DAC 2018 (doi:10.1145/3195970.3196056).
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

from openlithohub._utils.integrity import KnownGoodHash, verify_sha256
from openlithohub.data.base import DatasetAdapter, LithoSample, natural_sort_key

_UPSTREAM_REPO = "https://github.com/phdyang007/GAN-OPC"
# Pin to the upstream commit captured in acquisition_log.md so that a
# silent rewrite of ``main`` cannot reshape downstream training runs.
# Override via the ``revision`` argument to ``download_ganopc`` if a new
# upstream tag has been validated against the eval pipeline.
_DEFAULT_REVISION = "main"

# Per playbook §6: the upstream tracks no LFS / no release artefact, so we
# pin every volume of the multi-volume 7z archive by SHA-256. Hashes were
# captured at acquisition time (acquisition_log.md 2026-05-18); a silent
# upstream rewrite that swaps a volume changes the hash and the fetcher
# refuses to extract.
KNOWN_GOOD_SHA256: dict[str, KnownGoodHash] = {
    "ganopc-data.7z.001": KnownGoodHash(
        sha256="04d17ec4c657496f9b0b2273e24a3b0c4dca5522a5c5b84024218eb4c798ab71",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.002": KnownGoodHash(
        sha256="6d69d2af585f4c32b85b00e8b7799221b1533ad3f91ea3f4e160b28c103b31a6",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.003": KnownGoodHash(
        sha256="818a71d8dd825e50eb0b2ecd72326f851f6d85056df4114cc89a747a7e21ad0a",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.004": KnownGoodHash(
        sha256="050def5b23e368fb62cf71067e48cf7bf89037608aabe4dc735a9fe96e125504",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.005": KnownGoodHash(
        sha256="acf8573d8ae7ca1528d7de1d0d29c4df242b547fd5a09c1803bec7e1590e757b",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.006": KnownGoodHash(
        sha256="5c82bdc4150acf329fed6b46c3bfae82f2379e1739c4b262fe103e2ed7c053fd",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.007": KnownGoodHash(
        sha256="ad8dd8385422482d6f5a21a8916001772522a4a85c68486e2c30b65c8c63b060",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.008": KnownGoodHash(
        sha256="207442ee2d1b75937b2c9a7878224cbe491febd2f7d43044944e6d1e8dff83cc",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.009": KnownGoodHash(
        sha256="3417f9869e1755affd1bca8eb7d7d822c2d88ed1e3e6426c6c53f0047ee4e7f6",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.010": KnownGoodHash(
        sha256="4bb3f3fd4a6f858b94ac9a72a8a8a9bdc7ff1ebcd2693e643396fef535bc828f",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.011": KnownGoodHash(
        sha256="0682f62825e1f51e4e8db0ec88adec8b29077f6702857769f92a705b6db77a79",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.012": KnownGoodHash(
        sha256="ceb24b62c5d79c1243172fe8ce76d057b9caf33f3f6a9733ee63475d7a85cfdd",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.013": KnownGoodHash(
        sha256="300183b68dcedba49332442a297e09df072739d3a71913000d3da99af7741417",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.014": KnownGoodHash(
        sha256="dafe634d35b7fed2999cafccec6c29203f93d565f2e0cce564f6dd1913d297b2",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.015": KnownGoodHash(
        sha256="7b4c0f68d839ae515427083e64c978a914976c84ed8e595d6d22dcbf02bb6324",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.016": KnownGoodHash(
        sha256="b9dfa1bbe32718f3c4fe74bef65109ecdd1fa8ac584cddc2feeeaf6afd385003",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.017": KnownGoodHash(
        sha256="0b096debd193e0559bb53b1e60e4f2def864f40d7ab1899579c5af18744cc2bb",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.018": KnownGoodHash(
        sha256="0003a46b7662451284a5aab60facf560907c2333cdb8ce8eb339f9858e22eea5",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.019": KnownGoodHash(
        sha256="c092570a9caa9a128a9ad3bc88436d73f09c6c65487ce30f9202d55c69eb77b8",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.020": KnownGoodHash(
        sha256="4561317f03adbe5d7d4db6396d846a43c5a5c67666ef49ac893efad8392b90c5",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.021": KnownGoodHash(
        sha256="4b74db6c80e7b6923c11adc9bb50e674b89455e3360ab068c1c7ef12f8f1d18b",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.022": KnownGoodHash(
        sha256="a48881bf970e5d0ff4742a1aca2b5f1a30cc366e3fffb7c6e9c2ff788a617aa9",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.023": KnownGoodHash(
        sha256="c184deef46664049c4a631db49b8638466f7889b27f9e7cf72b91d21dd477a98",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.024": KnownGoodHash(
        sha256="03169fc883aa3ec62044ae16a9da390d49642746e578c09d0d2f9adadd60c0b5",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.025": KnownGoodHash(
        sha256="08cbd616589633124611137343c39e4ccf5db4940498a38f1c2c480d21fec2ba",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.026": KnownGoodHash(
        sha256="54e02a31f77c1064d2c767f994d16f59e859b5ac3a89de0b6be515d4a44924b2",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.027": KnownGoodHash(
        sha256="6cf794720ca4b1a62e7c04e3e681f0ae4fe1aa5983058a759c531433eabdf013",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.028": KnownGoodHash(
        sha256="ff25024f20bdab10c5ada5dea7e0828ec9a98730f84f1fc7b218e48e64a5805e",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.029": KnownGoodHash(
        sha256="475647cd11dfd35714b65ff2631cf082e24bf5d01dcbe929100d5ef4ea6fc0cd",
        size_bytes=1048576,
        source="acquisition_log.md 2026-05-18",
    ),
    "ganopc-data.7z.030": KnownGoodHash(
        sha256="d8384f6bccae6696514d903b5d87067bb57961c38139ebe3892a56e20e1297df",
        size_bytes=617579,
        source="acquisition_log.md 2026-05-18",
    ),
}


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
        import multivolumefile
        import py7zr
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

    for volume in volumes:
        pin = KNOWN_GOOD_SHA256.get(volume.name)
        if pin is None:
            raise RuntimeError(
                f"Unexpected upstream volume {volume.name}: no SHA-256 pin in "
                "KNOWN_GOOD_SHA256. The upstream repo may have re-cut the archive."
            )
        verify_sha256(volume, pin)

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
