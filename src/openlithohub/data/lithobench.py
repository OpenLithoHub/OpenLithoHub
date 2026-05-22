"""LithoBench dataset adapter (.npy format).

LithoBench (NeurIPS'23) organizes data as paired .npy arrays per sample:
    root/
      design/
        sample_0000.npy   # binary design layout (H, W)
        sample_0001.npy
        ...
      mask/
        sample_0000.npy   # optimized mask (H, W), may not exist for all samples
        ...
      resist/
        sample_0000.npy   # simulated resist contour (H, W), optional
        ...
      metadata.json       # optional: per-sample process parameters

Alternatively, a flat layout is supported:
    root/
      sample_0000_design.npy
      sample_0000_mask.npy
      sample_0000_resist.npy
      ...
"""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from openlithohub._utils.integrity import KnownGoodHash, verify_sha256
from openlithohub.data.base import DatasetAdapter, LithoSample, natural_sort_key

_FILENAME_RE = re.compile(r"^(?P<sample_id>.+?)_(?P<kind>design|mask|resist)\.npy$")
_VALID_KINDS: frozenset[str] = frozenset({"design", "mask", "resist"})

# Per playbook §6: every adapter that fetches bytes from a mutable URL
# pins them via SHA-256. Keys are the canonical artifact filenames as
# distributed upstream. Values were captured at acquisition time and
# the source attribution lets future maintainers audit each pin.
KNOWN_GOOD_SHA256: dict[str, KnownGoodHash] = {
    "lithomodels.tar.gz": KnownGoodHash(
        sha256="7fd9f08981caa99dd61a4dd2cd433efba6f7628b3142a3d2570f34364169d2b2",
        size_bytes=3_012_669_629,
        source="acquisition_log.md 2026-05-22 (gdrive 1N-VCv0gX49zzVWlwSs0yDqq2zKNQHKNB)",
    ),
    # ``lithodata.tar.gz`` (~14.7 GB) is rate-limited by Google Drive
    # behind a 24h cool-down; pin to be added once a clean local copy
    # exists. See ``acquisition_log.md`` row "LithoBench-data" for the
    # in-flight resume.
}

# Upstream Google Drive IDs for LithoBench artifacts. Recorded in
# ``acquisition_log.md`` 2026-05-22; bytes verified against
# ``KNOWN_GOOD_SHA256`` on every fetch so a silently re-uploaded artifact
# (gdrive doesn't pin contents to the file ID) cannot land here.
_GDRIVE_FILE_IDS: dict[str, str] = {
    "lithomodels.tar.gz": "1N-VCv0gX49zzVWlwSs0yDqq2zKNQHKNB",
}

Kind = Literal["design", "mask", "resist"]


class LithoBenchDataset(DatasetAdapter):
    """Adapter for the LithoBench dataset (NeurIPS'23, 45nm baseline).

    Supports two directory layouts:
    1. Subdirectory layout: root/{design,mask,resist}/sample_XXXX.npy
    2. Flat layout: root/sample_XXXX_{design,mask,resist}.npy

    Args:
        root: Path to the dataset directory.
        split: Optional split name (e.g. 'train', 'test'). If set, looks for root/split/.
        pixel_nm: Pixel resolution in nanometers (default 1.0 for LithoBench 45nm node).
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        pixel_nm: float = 1.0,
    ) -> None:
        self.root = Path(root)
        if split:
            self.root = self.root / split
        self.pixel_nm = pixel_nm
        self._index: list[str] = []
        self._layout: str = "unknown"
        self._metadata: dict[str, Any] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        design_dir = self.root / "design"
        if design_dir.is_dir():
            self._layout = "subdirectory"
            self._index = sorted((p.stem for p in design_dir.glob("*.npy")), key=natural_sort_key)
        else:
            self._layout = "flat"
            seen: set[str] = set()
            for p in self.root.glob("*.npy"):
                m = _FILENAME_RE.match(p.name)
                if m and m.group("kind") == "design":
                    seen.add(m.group("sample_id"))
            self._index = sorted(seen, key=natural_sort_key)

        meta_path = self.root / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    self._metadata = json.load(f)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Corrupt LithoBench metadata at {meta_path}: {exc}") from exc

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> LithoSample:
        if index < 0 or index >= len(self._index):
            raise IndexError(f"Index {index} out of range [0, {len(self._index)})")

        sample_id = self._index[index]
        design = self._load_array(sample_id, "design")
        mask = self._try_load_array(sample_id, "mask")
        resist = self._try_load_array(sample_id, "resist")

        metadata: dict[str, Any] = {
            "dataset": "lithobench",
            "sample_id": sample_id,
            "pixel_nm": self.pixel_nm,
        }
        if sample_id in self._metadata:
            metadata.update(self._metadata[sample_id])

        return LithoSample(
            design=torch.from_numpy(design).float(),
            mask=torch.from_numpy(mask).float() if mask is not None else None,
            resist=torch.from_numpy(resist).float() if resist is not None else None,
            metadata=metadata,
        )

    def _resolve_path(self, sample_id: str, kind: str) -> Path:
        # Sample IDs feed directly into a filesystem path; refuse anything that
        # could escape ``self.root`` via traversal. The legitimate index
        # (populated from sorted file globs) only ever contains plain names,
        # so this is a guard against caller-supplied IDs (``has_kind``).
        if not sample_id or "/" in sample_id or "\\" in sample_id or sample_id in (".", ".."):
            raise ValueError(f"Invalid sample_id: {sample_id!r}")
        if kind not in _VALID_KINDS:
            raise ValueError(f"Invalid kind: {kind!r}. Expected one of {sorted(_VALID_KINDS)}.")
        if self._layout == "subdirectory":
            return self.root / kind / f"{sample_id}.npy"
        return self.root / f"{sample_id}_{kind}.npy"

    def _load_array(self, sample_id: str, kind: Kind) -> np.ndarray:
        path = self._resolve_path(sample_id, kind)
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return np.load(path, allow_pickle=False)  # type: ignore[no-any-return]

    def _try_load_array(self, sample_id: str, kind: Kind) -> np.ndarray | None:
        path = self._resolve_path(sample_id, kind)
        if path.exists():
            return np.load(path, allow_pickle=False)  # type: ignore[no-any-return]
        return None

    def has_kind(self, sample_id: str, kind: str) -> bool:
        """Return True if the file for (sample_id, kind) exists on disk."""
        return self._resolve_path(sample_id, kind).exists()

    def download(self, root: str, artifact: str = "lithomodels.tar.gz") -> None:
        """Download a pinned LithoBench artifact via gdown and verify its SHA-256.

        Args:
            root: Destination directory. Created if missing. The tar is
                streamed to ``<root>/<artifact>`` and extracted into
                ``<root>``; if the file already exists *and* matches the
                pinned hash, the download is skipped (idempotent resume).
            artifact: Canonical filename of the artifact to fetch. Must be
                a key in :data:`KNOWN_GOOD_SHA256` and :data:`_GDRIVE_FILE_IDS`.

        Raises:
            ValueError: ``artifact`` is unknown.
            ImportError: ``gdown`` is not installed (``pip install gdown``).
            IntegrityError: bytes-on-disk don't match the pinned SHA-256.
        """
        if artifact not in KNOWN_GOOD_SHA256:
            raise ValueError(
                f"Unknown LithoBench artifact: {artifact!r}. "
                f"Known artifacts: {sorted(KNOWN_GOOD_SHA256)}"
            )
        if artifact not in _GDRIVE_FILE_IDS:
            raise NotImplementedError(
                f"No Google Drive ID registered for {artifact!r}. Open an issue or PR to add one."
            )

        try:
            import gdown
        except ImportError as e:
            raise ImportError(
                "Auto-fetching LithoBench requires gdown. Install it with: pip install gdown"
            ) from e

        dest_root = Path(root)
        dest_root.mkdir(parents=True, exist_ok=True)
        target = dest_root / artifact
        pin = KNOWN_GOOD_SHA256[artifact]

        if target.exists():
            try:
                verify_sha256(target, pin)
                self._extract_tarball(target, dest_root)
                return
            except Exception:
                # Stale or corrupt; re-download below.
                target.unlink()

        url = f"https://drive.google.com/uc?id={_GDRIVE_FILE_IDS[artifact]}"
        # ``resume=True`` is the gdown analogue of ``--continue``: a partial
        # ``<target>`` from a prior aborted run is appended to instead of
        # restarted from byte 0 — important for the ~14.7 GB ``lithodata``
        # tar where Google Drive can drop the connection mid-stream.
        # Proxy passthrough flows through ``HTTPS_PROXY`` / ``HTTP_PROXY``
        # env vars (see docs/developer-guide/network.md); we deliberately
        # do NOT name internal hosts in code per ``feedback_proxy_usage.md``.
        try:
            gdown.download(url, str(target), quiet=False, resume=True)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — re-raised below
            msg = str(exc).lower()
            if "quota" in msg or "rate" in msg or "too many requests" in msg:
                raise RuntimeError(
                    f"Google Drive rate-limited the LithoBench artifact "
                    f"({artifact!r}). Wait 24h and retry, or fetch from a "
                    f"different network. Original error: {exc}"
                ) from exc
            raise

        verify_sha256(target, pin)
        self._extract_tarball(target, dest_root)

    @staticmethod
    def _extract_tarball(tar_path: Path, dest: Path) -> None:
        """Extract ``tar_path`` into ``dest``, refusing path-traversal members.

        ``tarfile.extractall`` historically followed ``../`` and absolute
        member names — we add an explicit guard so a tampered upload (the
        SHA-256 is verified before this is called, but in case of future
        re-pinning) cannot escape ``dest``.
        """
        dest_resolved = dest.resolve()
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar.getmembers():
                member_path = (dest / member.name).resolve()
                if not str(member_path).startswith(str(dest_resolved)):
                    raise RuntimeError(
                        f"Refusing to extract path-traversal member: {member.name!r}"
                    )
            # B202: members were validated above; safe to extract.
            # ``filter="data"`` activates Python 3.12+ tar filter that
            # additionally blocks symlinks/hardlinks pointing outside the
            # destination — defence in depth on top of the manual guard.
            tar.extractall(dest, filter="data")  # nosec B202

    @property
    def sample_ids(self) -> list[str]:
        return list(self._index)

    # ---- Croissant metadata ----

    def croissant_name(self) -> str:
        return "LithoBench"

    def croissant_description(self) -> str:
        return (
            "LithoBench (NeurIPS'23) is a public benchmark for AI computational "
            "lithography spanning multiple design topologies and metrics. This "
            "adapter ingests the .npy distribution as (design, mask, resist) triples."
        )

    def croissant_url(self) -> str | None:
        return "https://github.com/shelljane/lithobench"

    def croissant_citation(self) -> str | None:
        return (
            "Zheng, S., Yang, H., Yu, B. et al. LithoBench: Benchmarking AI "
            "Computational Lithography for Semiconductor Manufacturing. NeurIPS 2023."
        )
