"""Model hub for downloading and caching pretrained weights."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

_DEFAULT_CACHE_DIR = Path.home() / ".openlithohub" / "models"


class ModelHub:
    """Manages download and caching of pretrained model weights.

    Supports HuggingFace Hub (if installed) and direct URL downloads.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_weights(
        self,
        model_id: str,
        filename: str = "model.pt",
        revision: str = "main",
    ) -> Path:
        """Download model weights, returning the cached file path.

        Tries HuggingFace Hub first; falls back to direct URL if model_id is a URL.
        """
        cached_path = self.cache_dir / model_id.replace("/", "--") / filename
        if cached_path.exists():
            return cached_path

        if model_id.startswith("http://") or model_id.startswith("https://"):
            return self._download_url(model_id, cached_path)

        try:
            return self._download_hf_hub(model_id, filename, revision)
        except ImportError:
            raise ImportError(
                f"Cannot download model '{model_id}': huggingface_hub not installed. "
                "Install with: pip install openlithohub[models]"
            ) from None

    def _download_hf_hub(self, repo_id: str, filename: str, revision: str) -> Path:
        """Download from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(self.cache_dir),
        )
        return Path(path)

    def _download_url(self, url: str, target: Path) -> Path:
        """Download from a direct URL with timeout and size limit."""
        if not url.startswith("https://"):
            raise ValueError("Only HTTPS URLs are supported for model downloads")
        target.parent.mkdir(parents=True, exist_ok=True)
        max_size = 2 * 1024 * 1024 * 1024  # 2 GB
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=300) as response:  # noqa: S310
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_size:
                raise ValueError(
                    f"File size {int(content_length)} bytes exceeds limit of {max_size} bytes"
                )
            downloaded = 0
            with open(target, "wb") as f:
                while chunk := response.read(8192):
                    downloaded += len(chunk)
                    if downloaded > max_size:
                        target.unlink(missing_ok=True)
                        raise ValueError(f"Download exceeded size limit of {max_size} bytes")
                    f.write(chunk)
        return target

    def list_cached(self) -> list[str]:
        """List model IDs that have cached weights."""
        if not self.cache_dir.exists():
            return []
        return sorted(d.name.replace("--", "/") for d in self.cache_dir.iterdir() if d.is_dir())

    def clear_cache(self, model_id: str | None = None) -> None:
        """Remove cached weights for a model (or all if model_id is None)."""
        import shutil

        if model_id is None:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            model_dir = self.cache_dir / model_id.replace("/", "--")
            if model_dir.exists():
                shutil.rmtree(model_dir)

    def get_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()
