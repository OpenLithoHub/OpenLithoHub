"""Model hub for downloading and caching pretrained weights.

Remote downloads (direct HTTPS URLs) require a known SHA256 — `torch.load`
on attacker-controlled bytes is a known RCE vector even with
``weights_only=True``. The HuggingFace path relies on the Hub's own
content-addressed verification.
"""

from __future__ import annotations

import hashlib
import ipaddress
import socket
import urllib.parse
import urllib.request
from pathlib import Path

_DEFAULT_CACHE_DIR = Path.home() / ".openlithohub" / "models"


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's SHA256 does not match the expected value."""


def _reject_internal_host(url: str) -> None:
    """Resolve `url`'s host and refuse private/loopback/link-local/multicast IPs.

    Without this, an HTTPS URL pointing at e.g. 169.254.169.254 (cloud metadata)
    or 10.0.0.0/8 (internal services) would be fetched by the model hub if a
    user passed a malicious URL. The SHA256 contract limits damage but cannot
    prevent the request itself from reaching internal services.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL has no host component: {url}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host {host!r}: {exc}") from exc
    for info in infos:
        addr = info[4][0]
        ip = ipaddress.ip_address(addr)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(
                f"Refusing to download from internal/non-routable address {addr} "
                f"for host {host!r}"
            )


class ModelHub:
    """Manages download and caching of pretrained model weights.

    Supports HuggingFace Hub (if installed) and direct URL downloads.
    Direct URL downloads MUST come with a SHA256 checksum.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_weights(
        self,
        model_id: str,
        filename: str = "model.pt",
        revision: str = "main",
        sha256: str | None = None,
    ) -> Path:
        """Download model weights, returning the cached file path.

        Args:
            model_id: A HuggingFace repo ID (``owner/repo``) or an HTTPS URL.
            filename: File name within the repo (HF Hub only).
            revision: Git revision (HF Hub only).
            sha256: Required for direct HTTPS downloads. Hex digest of the
                expected file contents. The Hub path uses HuggingFace's own
                hash verification and ignores this argument.
        """
        cached_path = self.cache_dir / model_id.replace("/", "--") / filename
        if cached_path.exists():
            if sha256 is not None and self.get_checksum(cached_path) != sha256.lower():
                raise ChecksumMismatchError(
                    f"Cached file {cached_path} does not match expected SHA256."
                )
            return cached_path

        if model_id.startswith("http://") or model_id.startswith("https://"):
            if sha256 is None:
                raise ValueError(
                    "Direct URL downloads require a sha256= argument. "
                    "Pass the expected hex digest of the weights file."
                )
            return self._download_url(model_id, cached_path, sha256)

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

    def _download_url(self, url: str, target: Path, sha256: str) -> Path:
        """Download from a direct URL with timeout, size limit, and SHA256 verification."""
        if not url.startswith("https://"):
            raise ValueError("Only HTTPS URLs are supported for model downloads")
        _reject_internal_host(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        max_size = 2 * 1024 * 1024 * 1024  # 2 GB
        req = urllib.request.Request(url)
        sha = hashlib.sha256()
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
                    sha.update(chunk)
                    f.write(chunk)
        actual = sha.hexdigest()
        if actual != sha256.lower():
            target.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"SHA256 mismatch for {url}: expected {sha256.lower()}, got {actual}"
            )
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
