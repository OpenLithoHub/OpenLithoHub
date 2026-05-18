"""Model hub for downloading and caching pretrained weights.

Remote downloads (direct HTTPS URLs) require a known SHA256 — `torch.load`
on attacker-controlled bytes is a known RCE vector even with
``weights_only=True``. The HuggingFace path relies on the Hub's own
content-addressed verification.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import socket
import ssl
import urllib.parse
from pathlib import Path

_DEFAULT_CACHE_DIR = Path.home() / ".openlithohub" / "models"


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's SHA256 does not match the expected value."""


def _vet_address(addr: str, host: str) -> None:
    """Refuse a single IP literal that would point at a private/non-routable address."""
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
            f"Refusing to download from internal/non-routable address {addr} for host {host!r}"
        )


def _resolve_and_vet(host: str) -> str:
    """Resolve `host`, refuse private IPs, return the single vetted IP to dial.

    Returns one address — the connection is later forced to that exact IP so
    a DNS rebinder cannot swap in a private IP between vet-time and dial-time.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host {host!r}: {exc}") from exc
    if not infos:
        raise ValueError(f"Host {host!r} did not resolve to any address")
    for info in infos:
        addr = info[4][0]
        _vet_address(str(addr), host)
    return str(infos[0][4][0])


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that dials a fixed IP while presenting a hostname for SNI/cert.

    The constructor arg `host` is the original hostname (used for SNI, cert
    verification, and the HTTP Host header). Dialing happens to `_dial_ip`
    instead — closes the DNS rebinding window between vet-time and connect-time.
    """

    def __init__(
        self,
        host: str,
        dial_ip: str,
        port: int,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(host, port=port, timeout=timeout, context=context)
        self._dial_ip = dial_ip

    def connect(self) -> None:
        # Dial the vetted IP, but keep self.host (used for SNI + cert hostname).
        sock = socket.create_connection((self._dial_ip, self.port), timeout=self.timeout)
        if self._tunnel_host:  # type: ignore[attr-defined]
            self.sock = sock
            self._tunnel()  # type: ignore[attr-defined]
        assert self.context is not None  # type: ignore[attr-defined]
        self.sock = self.context.wrap_socket(sock, server_hostname=self.host)  # type: ignore[attr-defined]


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
        """Download from a direct URL with timeout, size limit, and SHA256 verification.

        Resolves the host once, refuses private/non-routable addresses, and
        then forces the TLS connection to that exact IP. This closes the DNS
        TOCTOU window that a plain ``urlopen(url)`` leaves open: a rebinder
        cannot return a public IP for the vetting query and a private IP for
        the actual fetch, because the fetch never re-resolves.
        """
        if not url.startswith("https://"):
            raise ValueError("Only HTTPS URLs are supported for model downloads")

        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        if not host:
            raise ValueError(f"URL has no host component: {url}")
        port = parsed.port or 443
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        ip = _resolve_and_vet(host)

        target.parent.mkdir(parents=True, exist_ok=True)
        max_size = 2 * 1024 * 1024 * 1024  # 2 GB

        ctx = ssl.create_default_context()
        # Connect to the vetted IP directly; SNI/host header still uses the
        # original hostname so cert validation works.
        conn = _PinnedHTTPSConnection(host, ip, port=port, timeout=300, context=ctx)
        try:
            conn.request("GET", path, headers={"Host": host})
            response = conn.getresponse()
            # Reject redirects — we vetted only the original host, and a
            # 30x to a different host would leak the SSRF guard.
            if response.status in (301, 302, 303, 307, 308):
                raise ValueError(
                    f"Refusing to follow redirect from {url} to {response.getheader('Location')!r}"
                )
            if response.status != 200:
                raise ValueError(f"Unexpected HTTP status {response.status} fetching {url}")
            content_length = response.getheader("Content-Length")
            if content_length and int(content_length) > max_size:
                raise ValueError(
                    f"File size {int(content_length)} bytes exceeds limit of {max_size} bytes"
                )
            sha = hashlib.sha256()
            downloaded = 0
            with open(target, "wb") as f:
                while chunk := response.read(8192):
                    downloaded += len(chunk)
                    if downloaded > max_size:
                        target.unlink(missing_ok=True)
                        raise ValueError(f"Download exceeded size limit of {max_size} bytes")
                    sha.update(chunk)
                    f.write(chunk)
        finally:
            conn.close()
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
