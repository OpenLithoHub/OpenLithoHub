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
import warnings
from pathlib import Path

_DEFAULT_CACHE_DIR = Path.home() / ".openlithohub" / "models"

# Pinned for reproducibility per playbook §6. ``main`` is mutable, so any
# load that resolves the default branch is, by definition, irreproducible —
# but it is the only safe default at the hub level (per-repo SHA pins live
# on the model entry in `models/registry.py`).
_DEFAULT_REVISION: str = "main"


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


def _resolve_and_vet(host: str) -> list[str]:
    """Resolve `host`, refuse private IPs, return every vetted IP to dial.

    Returns the full list of vetted addresses in `getaddrinfo` order so the
    caller can iterate them and fall back from a broken IPv6 record to a
    working IPv4 one (common in CI runners with disabled IPv6). The
    connection is later forced to whichever IP succeeds, so a DNS rebinder
    cannot swap in a private IP between vet-time and dial-time.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host {host!r}: {exc}") from exc
    if not infos:
        raise ValueError(f"Host {host!r} did not resolve to any address")
    addrs: list[str] = []
    seen: set[str] = set()
    for info in infos:
        addr = str(info[4][0])
        _vet_address(addr, host)
        if addr not in seen:
            seen.add(addr)
            addrs.append(addr)
    return addrs


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


def _safe_cache_segment(value: str, *, kind: str) -> str:
    """Return ``value`` validated and normalized as a single cache path segment.

    Refuses absolute paths, path separators, and ``..``/``.`` components —
    these would let a caller-controlled ``filename`` or model id escape the
    cache directory (e.g. ``../../etc/passwd``).

    For ``kind="model_id"`` the legal HF Hub form is ``owner/repo``; this
    helper accepts the raw id, validates each slash-separated component, and
    returns the ``owner--repo`` form used on disk. Folding normalization in
    here keeps the validation authoritative: a caller that only saw the
    pre-normalized value (e.g. ``../foo``) would otherwise rewrite slashes
    away (``..--foo``) and slip the ``..`` past per-segment checks.
    """
    if not value or value.strip() == "":
        raise ValueError(f"{kind} must be a non-empty string")
    if "\x00" in value or value.startswith("/") or value.startswith("\\"):
        raise ValueError(f"Refusing unsafe {kind}: {value!r}")
    parts = value.replace("\\", "/").split("/")
    if any(part in ("", "..", ".") for part in parts):
        raise ValueError(f"Refusing path-traversal in {kind}: {value!r}")
    if kind == "model_id":
        if len(parts) != 2:
            raise ValueError(
                f"Refusing {kind} {value!r}: HuggingFace Hub ids must be "
                f"exactly owner/repo (got {len(parts)} segments)."
            )
        return "--".join(parts)
    if len(parts) > 1:
        raise ValueError(f"Refusing path separator in {kind}: {value!r}")
    return parts[0]


class ModelHub:
    """Manages download and caching of pretrained model weights.

    Supports HuggingFace Hub (if installed) and direct URL downloads.
    Direct URL downloads MUST come with a SHA256 checksum.

    Cache-key contract
    ------------------
    Three identifier shapes flow through this class. Knowing which
    shape goes where prevents the kind of round-trip bug fixed in the
    May 2026 review:

    - ``owner/repo`` — the **public** form a caller passes to
      :meth:`download_weights` for a HuggingFace-style model.
    - ``owner--repo`` — the **on-disk** form, with the path separator
      rewritten to a double-dash so the segment is filesystem-safe.
      :meth:`list_cached` decodes this back to ``owner/repo``.
    - ``url--<hex>`` — the **on-disk** form for direct-URL downloads,
      where ``<hex>`` is the first 32 hex chars of
      ``sha256(url.encode())``. :meth:`list_cached` returns this
      verbatim (there is no public ``owner/repo``-shaped name to decode
      back to), and :meth:`clear_cache` accepts both the original URL
      and the verbatim ``url--<hex>`` segment so the
      ``list_cached → clear_cache`` pipeline composes cleanly.

    Every caller-supplied identifier passes through
    ``_safe_cache_segment`` (or the URL hash) before it touches the
    filesystem; ``..``, embedded slashes, NUL bytes, and absolute
    paths are rejected. URL-keyed segments accept only hex suffixes.
    """

    def __init__(self, cache_dir: Path | None = None, timeout: float = 300.0) -> None:
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def download_weights(
        self,
        model_id: str,
        filename: str = "model.pt",
        revision: str = _DEFAULT_REVISION,
        sha256: str | None = None,
    ) -> Path:
        """Download model weights, returning the cached file path.

        Args:
            model_id: A HuggingFace repo ID (``owner/repo``) or an HTTPS URL.
            filename: File name within the repo (HF Hub only).
            revision: Git revision (HF Hub only). Pinning to a commit hash
                (40-hex) makes the download exactly reproducible — the
                default ``"main"`` is mutable and a publisher push can
                change the bytes you receive.
            sha256: Hex digest of the expected file contents. **Verified
                on both paths** — direct HTTPS downloads and HF Hub
                downloads. Issue #20: previously the HF Hub branch
                silently ignored this argument, so a malicious or
                accidentally-mutated repo at the configured revision
                would have produced bytes whose hash did not match what
                the caller pinned. ``None`` skips verification (HF
                Hub-only — the URL branch still requires it).
        """
        if model_id.startswith("http://") or model_id.startswith("https://"):
            cache_segment = "url--" + hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:32]
        else:
            cache_segment = _safe_cache_segment(model_id, kind="model_id")
        safe_filename = _safe_cache_segment(filename, kind="filename")
        cached_path = self.cache_dir / cache_segment / safe_filename
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
            hf_path = self._download_hf_hub(model_id, filename, revision)
        except ImportError:
            raise ImportError(
                f"Cannot download model '{model_id}': huggingface_hub not installed. "
                "Install with: pip install openlithohub[models]"
            ) from None
        if sha256 is not None:
            actual = self.get_checksum(hf_path)
            if actual != sha256.lower():
                raise ChecksumMismatchError(
                    f"SHA256 mismatch for HF Hub file {model_id}/{filename} "
                    f"@ {revision}: expected {sha256.lower()}, got {actual}. "
                    f"The file at this revision differs from the digest you "
                    f"pinned — refuse to load."
                )
        elif revision in {"main", "master"} or revision is None:
            # Mutable revision + no sha256 = the bytes you receive today
            # may differ from the bytes you receive tomorrow. Warn so
            # downstream reproducibility claims aren't silently false.
            warnings.warn(
                f"download_weights({model_id!r}, revision={revision!r}) "
                f"with no sha256= and a mutable revision: the publisher "
                f"can change the bytes at any time. Pass either a "
                f"40-hex commit hash as `revision=` or a `sha256=` "
                f"digest for reproducibility.",
                UserWarning,
                stacklevel=2,
            )
        return hf_path

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
        # Build the request target by joining the (already URL-encoded) path
        # and query verbatim — re-quoting either part would corrupt
        # already-encoded characters. ``parsed.path`` is empty for URLs like
        # ``https://host?x=1``; default to ``/`` per HTTP/1.1 spec.
        request_target = parsed.path or "/"
        if parsed.query:
            request_target = f"{request_target}?{parsed.query}"
        # Preserve the port in the Host header for non-default ports — some
        # virtual-host-aware origins use it for routing.
        host_header = host if port == 443 else f"{host}:{port}"
        ips = _resolve_and_vet(host)

        target.parent.mkdir(parents=True, exist_ok=True)
        max_size = 2 * 1024 * 1024 * 1024  # 2 GB

        ctx = ssl.create_default_context()
        # Try each vetted IP in order — falls back from a broken IPv6 to a
        # working IPv4 record. SNI/host header still use the original hostname
        # so cert validation works.
        conn: _PinnedHTTPSConnection | None = None
        last_err: Exception | None = None
        for ip in ips:
            candidate = _PinnedHTTPSConnection(
                host, ip, port=port, timeout=self.timeout, context=ctx
            )
            try:
                candidate.connect()
            except OSError as exc:
                candidate.close()
                last_err = exc
                continue
            conn = candidate
            break
        if conn is None:
            raise ValueError(f"Could not connect to any vetted address for {host!r}: {last_err}")
        try:
            conn.request("GET", request_target, headers={"Host": host_header})
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
        """List model IDs that have cached weights.

        HF-Hub-style entries are decoded from the on-disk ``owner--repo`` form
        back to ``owner/repo``. URL-keyed entries (stored under
        ``url--<sha256>``) are returned verbatim so a caller can hand the
        result straight back to :meth:`clear_cache` without re-keying.
        """
        if not self.cache_dir.exists():
            return []
        names: list[str] = []
        for d in self.cache_dir.iterdir():
            if not d.is_dir():
                continue
            if d.name.startswith("url--"):
                names.append(d.name)
            else:
                names.append(d.name.replace("--", "/"))
        return sorted(names)

    def clear_cache(self, model_id: str | None = None) -> None:
        """Remove cached weights for a model (or all if model_id is None).

        ``model_id`` is routed through the same per-segment validator used by
        ``download_weights`` so a caller-supplied ``..`` cannot escape the
        cache directory and rmtree something it shouldn't. URL-keyed entries
        (cached under the ``url--<sha256>`` segment that ``download_weights``
        creates) are also accepted.
        """
        import shutil

        if model_id is None:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            return

        if model_id.startswith("http://") or model_id.startswith("https://"):
            cache_segment = "url--" + hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:32]
        elif model_id.startswith("url--"):
            # The exact on-disk segment as returned by `list_cached` for
            # URL-keyed entries. Validate the suffix is a clean hex segment
            # so a caller can't sneak `..` past us via this branch.
            suffix = model_id[len("url--") :]
            if not suffix or any(c not in "0123456789abcdef" for c in suffix):
                raise ValueError(f"Refusing unsafe url-keyed cache id: {model_id!r}")
            cache_segment = model_id
        else:
            cache_segment = _safe_cache_segment(model_id, kind="model_id")
        model_dir = self.cache_dir / cache_segment
        if model_dir.exists():
            shutil.rmtree(model_dir)

    def get_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()
