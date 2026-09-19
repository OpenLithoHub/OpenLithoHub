"""Model identity for theorem-facing certificates (P-054 repo integration).

A certificate must name *which frozen model implementation* it certifies —
repo, implementation commit, schema, fixture — instead of silently
inheriting a global pinned commit.  Historical replay keeps its original
commit; a new certificate must never masquerade as an old one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelIdentity:
    """Identity of the declared/frozen model a certificate is about."""

    repo: str
    implementation_commit: str
    model_schema: str
    fixture_id: str
    fixture_manifest_sha256: str
    paper_id: str | None = None
    paper_doi: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
