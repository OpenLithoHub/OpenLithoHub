"""Abstract base class for dataset adapters."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

_NAT_SPLIT_RE = re.compile(r"(\d+)")


def natural_sort_key(s: str) -> tuple[Any, ...]:
    """Sort key that orders strings with embedded numbers numerically.

    `sample_2` < `sample_10` < `sample_100`, instead of the lexical
    `sample_10` < `sample_2`.
    """
    parts = _NAT_SPLIT_RE.split(s)
    return tuple(int(p) if p.isdigit() else p for p in parts)


@dataclass
class LithoSample:
    """A single lithography sample with unified tensor representation."""

    design: torch.Tensor
    mask: torch.Tensor | None = None
    resist: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Abstract adapter for lithography datasets.

    Subclasses must implement __len__ and __getitem__ to provide
    unified PyTorch Tensor access regardless of underlying format.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> LithoSample: ...

    def __iter__(self) -> Iterator[LithoSample]:
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def download(self, root: str) -> None:
        """Download dataset to the specified root directory."""
        ...

    # ---- ML metadata ----

    def croissant_name(self) -> str:
        """Human-readable name for Croissant metadata. Defaults to class name."""
        return type(self).__name__

    def croissant_description(self) -> str:
        """Free-text description for Croissant metadata.

        Subclasses should override with a one-paragraph dataset summary.
        """
        return f"Lithography dataset of type {type(self).__name__}."

    def croissant_license_url(self) -> str | None:
        """Upstream license URL, or ``None`` when not applicable."""
        return None

    def croissant_citation(self) -> str | None:
        """BibTeX or free-text citation, or ``None``."""
        return None

    def croissant_url(self) -> str | None:
        """Canonical landing-page URL for the dataset, or ``None``."""
        return None

    def to_croissant(self) -> dict[str, Any]:
        """Emit MLCommons Croissant 1.0 JSON-LD metadata.

        Croissant is the de-facto ML dataset metadata schema (HuggingFace,
        Google, Kaggle, MLCommons; published 2024-12). Producing it from
        ``DatasetAdapter`` lets downstream MLPerf-style benchmarks
        consume our datasets without bespoke adapters.

        The output is a Python dict matching the JSON-LD shape — caller
        serialises it with ``json.dumps`` (default), or feeds it to a
        Croissant validator. We emit the minimum compliant subset:
        ``@context``, ``@type``, ``name``, ``description``, ``license``,
        ``url``, ``citeAs``, plus a single ``RecordSet`` describing the
        sample shape (design / mask / resist tensors). Subclasses can
        override hook methods (``croissant_name`` / ``..._description``
        / ...) to enrich the output.
        """
        record_fields = [
            {
                "@type": "cr:Field",
                "name": "design",
                "description": "Target design tensor (binary mask of intended features).",
                "dataType": "cr:Tensor",
            },
            {
                "@type": "cr:Field",
                "name": "mask",
                "description": "Optimised lithography mask tensor (post-OPC), if available.",
                "dataType": "cr:Tensor",
            },
            {
                "@type": "cr:Field",
                "name": "resist",
                "description": "Simulated/measured resist contour tensor, if available.",
                "dataType": "cr:Tensor",
            },
        ]
        out: dict[str, Any] = {
            "@context": {
                "@vocab": "https://schema.org/",
                "cr": "http://mlcommons.org/croissant/",
                "sc": "https://schema.org/",
                "data": {"@id": "cr:data", "@type": "@json"},
            },
            "@type": "sc:Dataset",
            "name": self.croissant_name(),
            "description": self.croissant_description(),
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "recordSet": [
                {
                    "@type": "cr:RecordSet",
                    "name": "samples",
                    "description": "Per-sample lithography records.",
                    "field": record_fields,
                }
            ],
        }
        if (lic := self.croissant_license_url()) is not None:
            out["license"] = lic
        if (cite := self.croissant_citation()) is not None:
            out["citeAs"] = cite
        if (url := self.croissant_url()) is not None:
            out["url"] = url
        return out
