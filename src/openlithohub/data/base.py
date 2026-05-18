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
