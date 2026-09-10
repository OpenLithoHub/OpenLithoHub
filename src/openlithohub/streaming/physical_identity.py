"""Physical-instance identity for vector-layout verification.

Two kinds of repetition must not be conflated:

1. source repetition: different cell/array/repetition members are distinct
   physical instances, even when their transformed geometry is identical;
2. read repetition: the same physical instance observed through two tile/halo
   windows is the same owner and must not be double charged.

The physical key therefore excludes tile/read coordinates.  A read-view key
adds them explicitly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class InstancePathElement:
    source_cell_index: int
    array_i: int
    array_j: int
    specific_transform: str

    def token(self) -> str:
        return (
            f"cell={self.source_cell_index};"
            f"ia={self.array_i};ib={self.array_j};"
            f"tr={self.specific_transform}"
        )


@dataclass(frozen=True, order=True)
class PhysicalInstanceKey:
    """Identity of one physical source primitive after instantiation."""

    source_format: str
    source_cell_index: int
    source_shape_fingerprint: str
    instance_path: tuple[InstancePathElement, ...]
    quantization_policy: str

    def canonical_json(self) -> str:
        payload = {
            "source_format": self.source_format,
            "source_cell_index": self.source_cell_index,
            "source_shape_fingerprint": self.source_shape_fingerprint,
            "instance_path": [
                {
                    "source_cell_index": e.source_cell_index,
                    "array_i": e.array_i,
                    "array_j": e.array_j,
                    "specific_transform": e.specific_transform,
                }
                for e in self.instance_path
            ],
            "quantization_policy": self.quantization_policy,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def owner_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:32]


@dataclass(frozen=True, order=True)
class ReadViewKey:
    """One read/clip view of a physical instance."""

    physical_owner_id: str
    tile_id: str
    bbox_xyxy: tuple[int, int, int, int]

    @property
    def token(self) -> str:
        raw = (
            self.physical_owner_id,
            self.tile_id,
            self.bbox_xyxy,
        )
        return hashlib.sha256(repr(raw).encode()).hexdigest()[:24]


def distinct_source_members(keys: Iterable[PhysicalInstanceKey]) -> bool:
    ids = [k.owner_id for k in keys]
    return len(ids) == len(set(ids))
