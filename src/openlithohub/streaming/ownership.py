"""Leaf ownership tree for the streaming pipeline (R17 C3a).

Identity / ownership invariants:

- every physical core area is owned by exactly one *final* leaf;
- ``subdivide`` retires the parent leaf and installs an area-conserving
  partition ``P = C1 ⊔ ... ⊔ Cm`` (no overlap, no gap, all inside the
  parent);
- a child inherits the parent's forward history (``F(Cj) ← F(P)``): if the
  parent already ran the expensive forward model, every child is recorded
  as having seen that work;
- a retired parent never re-enters final completeness.

The tree deliberately knows nothing about verifiers or metrics — it is the
single source of truth for *who owns which physical area now*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import BoundingBox

TERMINAL_ACTIVE = "ACTIVE"
TERMINAL_EXACT_SKIP = "EXACT_OUTPUT_SKIP"
TERMINAL_VERIFY_SKIP = "VERIFICATION_ONLY_SKIP"

_TERMINAL_DISPOSITIONS = (TERMINAL_ACTIVE, TERMINAL_EXACT_SKIP, TERMINAL_VERIFY_SKIP)


@dataclass
class OwnershipLeaf:
    leaf_id: str
    core_bbox: BoundingBox
    parent_id: str | None
    forward_seen: bool = False
    terminal_disposition: str | None = None


def partition_core(parent: BoundingBox, subdivision: int) -> list[BoundingBox]:
    """Deterministic row-major ``subdivision``-way grid split of ``parent``.

    Area-conserving for every input: sum(child areas) == parent area, no
    overlap, no gap, children inside the parent — including edge cases the
    naive ``size // subdivision`` split gets wrong (1xN, Nx1, non-divisible
    sides).
    """
    if subdivision < 2:
        raise ValueError("subdivision must be >= 2")
    cols = subdivision
    rows = subdivision
    xs = _split_axis(parent.x0, parent.x1, cols)
    ys = _split_axis(parent.y0, parent.y1, rows)
    boxes: list[BoundingBox] = []
    for y0, y1 in ys:
        for x0, x1 in xs:
            if x1 > x0 and y1 > y0:
                boxes.append(BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1))
    return boxes


def _split_axis(lo: int, hi: int, parts: int) -> list[tuple[int, int]]:
    span = hi - lo
    base, rem = divmod(span, parts)
    edges = [lo]
    for i in range(parts):
        edges.append(edges[-1] + base + (1 if i < rem else 0))
    return [(edges[i], edges[i + 1]) for i in range(parts)]


@dataclass
class OwnershipTree:
    """Leaf states, parent→children edges, and the current final-leaf set."""

    _leaves: dict[str, OwnershipLeaf] = field(default_factory=dict)
    _children: dict[str, list[str]] = field(default_factory=dict)
    _final: set[str] = field(default_factory=set)

    def add_root(self, leaf_id: str, core_bbox: BoundingBox) -> OwnershipLeaf:
        if leaf_id in self._leaves:
            raise ValueError(f"leaf {leaf_id!r} already exists")
        leaf = OwnershipLeaf(leaf_id=leaf_id, core_bbox=core_bbox, parent_id=None)
        self._leaves[leaf_id] = leaf
        self._final.add(leaf_id)
        return leaf

    def get(self, leaf_id: str) -> OwnershipLeaf:
        return self._leaves[leaf_id]

    def children(self, leaf_id: str) -> list[str]:
        return list(self._children.get(leaf_id, ()))

    def final_leaves(self) -> list[OwnershipLeaf]:
        return [self._leaves[leaf_id] for leaf_id in self._final]

    def is_final(self, leaf_id: str) -> bool:
        return leaf_id in self._final

    def subdivide(
        self,
        parent_id: str,
        child_cores: list[tuple[str, BoundingBox]],
    ) -> list[OwnershipLeaf]:
        """Retire the parent and install an area-conserving child partition.

        Children inherit ``forward_seen`` from the parent (§10.3).  The
        partition is verified exactly: sum of child areas equals the parent
        area, boxes are disjoint and inside the parent — violations fail
        closed before any state mutates.
        """
        parent = self._leaves[parent_id]
        if parent_id not in self._final:
            raise ValueError(f"leaf {parent_id!r} is not a final leaf")
        self._check_partition(parent.core_bbox, child_cores)
        # fail-closed validation done; mutate
        self._final.discard(parent_id)
        child_leaves: list[OwnershipLeaf] = []
        for child_id, core in child_cores:
            if child_id in self._leaves:
                raise ValueError(f"leaf {child_id!r} already exists")
            leaf = OwnershipLeaf(
                leaf_id=child_id,
                core_bbox=core,
                parent_id=parent_id,
                forward_seen=parent.forward_seen,
            )
            self._leaves[child_id] = leaf
            self._final.add(child_id)
            child_leaves.append(leaf)
        self._children[parent_id] = [leaf.leaf_id for leaf in child_leaves]
        return child_leaves

    def mark_forward(self, leaf_id: str) -> None:
        self._leaves[leaf_id].forward_seen = True

    def set_terminal(self, leaf_id: str, disposition: str) -> None:
        if disposition not in _TERMINAL_DISPOSITIONS:
            raise ValueError(f"unknown terminal disposition {disposition!r}")
        leaf = self._leaves[leaf_id]
        if leaf.terminal_disposition is not None:
            raise ValueError(
                f"leaf {leaf_id!r} already has terminal disposition {leaf.terminal_disposition!r}"
            )
        leaf.terminal_disposition = disposition

    @staticmethod
    def _check_partition(
        parent_core: BoundingBox, child_cores: list[tuple[str, BoundingBox]]
    ) -> None:
        if not child_cores:
            raise ValueError("subdivision produced no children")
        total = 0
        seen: set[tuple[int, int, int, int]] = set()
        for _child_id, core in child_cores:
            if (
                core.x0 < parent_core.x0
                or core.y0 < parent_core.y0
                or core.x1 > parent_core.x1
                or core.y1 > parent_core.y1
            ):
                raise ValueError("child core escapes the parent core")
            key = (core.x0, core.y0, core.x1, core.y1)
            if key in seen:
                raise ValueError("duplicate child core in partition")
            seen.add(key)
            total += core.area
        if total != parent_core.area:
            raise ValueError(
                f"subdivision is not area-conserving: children cover {total} "
                f"of {parent_core.area} px"
            )

    # ------------------------------------------------------------------
    # R17 C4a — terminal coverage ledger
    # ------------------------------------------------------------------

    def terminal_pixel_counts(self) -> dict[str, int]:
        """Area per terminal disposition over the FINAL leaves.

        Fails closed on an unterminated final leaf or an unknown
        disposition; combined with the area-conserving partition invariant
        this makes ``sum(counts) == full_chip_pixels`` checkable.
        """
        counts = {
            TERMINAL_ACTIVE: 0,
            TERMINAL_EXACT_SKIP: 0,
            TERMINAL_VERIFY_SKIP: 0,
        }
        for leaf in self.final_leaves():
            if leaf.terminal_disposition is None:
                raise ValueError(f"final leaf {leaf.leaf_id!r} has no terminal disposition")
            counts[leaf.terminal_disposition] += leaf.core_bbox.area
        return counts

    def verify_total_coverage(self, full_chip_pixels: int) -> dict[str, int]:
        """Fail-closed coverage check: final leaves partition the chip."""
        counts = self.terminal_pixel_counts()
        total = sum(counts.values())
        if total != full_chip_pixels:
            raise ValueError(
                "terminal coverage ledger does not partition the chip: "
                f"{counts} sums to {total}, expected {full_chip_pixels}"
            )
        return counts
