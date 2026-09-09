"""Validated one-dimensional boundary-root isolation primitives.

The caller supplies interval oracles for f([a,b]) and f'([a,b]).  This keeps
this module independent of a particular interval arithmetic backend while
making the proof obligation explicit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .types import BoundaryRootStatus, RootBracket


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("interval lo must not exceed hi")

    def excludes_zero(self) -> bool:
        return self.hi < 0.0 or self.lo > 0.0

    def is_strict_positive(self) -> bool:
        return self.lo > 0.0

    def is_strict_negative(self) -> bool:
        return self.hi < 0.0


IntervalOracle = Callable[[float, float], Interval]


@dataclass(frozen=True)
class BoundaryIsolationResult:
    status: BoundaryRootStatus
    brackets: tuple[RootBracket, ...]
    unresolved_intervals: tuple[tuple[float, float], ...]


def _opposite_sign(a: Interval, b: Interval) -> bool:
    return (a.is_strict_negative() and b.is_strict_positive()) or (
        a.is_strict_positive() and b.is_strict_negative()
    )


def isolate_roots_on_segment(
    f_interval: IntervalOracle,
    df_interval: IntervalOracle,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    max_depth: int = 40,
    min_width: float = 1e-12,
    root_width: float = 1e-6,
) -> BoundaryIsolationResult:
    """Certify root-free pieces or isolate unique transverse roots.

    A root bracket is accepted only when f' excludes zero on the whole
    subinterval and the endpoint point-enclosures have strict opposite signs.
    If any subinterval remains undecided at the resource limit, the result is
    INCONCLUSIVE even if other roots were isolated.
    """
    if lo >= hi:
        raise ValueError("require lo < hi")
    if root_width <= 0.0:
        raise ValueError("root_width must be positive")

    stack: list[tuple[float, float, int]] = [(lo, hi, 0)]
    roots: list[RootBracket] = []
    unresolved: list[tuple[float, float]] = []

    while stack:
        a, b, depth = stack.pop()
        fi = f_interval(a, b)
        if fi.excludes_zero():
            continue

        dfi = df_interval(a, b)
        fa = f_interval(a, a)
        fb = f_interval(b, b)
        monotone = dfi.excludes_zero()

        if monotone and _opposite_sign(fa, fb):
            if (b - a) <= root_width:
                roots.append(RootBracket(a, b))
                continue
            mid = (a + b) * 0.5
            stack.append((mid, b, depth + 1))
            stack.append((a, mid, depth + 1))
            continue

        # A strictly monotone function whose endpoints are strictly on the
        # same side cannot cross zero inside.
        if monotone and (
            (fa.is_strict_positive() and fb.is_strict_positive())
            or (fa.is_strict_negative() and fb.is_strict_negative())
        ):
            continue

        if depth >= max_depth or (b - a) <= min_width:
            unresolved.append((a, b))
            continue

        mid = (a + b) * 0.5
        stack.append((mid, b, depth + 1))
        stack.append((a, mid, depth + 1))

    roots.sort(key=lambda r: (r.lo, r.hi))
    if unresolved:
        status = BoundaryRootStatus.INCONCLUSIVE
    elif roots:
        status = BoundaryRootStatus.ROOT_BRACKETS
    else:
        status = BoundaryRootStatus.ROOT_FREE_CERTIFIED
    return BoundaryIsolationResult(status, tuple(roots), tuple(unresolved))
