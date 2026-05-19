"""Sampling helpers shared across compliance/benchmark modules."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray


def evenly_spaced_indices(total: int, n: int) -> NDArray[np.int64]:
    """Return ``n`` indices into ``range(total)`` distributed evenly.

    Guarantees exactly ``min(total, n)`` indices and avoids the off-by-one
    that ``range(0, total, total // n)`` produces when ``total`` is not a
    clean multiple of ``n`` (e.g. ``total=10, n=3`` -> 4 samples instead of 3).
    """
    if total <= 0 or n <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(total, n)
    if k == 1:
        return np.array([0], dtype=np.int64)
    return cast(NDArray[np.int64], np.linspace(0, total - 1, num=k).round().astype(np.int64))
