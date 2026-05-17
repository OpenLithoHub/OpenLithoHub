"""Internal shared utilities."""

from openlithohub._utils.morphology import (
    binary_dilation,
    binary_erosion,
    distance_transform,
)
from openlithohub._utils.tensor_ops import ensure_2d

__all__ = [
    "binary_dilation",
    "binary_erosion",
    "distance_transform",
    "ensure_2d",
]
