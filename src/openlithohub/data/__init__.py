"""Layer 1: Unified Data Adapter — loads lithography datasets into PyTorch tensors."""

from openlithohub.data.base import DatasetAdapter, LithoSample
from openlithohub.data.dummy import DummyLayoutSpec, generate_dummy_layout, generate_dummy_pair
from openlithohub.data.ganopc import GanOpcDataset
from openlithohub.data.iccad16 import HotspotAnnotation, Iccad16Dataset
from openlithohub.data.lithobench import LithoBenchDataset
from openlithohub.data.lithosim import LithoSimDataset

__all__ = [
    "DatasetAdapter",
    "LithoSample",
    "GanOpcDataset",
    "Iccad16Dataset",
    "HotspotAnnotation",
    "LithoBenchDataset",
    "LithoSimDataset",
    "DummyLayoutSpec",
    "generate_dummy_layout",
    "generate_dummy_pair",
]
