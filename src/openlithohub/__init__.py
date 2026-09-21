"""OpenLithoHub — vendor-neutral computational lithography platform for OPC/ILT
benchmarking, scalable layout processing, manufacturability analysis, model
deployment, and proof-carrying verification."""

from openlithohub._version import __version__
from openlithohub.api import LitheEngine, Mask, Report

__all__ = ["LitheEngine", "Mask", "Report", "__version__"]
