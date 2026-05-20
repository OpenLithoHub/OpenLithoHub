"""Tests for Croissant 1.0 metadata export from DatasetAdapter."""

from __future__ import annotations

import json

import torch

from openlithohub.data.base import DatasetAdapter, LithoSample


class _ToyDataset(DatasetAdapter):
    """Minimal DatasetAdapter to exercise the base Croissant emitter."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> LithoSample:
        return LithoSample(design=torch.zeros(8, 8))

    def download(self, root: str) -> None:
        raise NotImplementedError


def test_base_croissant_minimum_compliant_keys() -> None:
    out = _ToyDataset().to_croissant()
    # MLCommons Croissant 1.0 requires at least @context, @type, name, description
    # and conformsTo declaration. The recordSet describes per-sample fields.
    assert out["@type"] == "sc:Dataset"
    assert out["name"] == "_ToyDataset"
    assert "description" in out
    assert out["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert "@context" in out
    assert "cr" in out["@context"]
    assert isinstance(out["recordSet"], list)
    assert out["recordSet"][0]["@type"] == "cr:RecordSet"
    field_names = {f["name"] for f in out["recordSet"][0]["field"]}
    assert {"design", "mask", "resist"}.issubset(field_names)


def test_base_croissant_is_json_serializable() -> None:
    out = _ToyDataset().to_croissant()
    s = json.dumps(out)
    # Round-trip survives — guards against accidentally adding non-serializable
    # types (Path, Tensor, set) into the metadata payload.
    assert json.loads(s) == out


def test_base_croissant_omits_optional_fields_by_default() -> None:
    out = _ToyDataset().to_croissant()
    # The toy dataset overrides nothing, so optional fields stay absent
    # rather than being serialized as None (Croissant validators reject null).
    assert "license" not in out
    assert "citeAs" not in out
    assert "url" not in out


class _RichDataset(_ToyDataset):
    def croissant_name(self) -> str:
        return "RichDataset"

    def croissant_description(self) -> str:
        return "Rich description."

    def croissant_license_url(self) -> str | None:
        return "https://example.com/LICENSE"

    def croissant_citation(self) -> str | None:
        return "Author. Title. Venue 2025."

    def croissant_url(self) -> str | None:
        return "https://example.com/dataset"


def test_base_croissant_overrides_propagate() -> None:
    out = _RichDataset().to_croissant()
    assert out["name"] == "RichDataset"
    assert out["description"] == "Rich description."
    assert out["license"] == "https://example.com/LICENSE"
    assert out["citeAs"] == "Author. Title. Venue 2025."
    assert out["url"] == "https://example.com/dataset"


def test_concrete_dataset_metadata_overrides() -> None:
    """Spot-check a couple of real adapters declare richer metadata.

    We don't need to instantiate them (their __init__ touches the
    filesystem) — call the hook methods directly on the unbound class.
    """
    from openlithohub.data.asap7 import Asap7Dataset
    from openlithohub.data.lithobench import LithoBenchDataset

    # Class-level hooks, no filesystem needed.
    assert Asap7Dataset.croissant_name(None) == "ASAP7"  # type: ignore[arg-type]
    assert "ASAP7" in Asap7Dataset.croissant_description(None)  # type: ignore[arg-type]
    assert Asap7Dataset.croissant_license_url(None).startswith("https://")  # type: ignore[union-attr,arg-type]

    assert LithoBenchDataset.croissant_name(None) == "LithoBench"  # type: ignore[arg-type]
    assert "NeurIPS" in LithoBenchDataset.croissant_citation(None)  # type: ignore[union-attr,arg-type]
