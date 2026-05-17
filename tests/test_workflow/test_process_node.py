"""Tests for openlithohub.workflow.process_node."""

from dataclasses import FrozenInstanceError

import pytest

from openlithohub.workflow.process_node import (
    PROCESS_NODES,
    ProcessNodeConfig,
    get_node,
    list_nodes,
)


class TestProcessNodeConfig:
    def test_all_presets_have_positive_sigma(self) -> None:
        for name, node in PROCESS_NODES.items():
            assert node.sigma_px > 0.0, f"{name} has non-positive sigma_px"

    def test_all_presets_have_positive_k1(self) -> None:
        for name, node in PROCESS_NODES.items():
            assert node.k1_factor > 0.0, f"{name} has non-positive k1"

    def test_euv_wavelength(self) -> None:
        euv_nodes = [n for n in PROCESS_NODES.values() if "euv" in n.name]
        for node in euv_nodes:
            assert node.wavelength_nm == 13.5

    def test_45nm_uses_193nm_wavelength(self) -> None:
        node = PROCESS_NODES["45nm"]
        assert node.wavelength_nm == 193.0
        assert node.numerical_aperture == 1.35

    def test_frozen_dataclass(self) -> None:
        node = PROCESS_NODES["3nm-euv"]
        with pytest.raises(FrozenInstanceError):
            node.wavelength_nm = 200.0  # type: ignore[misc]

    def test_sigma_px_calculation(self) -> None:
        node = ProcessNodeConfig(
            name="test",
            wavelength_nm=13.5,
            numerical_aperture=0.55,
            sigma_inner=0.2,
            sigma_outer=0.9,
            pixel_size_nm=1.0,
            min_feature_nm=14.0,
            min_spacing_nm=14.0,
        )
        expected = 0.5 * 13.5 / 0.55 / 1.0
        assert abs(node.sigma_px - expected) < 1e-6


class TestGetNode:
    def test_valid_node(self) -> None:
        node = get_node("3nm-euv")
        assert node.name == "3nm-euv"

    def test_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="Unknown process node"):
            get_node("nonexistent")

    def test_error_lists_available(self) -> None:
        with pytest.raises(KeyError, match="3nm-euv"):
            get_node("nonexistent")


class TestListNodes:
    def test_returns_sorted(self) -> None:
        nodes = list_nodes()
        assert nodes == sorted(nodes)

    def test_contains_expected_nodes(self) -> None:
        nodes = list_nodes()
        assert "3nm-euv" in nodes
        assert "45nm" in nodes
        assert "7nm" in nodes
