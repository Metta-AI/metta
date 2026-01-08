"""Tests for dynamic inventory limits with modifiers.

This tests the feature where inventory limits can scale based on other items held.
For example, battery limit starts at 0, each gear adds +1 capacity.

Note: Integration tests for dynamic limits are done in C++ (test_has_inventory.cpp)
because Python's set_inventory uses unordered_map iteration which has undefined order.
"""

from mettagrid.config.mettagrid_config import (
    ResourceLimitsConfig,
)


def test_resource_limits_config_with_modifiers():
    """Test that ResourceLimitsConfig correctly stores modifiers."""
    config = ResourceLimitsConfig(
        resources=["battery"],
        limit=0,
        modifiers={"gear": 5, "wrench": 3},
    )

    assert config.resources == ["battery"]
    assert config.limit == 0
    assert config.modifiers == {"gear": 5, "wrench": 3}


def test_resource_limits_config_default_modifiers():
    """Test that ResourceLimitsConfig has empty modifiers by default."""
    config = ResourceLimitsConfig(
        resources=["gold"],
        limit=100,
    )

    assert config.resources == ["gold"]
    assert config.limit == 100
    assert config.modifiers == {}


def test_resource_limits_config_model_dump():
    """Test that modifiers are correctly serialized in model_dump."""
    config = ResourceLimitsConfig(
        resources=["energy"],
        limit=0,
        modifiers={"battery": 25},
    )

    dumped = config.model_dump()
    assert dumped["resources"] == ["energy"]
    assert dumped["limit"] == 0
    assert dumped["modifiers"] == {"battery": 25}


def test_resource_limits_config_empty_modifiers_dump():
    """Test that empty modifiers are correctly serialized."""
    config = ResourceLimitsConfig(
        resources=["ore"],
        limit=50,
    )

    dumped = config.model_dump()
    assert dumped["modifiers"] == {}
