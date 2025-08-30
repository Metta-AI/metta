#!/usr/bin/env python3


import numpy as np
import pytest

from metta.mettagrid.core import MettaGridCore
from metta.mettagrid.map_builder.ascii import AsciiMapBuilder
from metta.mettagrid.mettagrid_c import dtype_actions
from metta.mettagrid.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    GameConfig,
    GlobalObsConfig,
    GroupConfig,
    MettaGridConfig,
    WallConfig,
)


def create_base_config(enable_visitation_counts=False):
    """Create base configuration with optional visitation counts.

    Args:
        enable_visitation_counts: If True, adds visitation_counts extension

    Returns:
        MettaGridConfig instance
    """
    base_config = {
        "num_agents": 1,
        "max_steps": 100,
        "obs_width": 5,
        "obs_height": 5,
        "num_observation_tokens": 100,
        "inventory_item_names": ["wood", "stone"],
        "actions": ActionsConfig(
            noop=ActionConfig(enabled=False),
            move=ActionConfig(enabled=True),
            get_items=ActionConfig(enabled=False),
        ),
        "objects": {"wall": WallConfig(type_id=1)},
        "groups": {"agent": GroupConfig(id=0)},
        "map_builder": AsciiMapBuilder.Config(
            map_data=[
                [".", ".", ".", ".", ".", ".", "."],
                [".", ".", ".", ".", ".", ".", "."],
                [".", ".", ".", ".", ".", ".", "."],
                [".", ".", ".", "@", ".", ".", "."],  # Agent in center
                [".", ".", ".", ".", ".", ".", "."],
                [".", ".", ".", ".", ".", ".", "."],
                [".", ".", ".", ".", ".", ".", "."],
            ]
        ),
         extensions=["episode_completion_pct", "last_action", "last_reward"]
    }

    if enable_visitation_counts:
        base_config["extensions"].append("visitation_counts")

    return MettaGridConfig(game=GameConfig(**base_config))


@pytest.fixture
def env_with_visitation():
    """Environment with visitation counts enabled."""
    config = create_base_config(enable_visitation_counts=True)
    return MettaGridCore(config)


@pytest.fixture
def env_without_visitation():
    """Environment with visitation counts disabled."""
    config = create_base_config(enable_visitation_counts=False)
    return MettaGridCore(config)


@pytest.fixture
def env_default():
    """Environment with default config (no visitation_counts specified)."""
    # This is now identical to env_without_visitation, but kept separate
    # for clarity of test intent
    config = create_base_config(enable_visitation_counts=False)
    return MettaGridCore(config)


# Helper functions
def extract_visitation_features(obs: np.ndarray, agent_idx: int = 0) -> list[int]:
    """Extract visitation count features from observation."""
    visitation_feature_id = 14
    features = []
    for i in range(obs.shape[1]):
        if obs[agent_idx, i, 1] == visitation_feature_id:
            features.append(obs[agent_idx, i, 2])
    return features


def count_visitation_features_at_center(obs: np.ndarray) -> int:
    """Count visitation count features at the center location."""
    visitation_feature_id = 14
    center_location = 34  # Center position in 5x5 observation grid

    count = 0
    num_agents = obs.shape[0]
    for agent_idx in range(num_agents):
        for token_idx in range(obs.shape[1]):
            if (
                obs[agent_idx, token_idx, 0] == center_location
                and obs[agent_idx, token_idx, 1] == visitation_feature_id
            ):
                count += 1
    return count


def get_agent_position(env: MettaGridCore) -> tuple[int, int]:
    """Get the current agent position."""
    grid_objects = env.grid_objects
    for obj in grid_objects.values():
        if "agent_id" in obj:
            return (obj["r"], obj["c"])
    raise ValueError("Agent not found in grid objects")


def test_visitation_counts_enabled(env_with_visitation):
    """Test that visitation counts work correctly when enabled."""
    obs, _ = env_with_visitation.reset(seed=42)

    # Check initial visitation counts
    initial_features = extract_visitation_features(obs)
    assert len(initial_features) == 5, f"Expected 5 visitation features, got {len(initial_features)}"
    # we mark our starting position as visited
    assert initial_features == [1, 0, 0, 0, 0], f"Expected initial counts [1,0,0,0,0], got {initial_features}"

    # Move and check counts update
    move_action = np.array([[0, 0]], dtype=dtype_actions)
    for step in range(3):
        obs, _, _, _, _ = env_with_visitation.step(move_action)
        features = extract_visitation_features(obs)
        assert len(features) == 5, f"Step {step}: Expected 5 features, got {len(features)}"
        # At least one count should be non-zero after movement
        # Note: visitation counts are incremented at the end of the step,
        # so we need to take at least 2 steps to see non-zero counts
        if step > 1:
            assert any(f > 0 for f in features), f"Expected non-zero counts after step {step}, got {features}"


def test_visitation_counts_disabled(env_without_visitation):
    """Test that visitation counts are not present when disabled."""
    obs, _ = env_without_visitation.reset(seed=42)

    features = extract_visitation_features(obs)
    assert len(features) == 0, f"Expected no visitation features when disabled, got {len(features)}"

    # Verify they stay disabled after steps
    move_action = np.array([[0, 0]], dtype=dtype_actions)
    obs, _, _, _, _ = env_without_visitation.step(move_action)
    features = extract_visitation_features(obs)
    assert len(features) == 0, "Visitation features appeared after step when disabled"


def test_visitation_counts_default_disabled(env_default):
    """Test that visitation counts are disabled by default."""
    obs, _ = env_default.reset(seed=42)

    features = extract_visitation_features(obs)
    assert len(features) == 0, "Visitation counts should be disabled by default"


def test_visitation_counts_configurable():
    """Test configuration variations for visitation counts."""

    simple_map = [
        [".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", "@", ".", ".", "."],  # Agent in center
        [".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "."],
    ]

    # Test enabled
    config_enabled = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs_width=5,
            obs_height=5,
            num_observation_tokens=100,
            inventory_item_names=["wood", "stone"],
            actions=ActionsConfig(move=ActionConfig()),
            objects={"wall": WallConfig(type_id=1)},
            groups={"agent": GroupConfig(id=0)},
            map_builder=AsciiMapBuilder.Config(map_data=simple_map),
            extensions=["visitation_counts"],
        )
    )
    env = MettaGridCore(config_enabled)
    obs, _ = env.reset(seed=42)
    count = count_visitation_features_at_center(obs)
    assert count == 5, f"Expected 5 features when enabled, got {count}"

    # Test disabled
    config_disabled = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs_width=5,
            obs_height=5,
            num_observation_tokens=100,
            inventory_item_names=["wood", "stone"],
            actions=ActionsConfig(move=ActionConfig()),
            objects={"wall": WallConfig(type_id=1)},
            groups={"agent": GroupConfig(id=0)},
            global_obs=GlobalObsConfig(),
            map_builder=AsciiMapBuilder.Config(map_data=simple_map),
        )
    )
    env = MettaGridCore(config_disabled)
    obs, _ = env.reset(seed=42)
    count = count_visitation_features_at_center(obs)
    assert count == 0, f"Expected 0 features when disabled, got {count}"

    # Test default (not specified)
    config_default = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs_width=5,
            obs_height=5,
            num_observation_tokens=100,
            inventory_item_names=["wood", "stone"],
            actions=ActionsConfig(move=ActionConfig()),
            objects={"wall": WallConfig(type_id=1)},
            groups={"agent": GroupConfig(id=0)},
            map_builder=AsciiMapBuilder.Config(map_data=simple_map),
        )
    )
    env = MettaGridCore(config_default)
    obs, _ = env.reset(seed=42)
    count = count_visitation_features_at_center(obs)
    assert count == 0, f"Expected 0 features by default, got {count}"


@pytest.fixture
def performance_config():
    """Config for performance testing with more steps."""

    simple_map = [
        [".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", "@", ".", ".", "."],  # Agent in center
        [".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "."],
    ]

    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=1000,
            obs_width=11,
            obs_height=11,
            num_observation_tokens=200,
            inventory_item_names=["wood", "stone"],
            actions=ActionsConfig(move=ActionConfig()),
            objects={"wall": WallConfig(type_id=1)},
            groups={"agent": GroupConfig(id=0)},
            map_builder=AsciiMapBuilder.Config(map_data=simple_map),
        )
    )
    return config


def test_agent_movement_tracking(env_with_visitation):
    """Test that agent position updates correctly."""
    env_with_visitation.reset(seed=42)

    initial_pos = get_agent_position(env_with_visitation)
    assert initial_pos == (3, 3), f"Expected agent at (3,3), got {initial_pos}"

    # Move and check position updates
    move_action = np.array([[0, 0]], dtype=dtype_actions)
    env_with_visitation.step(move_action)
    new_pos = get_agent_position(env_with_visitation)

    # Check that movement was successful
    action_success = env_with_visitation.action_success
    if action_success[0]:  # If move was successful
        assert new_pos != initial_pos, "Agent position should change after successful movement"


def test_observation_structure(env_with_visitation):
    """Test the structure of observations with visitation counts."""
    obs, _ = env_with_visitation.reset(seed=42)

    # Check observation shape
    assert len(obs.shape) == 3, f"Expected 3D observation array, got shape {obs.shape}"
    assert obs.shape[0] == 1, f"Expected 1 agent, got {obs.shape[0]}"

    # Find all feature IDs in use
    feature_ids = set()
    for i in range(obs.shape[1]):
        if obs[0, i, 0] != 0xFF:  # Not empty
            feature_ids.add(obs[0, i, 1])

    # Visitation count feature (ID 14) should be present
    assert 14 in feature_ids, f"Visitation count feature (14) not found. Found features: {sorted(feature_ids)}"

    # Count visitation features at different locations
    visitation_count = 0
    for i in range(obs.shape[1]):
        if obs[0, i, 1] == 14:
            visitation_count += 1

    assert visitation_count == 5, f"Expected 5 visitation count tokens, got {visitation_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
