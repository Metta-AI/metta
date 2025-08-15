#!/usr/bin/env python3

import copy
import time
from typing import Optional

import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


@pytest.fixture
def simple_map():
    """Provide the simple 7x7 map with agent in center."""
    return [
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "agent.test_group", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
    ]


@pytest.fixture
def base_config():
    """Provide the base game configuration."""
    return {
        "max_steps": 100,
        "num_agents": 1,
        "episode_truncates": True,
        "obs_width": 5,
        "obs_height": 5,
        "inventory_item_names": ["wood", "stone"],
        "num_observation_tokens": 100,
        "actions": {
            "move": {"enabled": True},
        },
        "objects": {"wall": {"type_id": 1}},
        "agent": {},
        "groups": {"test_group": {"id": 0, "props": {}}},
        "global_obs": {
            "episode_completion_pct": True,
            "last_action": True,
            "last_reward": True,
            "resource_rewards": False,
        },
    }


@pytest.fixture
def env_with_visitation(base_config, simple_map):
    """Environment with visitation counts enabled."""
    config = copy.deepcopy(base_config)
    config["global_obs"]["visitation_counts"] = True
    return MettaGrid(from_mettagrid_config(config), simple_map, 42)


@pytest.fixture
def env_without_visitation(base_config, simple_map):
    """Environment with visitation counts disabled."""
    config = copy.deepcopy(base_config)
    config["global_obs"]["visitation_counts"] = False
    return MettaGrid(from_mettagrid_config(config), simple_map, 42)


@pytest.fixture
def env_default(base_config, simple_map):
    """Environment with default config (no visitation_counts specified)."""
    return MettaGrid(from_mettagrid_config(base_config), simple_map, 42)


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


def get_agent_position(env: MettaGrid) -> Optional[tuple[int, int]]:
    """Get the current agent position."""
    grid_objects = env.grid_objects()
    for obj in grid_objects.values():
        if "agent_id" in obj:
            return (obj["r"], obj["c"])
    return None


def test_visitation_counts_enabled(env_with_visitation):
    """Test that visitation counts work correctly when enabled."""
    obs, _ = env_with_visitation.reset()

    # Check initial visitation counts
    initial_features = extract_visitation_features(obs)
    assert len(initial_features) == 5, f"Expected 5 visitation features, got {len(initial_features)}"
    assert initial_features == [0, 0, 0, 0, 0], f"Expected initial counts [0,0,0,0,0], got {initial_features}"

    # Move and check counts update
    move_action = np.array([[0, 0]], dtype=np.int32)
    for step in range(3):
        obs, _, _, _, _ = env_with_visitation.step(move_action)
        features = extract_visitation_features(obs)
        assert len(features) == 5, f"Step {step}: Expected 5 features, got {len(features)}"
        # At least one count should be non-zero after movement
        if step > 0:
            assert any(f > 0 for f in features), f"Expected non-zero counts after step {step}"


def test_visitation_counts_disabled(env_without_visitation):
    """Test that visitation counts are not present when disabled."""
    obs, _ = env_without_visitation.reset()

    features = extract_visitation_features(obs)
    assert len(features) == 0, f"Expected no visitation features when disabled, got {len(features)}"

    # Verify they stay disabled after steps
    move_action = np.array([[0, 0]], dtype=np.int32)
    obs, _, _, _, _ = env_without_visitation.step(move_action)
    features = extract_visitation_features(obs)
    assert len(features) == 0, "Visitation features appeared after step when disabled"


def test_visitation_counts_default_disabled(env_default):
    """Test that visitation counts are disabled by default."""
    obs, _ = env_default.reset()

    features = extract_visitation_features(obs)
    assert len(features) == 0, "Visitation counts should be disabled by default"


def test_visitation_counts_configurable(base_config, simple_map):
    """Test configuration variations for visitation counts."""
    # Test enabled
    config_enabled = copy.deepcopy(base_config)
    config_enabled["global_obs"]["visitation_counts"] = True
    env = MettaGrid(from_mettagrid_config(config_enabled), simple_map, 42)
    obs, _ = env.reset()
    count = count_visitation_features_at_center(obs)
    assert count == 5, f"Expected 5 features when enabled, got {count}"

    # Test disabled
    config_disabled = copy.deepcopy(base_config)
    config_disabled["global_obs"]["visitation_counts"] = False
    env = MettaGrid(from_mettagrid_config(config_disabled), simple_map, 42)
    obs, _ = env.reset()
    count = count_visitation_features_at_center(obs)
    assert count == 0, f"Expected 0 features when disabled, got {count}"

    # Test default (not specified)
    env = MettaGrid(from_mettagrid_config(base_config), simple_map, 42)
    obs, _ = env.reset()
    count = count_visitation_features_at_center(obs)
    assert count == 0, f"Expected 0 features by default, got {count}"


@pytest.fixture
def performance_config(base_config):
    """Config for performance testing with more steps."""
    config = copy.deepcopy(base_config)
    config["max_steps"] = 1000
    config["obs_width"] = 11
    config["obs_height"] = 11
    config["num_observation_tokens"] = 200
    return config


def _median_runtime(env, move_action, warmup_steps=10, test_steps=200, reps=15):
    # warmup
    for _ in range(warmup_steps):
        env.step(move_action)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(test_steps):
            env.step(move_action)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


@pytest.mark.skip(reason="Flaky performance test - timing variations on different systems")
def test_visitation_performance_impact(performance_config, simple_map):
    """Disabled visitation should not be materially slower."""
    move_action = np.array([[0, 0]], dtype=np.int32)

    # enabled
    cfg_on = copy.deepcopy(performance_config)
    cfg_on["global_obs"]["visitation_counts"] = True
    env_enabled = MettaGrid(from_mettagrid_config(cfg_on), simple_map, 42)
    env_enabled.reset()
    enabled_time = _median_runtime(env_enabled, move_action)

    # disabled
    cfg_off = copy.deepcopy(performance_config)
    cfg_off["global_obs"]["visitation_counts"] = False
    env_disabled = MettaGrid(from_mettagrid_config(cfg_off), simple_map, 42)
    env_disabled.reset()
    disabled_time = _median_runtime(env_disabled, move_action)

    # allow small jitter (≤10% slowdown)
    assert disabled_time <= enabled_time * 1.1, (
        f"Disabled visitation unexpectedly slower: "
        f"enabled={enabled_time:.6f}s, disabled={disabled_time:.6f}s, "
        f"delta={(disabled_time / enabled_time - 1) * 100:.2f}%"
    )


def test_agent_movement_tracking(env_with_visitation):
    """Test that agent position updates correctly."""
    env_with_visitation.reset()

    initial_pos = get_agent_position(env_with_visitation)
    assert initial_pos == (3, 3), f"Expected agent at (3,3), got {initial_pos}"

    # Move and check position updates
    move_action = np.array([[0, 0]], dtype=np.int32)
    env_with_visitation.step(move_action)
    new_pos = get_agent_position(env_with_visitation)

    # Check that movement was successful
    action_success = env_with_visitation.action_success()
    if action_success[0]:  # If move was successful
        assert new_pos != initial_pos, "Agent position should change after successful movement"


def test_observation_structure(env_with_visitation):
    """Test the structure of observations with visitation counts."""
    obs, _ = env_with_visitation.reset()

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
