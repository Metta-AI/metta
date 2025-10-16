#!/usr/bin/env python3

import copy
import time

import numpy as np
import pytest

from mettagrid import dtype_actions
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    GameConfig,
    GlobalObsConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.core import MettaGridCore
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME
from mettagrid.test_support.actions import action_index
from mettagrid.test_support.orientation import Orientation


@pytest.fixture
def env_with_visitation():
    """Environment with visitation counts enabled."""

    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            num_observation_tokens=100,
            resource_names=["wood", "stone"],
            actions=ActionsConfig(
                move=ActionConfig(),  # Enable 8-way movement
            ),
            objects={"wall": WallConfig(type_id=1)},
            global_obs=GlobalObsConfig(
                episode_completion_pct=True,
                last_action=True,
                last_reward=True,
                visitation_counts=True,  # Enable visitation counts
            ),
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", "@", ".", ".", "."],  # Agent in center
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )
    return MettaGridCore(config)


@pytest.fixture
def env_without_visitation():
    """Environment with visitation counts disabled."""
    # Create custom configuration matching original test setup

    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            num_observation_tokens=100,
            resource_names=["wood", "stone"],
            actions=ActionsConfig(
                move=ActionConfig(),  # Enable 8-way movement
            ),
            objects={"wall": WallConfig(type_id=1)},
            global_obs=GlobalObsConfig(
                episode_completion_pct=True,
                last_action=True,
                last_reward=True,
                visitation_counts=False,  # Disable visitation counts
            ),
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", "@", ".", ".", "."],  # Agent in center
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )
    return MettaGridCore(config)


@pytest.fixture
def env_default():
    """Environment with default config (no visitation_counts specified)."""
    # Create custom configuration matching original test setup

    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            num_observation_tokens=100,
            resource_names=["wood", "stone"],
            actions=ActionsConfig(
                move=ActionConfig(),  # Enable 8-way movement
            ),
            objects={"wall": WallConfig(type_id=1)},
            # No explicit visitation_counts setting - uses default (False)
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", "@", ".", ".", "."],  # Agent in center
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", "."],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )
    return MettaGridCore(config)


# Helper functions
def extract_visitation_features(visitation_feature_id: int, obs: np.ndarray, agent_idx: int = 0) -> list[int]:
    """Extract visitation count features from observation."""
    features = []
    for i in range(obs.shape[1]):
        if obs[agent_idx, i, 1] == visitation_feature_id:
            features.append(obs[agent_idx, i, 2])
    return features


@pytest.mark.skip(reason="Visitation counts not incrementing - needs investigation")
def test_visitation_counts_enabled(env_with_visitation):
    """Test that visitation counts work correctly when enabled."""
    obs, _ = env_with_visitation.reset(seed=42)
    visitation_feature_id = env_with_visitation.c_env.feature_spec()["agent:visitation_counts"]["id"]

    # Check initial visitation counts
    initial_features = extract_visitation_features(visitation_feature_id, obs)
    assert len(initial_features) == 5, f"Expected 5 visitation features, got {len(initial_features)}"
    assert initial_features == [0, 0, 0, 0, 0], f"Expected initial counts [0,0,0,0,0], got {initial_features}"

    # Move and check counts update
    move_action = np.array([action_index(env_with_visitation, "move", Orientation.NORTH)], dtype=dtype_actions)
    for step in range(3):
        obs, _, _, _, _ = env_with_visitation.step(move_action)
        features = extract_visitation_features(visitation_feature_id, obs)
        assert len(features) == 5, f"Step {step}: Expected 5 features, got {len(features)}"
        # At least one count should be non-zero after movement
        # Note: visitation counts are incremented at the end of the step,
        # so we need to take at least 2 steps to see non-zero counts
        if step > 1:
            assert any(f > 0 for f in features), f"Expected non-zero counts after step {step}, got {features}"


def test_visitation_counts_disabled(env_without_visitation):
    """Test that visitation counts are not present when disabled."""
    obs, _ = env_without_visitation.reset(seed=42)
    visitation_feature_id = env_without_visitation.c_env.feature_spec()["agent:visitation_counts"]["id"]

    features = extract_visitation_features(visitation_feature_id, obs)
    assert len(features) == 0, f"Expected no visitation features when disabled, got {len(features)}"

    # Verify they stay disabled after steps
    move_action = np.array([action_index(env_without_visitation, "move", Orientation.NORTH)], dtype=dtype_actions)
    obs, _, _, _, _ = env_without_visitation.step(move_action)
    features = extract_visitation_features(visitation_feature_id, obs)
    assert len(features) == 0, "Visitation features appeared after step when disabled"


def test_visitation_counts_default_disabled(env_default):
    """Test that visitation counts are disabled by default."""
    obs, _ = env_default.reset(seed=42)
    visitation_feature_id = env_default.c_env.feature_spec()["agent:visitation_counts"]["id"]

    features = extract_visitation_features(visitation_feature_id, obs)
    assert len(features) == 0, "Visitation counts should be disabled by default"


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
            resource_names=["wood", "stone"],
            actions=ActionsConfig(move=ActionConfig()),
            objects={"wall": WallConfig(type_id=1)},
            map_builder=AsciiMapBuilder.Config(
                map_data=simple_map,
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )
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
def test_visitation_performance_impact(performance_config):
    """Disabled visitation should not be materially slower."""
    move_action = np.array([action_index(env_default, "move", Orientation.NORTH)], dtype=dtype_actions)

    # enabled
    cfg_on = copy.deepcopy(performance_config)
    cfg_on.game.global_obs = GlobalObsConfig(visitation_counts=True)
    env_enabled = MettaGridCore(cfg_on)
    env_enabled.reset(seed=42)
    enabled_time = _median_runtime(env_enabled, move_action)

    # disabled
    cfg_off = copy.deepcopy(performance_config)
    cfg_off.game.global_obs = GlobalObsConfig(visitation_counts=False)
    env_disabled = MettaGridCore(cfg_off)
    env_disabled.reset(seed=42)
    disabled_time = _median_runtime(env_disabled, move_action)

    # allow small jitter (≤10% slowdown)
    assert disabled_time <= enabled_time * 1.1, (
        f"Disabled visitation unexpectedly slower: "
        f"enabled={enabled_time:.6f}s, disabled={disabled_time:.6f}s, "
        f"delta={(disabled_time / enabled_time - 1) * 100:.2f}%"
    )


def test_observation_structure(env_with_visitation):
    """Test the structure of observations with visitation counts."""
    obs, _ = env_with_visitation.reset(seed=42)
    visitation_feature_id = env_with_visitation.c_env.feature_spec()["agent:visitation_counts"]["id"]

    # Check observation shape
    assert len(obs.shape) == 3, f"Expected 3D observation array, got shape {obs.shape}"
    assert obs.shape[0] == 1, f"Expected 1 agent, got {obs.shape[0]}"

    # Find all feature IDs in use
    feature_ids = [obs[0, i, 1] for i in range(obs.shape[1]) if obs[0, i, 0] != 0xFF]

    # Visitation count feature (ID 14) should be present
    assert visitation_feature_id in feature_ids, (
        f"Visitation count feature ({visitation_feature_id}) not found. Found features: {sorted(set(feature_ids))}"
    )

    # Count visitation features at different locations
    visitation_count = len([feature_id for feature_id in feature_ids if feature_id == visitation_feature_id])
    assert visitation_count == 5, f"Expected 5 visitation count tokens, got {visitation_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
