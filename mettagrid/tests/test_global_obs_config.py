"""Test global observation configuration functionality."""

from metta.mettagrid.mettagrid_c import MettaGrid, PackedCoordinate
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


def create_test_env(global_obs_config):
    """Create test environment with specified global_obs configuration."""
    game_config = {
        "num_agents": 2,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 100,
        "max_steps": 100,
        "inventory_item_names": ["item1", "item2"],
        "global_obs": global_obs_config,
        "agent": {
            "default_resource_limit": 10,
            "freeze_duration": 0,
            "rewards": {},
            "action_failure_penalty": 0,
        },
        "groups": {"agent": {"id": 0, "sprite": 0, "props": {}}},
        "actions": {"noop": {"enabled": True}, "move": {"enabled": True}},
        "objects": {"wall": {"type_id": 1, "swappable": False}},
    }

    game_map = [
        ["wall", "wall", "wall", "wall"],
        ["wall", "agent.agent", "agent.agent", "wall"],
        ["wall", "wall", "wall", "wall"],
    ]

    env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)
    return env


def count_global_features(obs, expected_feature_ids):
    """Count specific global feature tokens in observations."""
    center_location = PackedCoordinate.pack(5, 5)  # 11x11 obs, center is (5,5)

    total_count = 0
    num_agents = obs.shape[0]
    for agent_idx in range(num_agents):
        found_features = set()
        for token_idx in range(obs.shape[1]):
            if obs[agent_idx, token_idx, 0] == center_location:
                feature_id = obs[agent_idx, token_idx, 1]
                if feature_id in expected_feature_ids:
                    found_features.add(feature_id)
        total_count += len(found_features)

    return total_count


def test_all_global_tokens_enabled():
    """Test that all global tokens are present when enabled."""
    global_obs = {"episode_completion_pct": True, "last_action": True, "last_reward": True}

    env = create_test_env(global_obs)
    obs, _ = env.reset()

    # Feature IDs from constants.hpp:
    # EpisodeCompletionPct = 8, LastAction = 9, LastActionArg = 10, LastReward = 11
    expected_features = {8, 9, 10, 11}
    global_token_count = count_global_features(obs, expected_features)

    # Each agent should have 4 global tokens
    assert global_token_count == 8  # 2 agents * 4 tokens


def test_episode_completion_disabled():
    """Test that episode completion token is not present when disabled."""
    global_obs = {"episode_completion_pct": False, "last_action": True, "last_reward": True}

    env = create_test_env(global_obs)
    obs, _ = env.reset()

    # Should have last_action, last_action_arg, last_reward
    expected_features = {9, 10, 11}
    global_token_count = count_global_features(obs, expected_features)

    # Each agent should have 3 global tokens
    assert global_token_count == 6  # 2 agents * 3 tokens


def test_last_action_disabled():
    """Test that last action tokens are not present when disabled."""
    global_obs = {"episode_completion_pct": True, "last_action": False, "last_reward": True}

    env = create_test_env(global_obs)
    obs, _ = env.reset()

    # Should have episode_pct and last_reward
    expected_features = {8, 11}
    global_token_count = count_global_features(obs, expected_features)

    # Each agent should have 2 global tokens
    assert global_token_count == 4  # 2 agents * 2 tokens


def test_all_global_tokens_disabled():
    """Test that no global tokens are present when all disabled."""
    global_obs = {"episode_completion_pct": False, "last_action": False, "last_reward": False}

    env = create_test_env(global_obs)
    obs, _ = env.reset()

    # Should have no global tokens
    expected_features = {8, 9, 10, 11}
    global_token_count = count_global_features(obs, expected_features)

    # No global tokens should be present
    assert global_token_count == 0


def test_global_obs_default_values():
    """Test that global_obs uses default values when not specified."""
    # Test with no global_obs specified - should use defaults (all True)
    game_config = {
        "num_agents": 1,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 100,
        "max_steps": 100,
        "inventory_item_names": ["item1"],
        # No global_obs specified - should use defaults
        "agent": {
            "default_resource_limit": 10,
            "freeze_duration": 0,
            "rewards": {},
            "action_failure_penalty": 0,
        },
        "groups": {"agent": {"id": 0, "sprite": 0, "props": {}}},
        "actions": {"noop": {"enabled": True}},
        "objects": {"wall": {"type_id": 1, "swappable": False}},
    }

    game_map = [["agent.agent"]]

    # This should work without error, using default global_obs values
    env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)
    obs, _ = env.reset()

    # Should have all 4 global tokens by default
    expected_features = {8, 9, 10, 11}
    global_token_count = count_global_features(obs, expected_features)

    assert global_token_count == 4
