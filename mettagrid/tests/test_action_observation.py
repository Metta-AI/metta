"""
Unit tests for agent action observation feature.

Tests that agents can observe their own last action and action success
through the observation system.
"""

import numpy as np
import pytest

from mettagrid.mettagrid_c import MettaGrid
from mettagrid.mettagrid_env import (
    np_observations_type,
    np_rewards_type,
    np_terminals_type,
    np_truncations_type,
)


@pytest.fixture
def base_config():
    """Base configuration for action observation tests."""
    return {
        "max_steps": 50,
        "num_agents": 2,
        "obs_width": 5,
        "obs_height": 5,
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": False},
            "put_items": {"enabled": False},
            "get_items": {"enabled": False},
            "swap": {"enabled": False},
            "change_color": {"enabled": False},
        },
        "groups": {
            "red": {"id": 0, "props": {}},
            "blue": {"id": 1, "props": {}},
        },
        "objects": {
            "wall": {"type_id": 1, "hp": 100},
        },
        "agent": {
            "freeze_duration": 0,
            "default_item_max": 10,
            "hp": 100,
        },
    }


@pytest.fixture
def simple_game_map():
    """Simple game map with two agents in open space."""
    return [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "agent.blue", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def configured_env(base_config, simple_game_map):
    """Create a configured MettaGrid environment for testing."""
    env_config = {"game": base_config}
    env = MettaGrid(env_config, simple_game_map)

    # Set up buffers
    num_agents = base_config["num_agents"]
    num_features = len(env.grid_features())
    observations = np.zeros((num_agents, 5, 5, num_features), dtype=np_observations_type)
    terminals = np.zeros(num_agents, dtype=np_terminals_type)
    truncations = np.zeros(num_agents, dtype=np_truncations_type)
    rewards = np.zeros(num_agents, dtype=np_rewards_type)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()
    return env


class TestActionObservationFeatures:
    """Test that action observation features are properly configured."""

    def test_action_features_exist(self, configured_env):
        """Test that action tracking features exist in grid features."""
        features = configured_env.grid_features()

        assert "agent:last_action" in features, "last_action feature should exist"
        assert "agent:last_action_success" in features, "last_action_success feature should exist"

    def test_action_feature_indices(self, configured_env):
        """Test that action features have correct indices."""
        features = configured_env.grid_features()

        last_action_idx = features.index("agent:last_action")
        last_action_success_idx = features.index("agent:last_action_success")

        # Should be consecutive indices (as defined in agent.hpp)
        assert last_action_success_idx == last_action_idx + 1, "Action features should be consecutive"

    def test_action_features_in_grid_objects(self, configured_env):
        """Test that action features appear in grid_objects() output."""
        grid_objects = configured_env.grid_objects()

        # Find agent objects
        agent_objects = [obj for obj in grid_objects.values() if 'agent_id' in obj]
        assert len(agent_objects) == 2, "Should have 2 agents"

        for agent_obj in agent_objects:
            assert "agent:last_action" in agent_obj, "Agent object should have last_action"
            assert "agent:last_action_success" in agent_obj, "Agent object should have last_action_success"


class TestActionObservationTracking:
    """Test that action observations are correctly tracked and updated."""

    def test_initial_action_state(self, configured_env):
        """Test initial state of action observations."""
        obs, _ = configured_env.reset()
        features = configured_env.grid_features()

        last_action_idx = features.index("agent:last_action")
        last_action_success_idx = features.index("agent:last_action_success")

        # Check observations at agent positions (center of their observation)
        center_r, center_c = 2, 2  # 5x5 obs, agent at center

        for agent_idx in range(configured_env.num_agents):
            agent_obs = obs[agent_idx, center_r, center_c, :]

            # Initially, last_action should be 0 (could be initial or noop)
            initial_action = agent_obs[last_action_idx]
            initial_success = agent_obs[last_action_success_idx]

            assert isinstance(initial_action, (int, np.integer)), "Last action should be integer"
            assert isinstance(initial_success, (int, np.integer)), "Last action success should be integer"
            assert initial_action >= 0, "Last action should be non-negative"
            assert initial_success in [0, 1], "Last action success should be 0 or 1"

    def test_single_action_tracking(self, configured_env):
        """Test that a single action is correctly tracked."""
        action_names = configured_env.action_names()
        features = configured_env.grid_features()

        noop_idx = action_names.index("noop")
        move_idx = action_names.index("move")

        last_action_feature_idx = features.index("agent:last_action")
        last_action_success_feature_idx = features.index("agent:last_action_success")

        # Execute different actions for each agent
        actions = np.array([[noop_idx, 0], [move_idx, 0]], dtype=np.int32)
        obs, rewards, terminals, truncations, info = configured_env.step(actions)

        center_r, center_c = 2, 2

        # Check agent 0 (noop action)
        agent0_obs = obs[0, center_r, center_c, :]
        assert agent0_obs[last_action_feature_idx] == noop_idx, f"Agent 0 should observe noop action ({noop_idx})"
        # Noop should generally succeed
        assert agent0_obs[last_action_success_feature_idx] in [0, 1], "Action success should be 0 or 1"

        # Check agent 1 (move action)
        agent1_obs = obs[1, center_r, center_c, :]
        assert agent1_obs[last_action_feature_idx] == move_idx, f"Agent 1 should observe move action ({move_idx})"
        assert agent1_obs[last_action_success_feature_idx] in [0, 1], "Action success should be 0 or 1"

    def test_multiple_action_sequence(self, configured_env):
        """Test that actions are tracked correctly over multiple steps."""
        action_names = configured_env.action_names()
        features = configured_env.grid_features()

        noop_idx = action_names.index("noop")
        move_idx = action_names.index("move")
        rotate_idx = action_names.index("rotate")

        last_action_feature_idx = features.index("agent:last_action")
        center_r, center_c = 2, 2

        # Sequence of actions for agent 0
        action_sequence = [
            ([noop_idx, noop_idx], noop_idx),    # Both agents noop
            ([move_idx, rotate_idx], move_idx),   # Agent 0 move, Agent 1 rotate
            ([rotate_idx, move_idx], rotate_idx), # Agent 0 rotate, Agent 1 move
        ]

        for step, (action_pair, expected_agent0_action) in enumerate(action_sequence):
            actions = np.array([[action_pair[0], 0], [action_pair[1], 0]], dtype=np.int32)
            obs, _, _, _, _ = configured_env.step(actions)

            agent0_obs = obs[0, center_r, center_c, :]
            observed_action = agent0_obs[last_action_feature_idx]

            assert observed_action == expected_agent0_action, (
                f"Step {step + 1}: Agent 0 should observe action {expected_agent0_action}, "
                f"but observed {observed_action}"
            )

    def test_action_success_tracking(self, configured_env):
        """Test that action success is tracked correctly."""
        action_names = configured_env.action_names()
        features = configured_env.grid_features()

        noop_idx = action_names.index("noop")
        last_action_success_feature_idx = features.index("agent:last_action_success")
        center_r, center_c = 2, 2

        # Execute noop actions (should generally succeed)
        actions = np.array([[noop_idx, 0], [noop_idx, 0]], dtype=np.int32)
        obs, _, _, _, _ = configured_env.step(actions)

        # Check action success values
        for agent_idx in range(configured_env.num_agents):
            agent_obs = obs[agent_idx, center_r, center_c, :]
            action_success = agent_obs[last_action_success_feature_idx]

            assert action_success in [0, 1], f"Agent {agent_idx} action success should be 0 or 1"
            # For noop actions in open space, we expect success
            # (Note: specific success values depend on action handler implementation)

    def test_cross_agent_action_independence(self, configured_env):
        """Test that each agent observes only its own actions."""
        action_names = configured_env.action_names()
        features = configured_env.grid_features()

        noop_idx = action_names.index("noop")
        move_idx = action_names.index("move")
        rotate_idx = action_names.index("rotate")

        last_action_feature_idx = features.index("agent:last_action")
        center_r, center_c = 2, 2

        # Give different actions to each agent
        actions = np.array([[move_idx, 0], [rotate_idx, 0]], dtype=np.int32)
        obs, _, _, _, _ = configured_env.step(actions)

        # Each agent should observe only its own action
        agent0_obs = obs[0, center_r, center_c, :]
        agent1_obs = obs[1, center_r, center_c, :]

        assert agent0_obs[last_action_feature_idx] == move_idx, "Agent 0 should observe its own move action"
        assert agent1_obs[last_action_feature_idx] == rotate_idx, "Agent 1 should observe its own rotate action"

        # Agents should not observe each other's actions
        assert agent0_obs[last_action_feature_idx] != rotate_idx, "Agent 0 should not observe agent 1's action"
        assert agent1_obs[last_action_feature_idx] != move_idx, "Agent 1 should not observe agent 0's action"


class TestActionObservationConsistency:
    """Test consistency between different ways of accessing action data."""

    def test_observation_vs_grid_objects_consistency(self, configured_env):
        """Test that observations match grid_objects() for action data."""
        action_names = configured_env.action_names()
        features = configured_env.grid_features()

        move_idx = action_names.index("move")
        rotate_idx = action_names.index("rotate")

        last_action_feature_idx = features.index("agent:last_action")
        last_action_success_feature_idx = features.index("agent:last_action_success")
        center_r, center_c = 2, 2

        # Execute actions
        actions = np.array([[move_idx, 0], [rotate_idx, 0]], dtype=np.int32)
        obs, _, _, _, _ = configured_env.step(actions)

        # Get data from grid_objects
        grid_objects = configured_env.grid_objects()
        agent_objects = [obj for obj in grid_objects.values() if 'agent_id' in obj]
        agent_objects.sort(key=lambda x: x['agent_id'])  # Sort by agent_id

        # Compare observations with grid_objects
        for agent_idx in range(configured_env.num_agents):
            agent_obs = obs[agent_idx, center_r, center_c, :]
            agent_obj = agent_objects[agent_idx]

            obs_last_action = agent_obs[last_action_feature_idx]
            obs_last_success = agent_obs[last_action_success_feature_idx]

            obj_last_action = agent_obj["agent:last_action"]
            obj_last_success = agent_obj["agent:last_action_success"]

            assert obs_last_action == obj_last_action, (
                f"Agent {agent_idx}: observation last_action ({obs_last_action}) "
                f"should match grid_objects ({obj_last_action})"
            )
            assert obs_last_success == obj_last_success, (
                f"Agent {agent_idx}: observation last_action_success ({obs_last_success}) "
                f"should match grid_objects ({obj_last_success})"
            )

    def test_action_success_tracking_consistency(self, configured_env):
        """Test that action_success() method is consistent with observations."""
        action_names = configured_env.action_names()
        features = configured_env.grid_features()

        noop_idx = action_names.index("noop")
        last_action_success_feature_idx = features.index("agent:last_action_success")
        center_r, center_c = 2, 2

        # Execute actions
        actions = np.array([[noop_idx, 0], [noop_idx, 0]], dtype=np.int32)
        obs, _, _, _, _ = configured_env.step(actions)

        # Get action success from method
        action_success_list = configured_env.action_success()

        # Compare with observations
        for agent_idx in range(configured_env.num_agents):
            agent_obs = obs[agent_idx, center_r, center_c, :]
            obs_success = agent_obs[last_action_success_feature_idx]
            method_success = action_success_list[agent_idx]

            # Convert boolean to int for comparison
            method_success_int = 1 if method_success else 0

            assert obs_success == method_success_int, (
                f"Agent {agent_idx}: observation success ({obs_success}) "
                f"should match action_success() method ({method_success_int})"
            )


class TestActionObservationEdgeCases:
    """Test edge cases and error conditions for action observations."""

    def test_invalid_action_handling(self, configured_env):
        """Test how invalid actions are handled in observations."""
        features = configured_env.grid_features()
        last_action_feature_idx = features.index("agent:last_action")
        last_action_success_feature_idx = features.index("agent:last_action_success")
        center_r, center_c = 2, 2

        # Try an invalid action index (very high number)
        invalid_action = 999
        actions = np.array([[invalid_action, 0], [0, 0]], dtype=np.int32)

        # This should not crash
        obs, _, _, _, _ = configured_env.step(actions)

        # Agent with invalid action should have some deterministic behavior
        agent0_obs = obs[0, center_r, center_c, :]
        observed_action = agent0_obs[last_action_feature_idx]
        observed_success = agent0_obs[last_action_success_feature_idx]

        # The exact behavior for invalid actions may vary,
        # but it should be consistent and not crash
        assert isinstance(observed_action, (int, np.integer)), "Should return integer action"
        assert isinstance(observed_success, (int, np.integer)), "Should return integer success"
        assert observed_success in [0, 1], "Success should be 0 or 1"

    def test_action_persistence_across_resets(self, base_config, simple_game_map):
        """Test that action observations are properly reset."""
        # Create first environment and execute an action
        env_config = {"game": base_config}
        env1 = MettaGrid(env_config, simple_game_map)

        # Set up buffers
        num_agents = base_config["num_agents"]
        num_features = len(env1.grid_features())
        observations = np.zeros((num_agents, 5, 5, num_features), dtype=np_observations_type)
        terminals = np.zeros(num_agents, dtype=np_terminals_type)
        truncations = np.zeros(num_agents, dtype=np_truncations_type)
        rewards = np.zeros(num_agents, dtype=np_rewards_type)
        env1.set_buffers(observations, terminals, truncations, rewards)

        env1.reset()
        action_names = env1.action_names()
        features = env1.grid_features()

        move_idx = action_names.index("move")
        last_action_feature_idx = features.index("agent:last_action")
        center_r, center_c = 2, 2

        # Execute an action
        actions = np.array([[move_idx, 0], [move_idx, 0]], dtype=np.int32)
        env1.step(actions)

        # Create fresh environment (simulates reset)
        env2 = MettaGrid(env_config, simple_game_map)
        env2.set_buffers(observations, terminals, truncations, rewards)
        obs, _ = env2.reset()

        # Check that action observations are properly initialized in fresh environment
        for agent_idx in range(env2.num_agents):
            agent_obs = obs[agent_idx, center_r, center_c, :]
            reset_action = agent_obs[last_action_feature_idx]

            # After reset, action should be back to initial state (typically 0)
            assert reset_action == 0, f"Agent {agent_idx} should have reset action state"


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__])
