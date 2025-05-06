import numpy as np
import pytest
from omegaconf import OmegaConf

# Import both implementations
from mettagrid.env import MettaGrid  # New C++ implementation
from mettagrid.old.env import MettaGrid as MettaGridOld  # Original implementation


# Sample configuration and map data for testing
@pytest.fixture
def env_config():
    """Create a simple environment configuration for testing."""
    config_dict = {
        "game": {
            "num_agents": 2,
            "max_steps": 100,
            "obs_width": 7,
            "obs_height": 7,
            "objects": {
                "wall": {"health": 10},
                "block": {"health": 5},
                "mine.red": {"cooldown": 5, "output_item": "resource"},
                "generator.red": {"cooldown": 5, "output_item": "energy"},
                "altar": {"recipes": {"resource": {"output": "gem"}}},
                "armory": {"recipes": {"resource": {"output": "sword"}}},
                "lasery": {"recipes": {"resource": {"output": "laser"}}},
                "lab": {"recipes": {"resource": {"output": "potion"}}},
                "factory": {"recipes": {"resource": {"output": "robot"}}},
                "temple": {"recipes": {"gem": {"output": "blessing"}}},
            },
            "agent": {"health": 100, "inventory_capacity": 10},
            "groups": {
                "red": {"id": 0, "group_reward_pct": 0.2, "props": {"color": "red"}},
                "blue": {"id": 1, "group_reward_pct": 0.2, "props": {"color": "blue"}},
            },
            "actions": {
                "move": {"enabled": True, "priority": 1},
                "rotate": {"enabled": True, "priority": 1},
                "put_items": {"enabled": True, "priority": 2},
                "get_items": {"enabled": True, "priority": 2},
                "attack": {"enabled": True, "priority": 3},
                "noop": {"enabled": True, "priority": 0},
                "swap": {"enabled": True, "priority": 1},
                "change_color": {"enabled": True, "priority": 1},
            },
        }
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def simple_map():
    """Create a simple map for testing."""
    # Create a 10x10 grid with walls on the edges, some objects, and agents
    map_data = np.full((10, 10), "empty", dtype=object)

    # Add walls around the edges
    map_data[0, :] = "wall"
    map_data[-1, :] = "wall"
    map_data[:, 0] = "wall"
    map_data[:, -1] = "wall"

    # Add some objects
    map_data[2, 2] = "mine.red"
    map_data[2, 7] = "generator.red"
    map_data[7, 2] = "altar"
    map_data[7, 7] = "factory"

    # Add agents
    map_data[3, 3] = "agent.red"
    map_data[6, 6] = "agent.blue"

    return map_data


@pytest.mark.parametrize(
    "action_sequence",
    [
        # Test with no actions
        np.zeros((2, 2), dtype=np.int32),
        # Test with simple movement actions
        np.array([[0, 1], [0, 3]], dtype=np.int32),  # Move agent 0 up, agent 1 left
        # Test with rotation actions
        np.array([[1, 1], [1, 3]], dtype=np.int32),  # Rotate agent 0 clockwise, agent 1 counterclockwise
        # Test with more complex action sequence
        np.array([[0, 2], [3, 1]], dtype=np.int32),  # Move agent 0 right, get_items for agent 1
    ],
)
def test_observation_consistency(env_config, simple_map, action_sequence):
    """Test that observations are consistent between old and new implementations."""

    # Initialize both implementations with the same config and map
    old_env = MettaGridOld(env_config, simple_map)
    new_env = MettaGrid(env_config, simple_map)

    # Reset both environments
    old_obs, old_info = old_env.reset()
    new_obs, new_info = new_env.reset()

    # Compare initial observations
    np.testing.assert_array_equal(old_obs, new_obs, err_msg="Initial observations differ between implementations")

    # Take a step with the provided actions
    old_result = old_env.step(action_sequence)
    new_result = new_env.step(action_sequence)

    # Unpack results
    old_obs, old_rewards, old_terminals, old_truncs, old_info = old_result
    new_obs, new_rewards, new_terminals, new_truncs, new_info = new_result

    # Compare observations after step
    np.testing.assert_array_equal(
        old_obs, new_obs, err_msg=f"Observations differ after action sequence {action_sequence}"
    )

    # Compare rewards
    np.testing.assert_array_almost_equal(
        old_rewards, new_rewards, decimal=5, err_msg=f"Rewards differ after action sequence {action_sequence}"
    )

    # Compare terminals and truncations
    np.testing.assert_array_equal(
        old_terminals, new_terminals, err_msg=f"Terminal flags differ after action sequence {action_sequence}"
    )
    np.testing.assert_array_equal(
        old_truncs, new_truncs, err_msg=f"Truncation flags differ after action sequence {action_sequence}"
    )


def test_multi_step_consistency(env_config, simple_map):
    """Test consistency over multiple steps with random actions."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Initialize both implementations
    old_env = MettaGridOld(env_config, simple_map)
    new_env = MettaGrid(env_config, simple_map)

    # Reset both environments
    old_obs, _ = old_env.reset()
    new_obs, _ = new_env.reset()

    # Run for 10 steps with random actions
    for step in range(10):
        # Generate random actions
        num_agents = old_env.num_agents()
        action_types = len(old_env.action_names())
        max_args = np.max(old_env.max_action_args())

        actions = np.zeros((num_agents, 2), dtype=np.int32)
        actions[:, 0] = np.random.randint(0, action_types, size=num_agents)
        actions[:, 1] = np.random.randint(0, max_args + 1, size=num_agents)

        # Take steps in both environments
        old_result = old_env.step(actions)
        new_result = new_env.step(actions)

        # Unpack results
        old_obs, old_rewards, old_terminals, old_truncs, old_info = old_result
        new_obs, new_rewards, new_terminals, new_truncs, new_info = new_result

        # Compare results
        np.testing.assert_array_equal(
            old_obs, new_obs, err_msg=f"Observations differ at step {step} with actions {actions}"
        )
        np.testing.assert_array_almost_equal(
            old_rewards, new_rewards, decimal=5, err_msg=f"Rewards differ at step {step} with actions {actions}"
        )
        np.testing.assert_array_equal(
            old_terminals, new_terminals, err_msg=f"Terminal flags differ at step {step} with actions {actions}"
        )
        np.testing.assert_array_equal(
            old_truncs, new_truncs, err_msg=f"Truncation flags differ at step {step} with actions {actions}"
        )


def test_observation_shapes(env_config, simple_map):
    """Test that observation shapes are consistent between implementations."""
    # Initialize both implementations
    old_env = MettaGridOld(env_config, simple_map)
    new_env = MettaGrid(env_config, simple_map)

    # Reset both environments
    old_obs, _ = old_env.reset()
    new_obs, _ = new_env.reset()

    # Check observation shapes
    assert old_obs.shape == new_obs.shape, "Observation shapes differ between implementations"

    # Check specific dimensions
    num_agents = old_env.num_agents()
    obs_height = env_config.game.obs_height
    obs_width = env_config.game.obs_width

    # Get the number of features from both implementations
    old_features = len(old_env.grid_features())
    new_features = len(new_env.grid_features())

    assert old_features == new_features, "Number of features differs between implementations"
    expected_shape = (num_agents, old_features, obs_height, obs_width)
    assert old_obs.shape == expected_shape, f"Old observation shape {old_obs.shape} != expected {expected_shape}"
    assert new_obs.shape == expected_shape, f"New observation shape {new_obs.shape} != expected {expected_shape}"


def test_custom_observation(env_config, simple_map):
    """Test custom observation from a specific location."""
    # Initialize both implementations
    old_env = MettaGridOld(env_config, simple_map)
    new_env = MettaGrid(env_config, simple_map)

    # Reset both environments
    old_env.reset()
    new_env.reset()

    # Create observation buffers
    obs_width = 5
    obs_height = 5
    num_features = len(old_env.grid_features())

    old_obs_buffer = np.zeros((obs_height, obs_width, num_features), dtype=np.uint8)
    new_obs_buffer = np.zeros((obs_height, obs_width, num_features), dtype=np.uint8)

    # Get observations from a specific location
    row, col = 5, 5
    old_env.observe_at(row, col, obs_width, obs_height, old_obs_buffer)
    new_env.observe_at(row, col, obs_width, obs_height, new_obs_buffer)

    # Compare observations
    np.testing.assert_array_equal(
        old_obs_buffer, new_obs_buffer, err_msg=f"Custom observations differ at location ({row}, {col})"
    )


def test_edge_cases(env_config, simple_map):
    """Test edge cases for observation consistency."""
    # Initialize both implementations
    old_env = MettaGridOld(env_config, simple_map)
    new_env = MettaGrid(env_config, simple_map)

    # Reset both environments
    old_env.reset()
    new_env.reset()

    # Test observation at map edge
    edge_row, edge_col = 0, 0
    obs_width = 3
    obs_height = 3
    num_features = len(old_env.grid_features())

    old_edge_obs = np.zeros((obs_height, obs_width, num_features), dtype=np.uint8)
    new_edge_obs = np.zeros((obs_height, obs_width, num_features), dtype=np.uint8)

    old_env.observe_at(edge_row, edge_col, obs_width, obs_height, old_edge_obs)
    new_env.observe_at(edge_row, edge_col, obs_width, obs_height, new_edge_obs)

    np.testing.assert_array_equal(
        old_edge_obs, new_edge_obs, err_msg=f"Edge case observations differ at map edge ({edge_row}, {edge_col})"
    )

    # Test observation at map corner with larger observation window than map
    corner_row, corner_col = 9, 9
    large_obs_width = 15
    large_obs_height = 15

    old_corner_obs = np.zeros((large_obs_height, large_obs_width, num_features), dtype=np.uint8)
    new_corner_obs = np.zeros((large_obs_height, large_obs_width, num_features), dtype=np.uint8)

    old_env.observe_at(corner_row, corner_col, large_obs_width, large_obs_height, old_corner_obs)
    new_env.observe_at(corner_row, corner_col, large_obs_width, large_obs_height, new_corner_obs)

    np.testing.assert_array_equal(
        old_corner_obs, new_corner_obs, err_msg="Edge case observations differ at map corner with large window"
    )
