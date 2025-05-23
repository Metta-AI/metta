import numpy as np

from mettagrid.mettagrid_c import MettaGrid  # pylint: disable=E0611
from mettagrid.tests.actions import move

NUM_AGENTS = 2
OBS_HEIGHT = 3
OBS_WIDTH = 3

# Rebuild the NumPy types using the exposed function
np_observations_type = np.dtype(MettaGrid.get_numpy_type_name("observations"))
np_terminals_type = np.dtype(MettaGrid.get_numpy_type_name("terminals"))
np_truncations_type = np.dtype(MettaGrid.get_numpy_type_name("truncations"))
np_rewards_type = np.dtype(MettaGrid.get_numpy_type_name("rewards"))
np_actions_type = np.dtype(MettaGrid.get_numpy_type_name("actions"))
np_masks_type = np.dtype(MettaGrid.get_numpy_type_name("masks"))
np_success_type = np.dtype(MettaGrid.get_numpy_type_name("success"))

# Map string directions to integer orientations
DIRECTION_MAP = {
    "north": 0,  # Up
    "south": 1,  # Down
    "west": 2,  # Left
    "east": 3,  # Right
}


def create_minimal_config():
    """Create a complete minimal config with all required action entries."""
    return {
        "max_steps": 5,
        "num_agents": 1,
        "obs_width": OBS_WIDTH,
        "obs_height": OBS_HEIGHT,
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": False},
            "rotate": {"enabled": False},
            "attack": {"enabled": False},
            "put_items": {"enabled": False},
            "get_items": {"enabled": False},
            "swap": {"enabled": False},
            "change_color": {"enabled": False},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1, "hp": 100},
            "altar": {"type_id": 2, "hp": 100},
        },
        "agent": {
            "inventory_size": 0,
            "hp": 100,
        },
    }


def minimal_reset_debug():
    """Minimal test to debug the reset segfault."""
    print("=== Minimal Reset Debug ===")

    # Try different map configurations to see if it's map-related
    print("Testing with minimal 3x3 map...")

    # Start with absolute minimal map
    game_map = [["wall", "wall", "wall"], ["wall", "agent.red", "wall"], ["wall", "wall", "wall"]]

    # Use complete minimal config
    env_config = {"game": create_minimal_config()}

    print("Creating environment...")
    env = MettaGrid(env_config, game_map)
    print("Environment created successfully")
    print(f"Num agents: {env.num_agents()}")
    print(f"Grid features: {env.grid_features()}")
    print(f"Map dimensions: {env.map_width()} x {env.map_height()}")

    print("Setting buffers...")
    # Set buffers before reset
    num_features = len(env.grid_features())
    observations = np.zeros((1, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np_observations_type)
    terminals = np.zeros(1, dtype=np_terminals_type)
    truncations = np.zeros(1, dtype=np_truncations_type)
    rewards = np.zeros(1, dtype=np_rewards_type)

    env.set_buffers(observations, terminals, truncations, rewards)

    print("Attempting reset...")
    try:
        obs, _info = env.reset()
        print("✅ Reset successful!")
        print(f"Observation shape: {obs.shape}")
        print(f"Observation dtype: {obs.dtype}")
        print(f"Observation sum: {obs.sum()}")
        return True
    except Exception as e:
        print(f"❌ Reset failed with exception: {e}")
        import traceback

        traceback.print_exc()
        raise AssertionError(f"Reset failed: {e}") from e


def test_different_map_sizes():
    """Test different map sizes to see if size is the issue."""
    sizes = [(3, 3), (4, 4), (5, 5)]

    for width, height in sizes:
        print(f"\n=== Testing {width}x{height} map ===")

        # Create map of specified size
        game_map = []
        for r in range(height):
            row = []
            for c in range(width):
                if r == 0 or r == height - 1 or c == 0 or c == width - 1:
                    row.append("wall")
                elif r == 1 and c == 1:
                    row.append("agent.red")
                else:
                    row.append("empty")
            game_map.append(row)

        # Use complete config
        config = create_minimal_config()
        env_config = {"game": config}

        try:
            env = MettaGrid(env_config, game_map)
            print(f"Environment created for {width}x{height}")

            # Set buffers before reset
            num_features = len(env.grid_features())
            observations = np.zeros((1, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np_observations_type)
            terminals = np.zeros(1, dtype=np_terminals_type)
            truncations = np.zeros(1, dtype=np_truncations_type)
            rewards = np.zeros(1, dtype=np_rewards_type)

            env.set_buffers(observations, terminals, truncations, rewards)

            obs, info = env.reset()
            print(f"✅ Reset successful for {width}x{height}")
        except Exception as e:
            print(f"❌ Failed for {width}x{height}: {e}")
            raise AssertionError(f"Failed for {width}x{height}: {e}") from e


def test_observation_sizes():
    """Test different observation sizes."""
    obs_sizes = [(1, 1), (3, 3), (5, 5)]

    for obs_h, obs_w in obs_sizes:
        print(f"\n=== Testing obs size {obs_h}x{obs_w} ===")

        game_map = [["wall", "wall", "wall"], ["wall", "agent.red", "wall"], ["wall", "wall", "wall"]]

        # Use complete config with custom obs size
        config = create_minimal_config()
        config["obs_width"] = obs_w
        config["obs_height"] = obs_h
        env_config = {"game": config}

        try:
            env = MettaGrid(env_config, game_map)
            print(f"Environment created with obs {obs_h}x{obs_w}")

            # Set buffers before reset
            num_features = len(env.grid_features())
            observations = np.zeros((1, obs_h, obs_w, num_features), dtype=np_observations_type)
            terminals = np.zeros(1, dtype=np_terminals_type)
            truncations = np.zeros(1, dtype=np_truncations_type)
            rewards = np.zeros(1, dtype=np_rewards_type)

            env.set_buffers(observations, terminals, truncations, rewards)

            obs, info = env.reset()
            print(f"✅ Reset successful with obs {obs_h}x{obs_w}")
            print(f"Returned obs shape: {obs.shape}")
        except Exception as e:
            print(f"❌ Failed with obs {obs_h}x{obs_w}: {e}")
            raise AssertionError(f"Failed with obs {obs_h}x{obs_w}: {e}") from e


def test_agent_positions():
    """Test different agent starting positions."""
    positions = [(1, 1), (1, 2), (2, 1)]

    for r, c in positions:
        print(f"\n=== Testing agent at ({r}, {c}) ===")

        # Create 4x4 map with agent at different positions
        game_map = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "empty", "empty", "wall"],
            ["wall", "empty", "empty", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]
        game_map[r][c] = "agent.red"

        # Use complete config with custom obs size
        config = create_minimal_config()
        config["obs_width"] = 3
        config["obs_height"] = 3
        env_config = {"game": config}

        try:
            env = MettaGrid(env_config, game_map)
            print(f"Environment created with agent at ({r}, {c})")

            # Set buffers before reset
            num_features = len(env.grid_features())
            observations = np.zeros((1, 3, 3, num_features), dtype=np_observations_type)
            terminals = np.zeros(1, dtype=np_terminals_type)
            truncations = np.zeros(1, dtype=np_truncations_type)
            rewards = np.zeros(1, dtype=np_rewards_type)

            env.set_buffers(observations, terminals, truncations, rewards)

            obs, info = env.reset()
            print(f"✅ Reset successful with agent at ({r}, {c})")
        except Exception as e:
            print(f"❌ Failed with agent at ({r}, {c}): {e}")
            raise AssertionError(f"Failed with agent at ({r}, {c}): {e}") from e


def create_test_env(max_steps=5):
    """Create a minimal environment for testing."""
    # Simple 5x5 map with walls around perimeter
    game_map = np.full((5, 5), "empty", dtype="<U50")
    game_map[0, :] = "wall"
    game_map[-1, :] = "wall"
    game_map[:, 0] = "wall"
    game_map[:, -1] = "wall"
    game_map[1, 1] = "agent.red"
    game_map[3, 3] = "agent.red"

    env_config = {
        "game": {
            "max_steps": max_steps,
            "num_agents": NUM_AGENTS,
            "obs_width": OBS_WIDTH,
            "obs_height": OBS_HEIGHT,
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": False},
                "attack": {"enabled": False},
                "put_items": {"enabled": False},
                "get_items": {"enabled": False},
                "swap": {"enabled": False},
                "change_color": {"enabled": False},
            },
            "groups": {"red": {"id": 0, "props": {}}},
            "objects": {
                "wall": {"type_id": 1, "hp": 100},
                "block": {"type_id": 2, "hp": 100},
            },
            "agent": {
                "inventory_size": 0,
                "hp": 100,
            },
        }
    }

    return MettaGrid(env_config, game_map.tolist())


def test_basic_import():
    """Test that we can import and access basic MettaGrid functionality."""
    print("Testing basic import...")

    # Test type names
    obs_type = MettaGrid.get_numpy_type_name("observations")
    print(f"Observations type: {obs_type}")

    assert obs_type is not None
    print("✓ Basic import test passed")


def test_environment_creation():
    """Test basic environment creation without any operations."""
    print("Testing environment creation...")

    env = create_test_env()
    print(f"Environment created with {env.num_agents()} agents")
    print(f"Action names: {env.action_names()}")

    assert env.num_agents() == NUM_AGENTS
    print("✓ Environment creation test passed")


def test_reset_only():
    """Test just the reset operation."""
    print("Testing reset only...")

    env = create_test_env()

    # Set buffers before reset
    num_features = len(env.grid_features())
    observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np_observations_type)
    terminals = np.zeros(NUM_AGENTS, dtype=np_terminals_type)
    truncations = np.zeros(NUM_AGENTS, dtype=np_truncations_type)
    rewards = np.zeros(NUM_AGENTS, dtype=np_rewards_type)

    env.set_buffers(observations, terminals, truncations, rewards)

    obs, info = env.reset()

    print(f"Reset returned observation shape: {obs.shape}")
    print(f"Reset returned observation dtype: {obs.dtype}")

    num_features = len(env.grid_features())
    expected_shape = (NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features)
    assert obs.shape == expected_shape
    print("✓ Reset only test passed")


def test_step_with_default_buffers():
    """Test step operation after setting buffers."""
    print("Testing step with buffers...")

    env = create_test_env(max_steps=2)

    # Set buffers before reset
    num_features = len(env.grid_features())
    observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np_observations_type)
    terminals = np.zeros(NUM_AGENTS, dtype=np_terminals_type)
    truncations = np.zeros(NUM_AGENTS, dtype=np_truncations_type)
    rewards = np.zeros(NUM_AGENTS, dtype=np_rewards_type)

    env.set_buffers(observations, terminals, truncations, rewards)

    obs, info = env.reset()
    print("Reset completed")

    # Create actions
    noop_idx = env.action_names().index("noop")
    actions = np.full((NUM_AGENTS, 2), [noop_idx, 0], dtype=np_actions_type)
    print(f"Actions created: {actions}")

    # Try one step
    print("Attempting step...")
    obs, rewards, terminals, truncations, info = env.step(actions)
    print("Step completed successfully")

    print(f"Rewards: {rewards}")
    print(f"Terminals: {terminals}")
    print(f"Truncations: {truncations}")
    print("✓ Step with buffers test passed")


def test_set_buffers_basic():
    """Test basic set_buffers functionality."""
    print("Testing basic set_buffers...")

    env = create_test_env()
    num_features = len(env.grid_features())

    # Create buffers
    observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np_observations_type)
    terminals = np.zeros(NUM_AGENTS, dtype=np_terminals_type)
    truncations = np.zeros(NUM_AGENTS, dtype=np_truncations_type)
    rewards = np.zeros(NUM_AGENTS, dtype=np_rewards_type)

    print("Buffers created")
    print(f"Observations shape: {observations.shape}, dtype: {observations.dtype}")
    print(f"Terminals shape: {terminals.shape}, dtype: {terminals.dtype}")
    print(f"Truncations shape: {truncations.shape}, dtype: {truncations.dtype}")
    print(f"Rewards shape: {rewards.shape}, dtype: {rewards.dtype}")

    # Set buffers
    print("Calling set_buffers...")
    env.set_buffers(observations, terminals, truncations, rewards)
    print("set_buffers completed")

    print("✓ Set buffers basic test passed")


def test_reset_after_set_buffers():
    """Test reset after setting buffers."""
    print("Testing reset after set_buffers...")

    env = create_test_env()
    num_features = len(env.grid_features())

    # Create and set buffers
    observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np_observations_type)
    terminals = np.zeros(NUM_AGENTS, dtype=np_terminals_type)
    truncations = np.zeros(NUM_AGENTS, dtype=np_truncations_type)
    rewards = np.zeros(NUM_AGENTS, dtype=np_rewards_type)

    env.set_buffers(observations, terminals, truncations, rewards)
    print("Buffers set")

    # Reset
    print("Calling reset...")
    obs_from_env, info = env.reset()
    print("Reset completed")

    print(f"Memory sharing: {np.shares_memory(obs_from_env, observations)}")
    print("✓ Reset after set_buffers test passed")


def test_step_after_set_buffers():
    """Test step after setting buffers."""
    print("Testing step after set_buffers...")

    env = create_test_env(max_steps=2)
    num_features = len(env.grid_features())

    # Create and set buffers
    observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np_observations_type)
    terminals = np.zeros(NUM_AGENTS, dtype=np_terminals_type)
    truncations = np.zeros(NUM_AGENTS, dtype=np_truncations_type)
    rewards = np.zeros(NUM_AGENTS, dtype=np_rewards_type)

    env.set_buffers(observations, terminals, truncations, rewards)
    obs, info = env.reset()
    print("Reset completed")

    # Create actions and step
    noop_idx = env.action_names().index("noop")
    actions = np.full((NUM_AGENTS, 2), [noop_idx, 0], dtype=np_actions_type)

    print("Attempting step...")
    obs_step, rewards_step, terminals_step, truncations_step, info = env.step(actions)
    print("Step completed")

    print(f"Memory sharing - obs: {np.shares_memory(obs_step, observations)}")
    print(f"Memory sharing - rewards: {np.shares_memory(rewards_step, rewards)}")
    print("✓ Step after set_buffers test passed")


def test_agent_walks_across_room():
    """
    Test where a single agent walks across a room.
    Creates a simple corridor and attempts to walk the agent from one end to the other.
    The move() function already handles observation validation.
    """
    print("Testing agent walking across room...")

    # Create a simple 7x4 corridor map
    game_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "altar", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    config = create_minimal_config()
    config["actions"]["move"] = {"enabled": True}
    config["actions"]["rotate"] = {"enabled": True}
    config["num_agents"] = 1
    config["obs_width"] = 3
    config["obs_height"] = 3
    config["max_steps"] = 20
    env_config = {"game": config}

    # Create environment
    env = MettaGrid(env_config, game_map)
    num_features = len(env.grid_features())
    observations = np.zeros((1, 3, 3, num_features), dtype=np_observations_type)
    terminals = np.zeros(1, dtype=np_terminals_type)
    truncations = np.zeros(1, dtype=np_truncations_type)
    rewards = np.zeros(1, dtype=np_rewards_type)
    env.set_buffers(observations, terminals, truncations, rewards)
    env.reset()

    print(f"Environment created: {env.map_width()}x{env.map_height()}")
    print(f"Initial timestep: {env.current_timestep}")

    # Find a working direction
    successful_moves = []
    total_moves = 0

    print("\n=== Testing which direction allows movement ===")
    working_direction = None

    for direction_str in ["east", "west", "north", "south"]:
        orientation = DIRECTION_MAP[direction_str]
        print(f"\nTesting movement {direction_str}...")

        result = move(env, orientation, agent_idx=0)

        if result["success"]:
            print(f"✓ Found working direction: {direction_str}")
            working_direction = direction_str
            break
        else:
            print(f"✗ Direction {direction_str} failed: {result.get('error', 'Unknown error')}")

    if working_direction is None:
        print("❌ No direction successfully moved the agent")
        return

    print(f"\n=== Walking across room in direction: {working_direction} ===")

    # Reset for clean walk
    env = MettaGrid(env_config, game_map)
    env.set_buffers(observations, terminals, truncations, rewards)
    env.reset()

    # Walk multiple steps
    working_orientation = DIRECTION_MAP[working_direction]
    max_steps = 5

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step}: Moving {working_direction} ---")

        result = move(env, working_orientation, agent_idx=0)
        total_moves += 1

        if result["success"]:
            successful_moves.append(step)
            print(f"✓ Successful move #{len(successful_moves)}")
            print(f"  Position: {result['position_before']} → {result['position_after']}")
        else:
            print(f"✗ Move failed: {result.get('error', 'Unknown error')}")
            if not result["move_success"]:
                print("  Agent likely hit an obstacle or boundary")
                break

        if env.current_timestep >= config["max_steps"] - 2:
            print("  Approaching max steps limit")
            break

    print("\n=== Walking Test Summary ===")
    print(f"Working direction: {working_direction}")
    print(f"Total move attempts: {total_moves}")
    print(f"Successful moves: {len(successful_moves)}")
    print(f"Success rate: {len(successful_moves) / total_moves:.1%}" if total_moves > 0 else "N/A")

    # Validation
    if len(successful_moves) >= 1:
        print("✅ Agent walking test passed!")
    else:
        print("❌ No successful moves - test failed")

    assert len(successful_moves) >= 1, (
        f"Agent should have moved at least once. Got {len(successful_moves)} successful moves."
    )


if __name__ == "__main__":
    test_agent_walks_across_room()
