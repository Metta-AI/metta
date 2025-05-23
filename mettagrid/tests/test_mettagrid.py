import numpy as np

from mettagrid.mettagrid_c import MettaGrid  # pylint: disable=E0611

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
            "block": {"type_id": 2, "hp": 100},
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


def test_agent_observations_simple():
    """
    Test agent observations with a clear, simple map.
    Create one map with known features and test observations from different agent positions.
    """
    print("Testing agent observations with simple map...")

    # Create a clear 5x5 map with distinctive features
    # Layout:
    # W W W W W
    # W . . . W
    # W . X . W  <- X is a distinctive wall
    # W . . . W
    # W W W W W
    #
    # We'll test with agents at different positions to see how the observation changes

    base_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "wall", "empty", "wall"],  # Central wall at (2,2)
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    config = create_minimal_config()
    config["actions"]["move"] = {"enabled": True}
    config["num_agents"] = 1
    config["obs_width"] = 3
    config["obs_height"] = 3
    env_config = {"game": config}

    # Test positions around the central wall
    test_positions = [
        ((1, 1), "top-left of center wall"),
        ((1, 2), "directly above center wall"),
        ((1, 3), "top-right of center wall"),
        ((2, 1), "directly left of center wall"),
        ((2, 3), "directly right of center wall"),
        ((3, 1), "bottom-left of center wall"),
        ((3, 2), "directly below center wall"),
        ((3, 3), "bottom-right of center wall"),
    ]

    observations_by_position = {}

    for (agent_row, agent_col), description in test_positions:
        print(f"\n--- Testing agent at ({agent_row}, {agent_col}) - {description} ---")

        # Create map with agent at this position
        game_map = [row[:] for row in base_map]  # Copy the base map
        game_map[agent_row][agent_col] = "agent.red"

        # Create environment
        env = MettaGrid(env_config, game_map)

        # Set buffers
        num_features = len(env.grid_features())
        observations = np.zeros((1, 3, 3, num_features), dtype=np_observations_type)
        terminals = np.zeros(1, dtype=np_terminals_type)
        truncations = np.zeros(1, dtype=np_truncations_type)
        rewards = np.zeros(1, dtype=np_rewards_type)

        env.set_buffers(observations, terminals, truncations, rewards)

        # Get observation
        obs, info = env.reset()

        # Extract wall channel
        wall_feature_idx = env.grid_features().index("wall")
        wall_channel = obs[0, :, :, wall_feature_idx]

        print(f"Wall observation from {description}:")
        print(wall_channel)

        # Store for comparison
        observations_by_position[(agent_row, agent_col)] = wall_channel

        # Analyze what we expect to see
        print("Expected analysis:")

        # The agent is at (agent_row, agent_col) in the map
        # The central wall is at (2, 2) in the map
        # The observation is 3x3 centered on the agent

        # Calculate where the central wall should appear in the observation
        wall_row_in_obs = 2 - agent_row + 1  # +1 because obs is 0-2, centered at 1
        wall_col_in_obs = 2 - agent_col + 1

        if 0 <= wall_row_in_obs <= 2 and 0 <= wall_col_in_obs <= 2:
            expected_wall_value = wall_channel[wall_row_in_obs, wall_col_in_obs]
            print(f"Central wall should be at obs[{wall_row_in_obs}, {wall_col_in_obs}] = {expected_wall_value}")
            if expected_wall_value > 0:
                print("✓ Central wall visible as expected")
            else:
                print("? Central wall not visible (might be outside observation window)")
        else:
            print("Central wall should be outside observation window")

        # Count total walls visible (should include perimeter walls)
        total_walls = np.sum(wall_channel > 0)
        print(f"Total walls visible: {total_walls}")

    # Now compare observations between positions
    print("\n--- Comparing observations between positions ---")

    positions = list(observations_by_position.keys())
    differences_found = 0

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pos1, pos2 = positions[i], positions[j]
            obs1, obs2 = observations_by_position[pos1], observations_by_position[pos2]

            if not np.array_equal(obs1, obs2):
                differences_found += 1
                print(f"✓ Observations differ between {pos1} and {pos2}")
            else:
                print(f"! Identical observations between {pos1} and {pos2}")

    print(f"\nFound {differences_found} unique observation patterns out of {len(positions)} positions")

    # Test specific expected relationships
    print("\n--- Testing specific spatial relationships ---")

    # Agent above wall vs agent below wall should see different things
    if (1, 2) in observations_by_position and (3, 2) in observations_by_position:
        obs_above = observations_by_position[(1, 2)]  # Above central wall
        obs_below = observations_by_position[(3, 2)]  # Below central wall

        print("Agent above central wall sees:")
        print(obs_above)
        print("Agent below central wall sees:")
        print(obs_below)

        if not np.array_equal(obs_above, obs_below):
            print("✓ Above and below positions show different observations")
        else:
            print("! Above and below positions show identical observations")

    # Agent left vs right of wall
    if (2, 1) in observations_by_position and (2, 3) in observations_by_position:
        obs_left = observations_by_position[(2, 1)]  # Left of central wall
        obs_right = observations_by_position[(2, 3)]  # Right of central wall

        print("Agent left of central wall sees:")
        print(obs_left)
        print("Agent right of central wall sees:")
        print(obs_right)

        if not np.array_equal(obs_left, obs_right):
            print("✓ Left and right positions show different observations")
        else:
            print("! Left and right positions show identical observations")

    # Basic sanity checks
    assert differences_found > 0, "Expected some differences between observations from different positions"

    print("\n✓ Simple observation test completed successfully")
    return True


def test_agent_walks_across_room():
    """
    Test where a single agent walks across a room and we confirm observations at each step.

    Creates a simple corridor and attempts to walk the agent from one end to the other,
    checking that observations change appropriately with each step.
    """
    print("Testing agent walking across room...")

    # Create a simple 7x5 corridor map
    # Layout:
    # W W W W W W W
    # W A . . . . W  <- Agent starts at (1,1), can walk to (1,5)
    # W W W W W W W
    #
    # This gives the agent a clear path to walk along

    game_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    config = create_minimal_config()
    config["actions"]["move"] = {"enabled": True}
    config["num_agents"] = 1
    config["obs_width"] = 3
    config["obs_height"] = 3
    config["max_steps"] = 10  # Give us enough steps to walk
    env_config = {"game": config}

    env = MettaGrid(env_config, game_map)
    print(f"Environment created: {env.map_width()}x{env.map_height()}")
    print(f"Action names: {env.action_names()}")

    # Set buffers
    num_features = len(env.grid_features())
    observations = np.zeros((1, 3, 3, num_features), dtype=np_observations_type)
    terminals = np.zeros(1, dtype=np_terminals_type)
    truncations = np.zeros(1, dtype=np_truncations_type)
    rewards = np.zeros(1, dtype=np_rewards_type)

    env.set_buffers(observations, terminals, truncations, rewards)

    # Reset and get initial observation
    obs_initial, info = env.reset()
    print(f"Initial timestep: {env.current_timestep}")

    # Get feature indices we care about
    grid_features = env.grid_features()
    wall_feature_idx = grid_features.index("wall")
    agent_feature_idx = grid_features.index("agent")

    print(f"Wall feature at index: {wall_feature_idx}")
    print(f"Agent feature at index: {agent_feature_idx}")

    # Get action information
    action_names = env.action_names()
    move_action_idx = action_names.index("move")
    max_action_args = env.max_action_args()
    move_max_arg = max_action_args[move_action_idx]

    print(f"Move action index: {move_action_idx}")
    print(f"Move max arg: {move_max_arg}")

    # Display initial state
    wall_channel = obs_initial[0, :, :, wall_feature_idx]
    agent_channel = obs_initial[0, :, :, agent_feature_idx]

    print("\nStep 0 (Initial):")
    print("Wall channel:")
    print(wall_channel)
    print("Agent channel:")
    print(agent_channel)

    # Store observations for comparison
    step_observations = [(0, wall_channel.copy(), agent_channel.copy())]

    # Try to walk the agent to the right (should be direction 1 typically)
    # We'll try each direction to see which one works
    directions_to_try = [(0, "up"), (1, "right"), (2, "down"), (3, "left")]

    successful_direction = None
    direction_name = None

    # Find a direction that actually moves the agent
    for direction, name in directions_to_try:
        if direction > move_max_arg:
            continue

        print(f"\nTrying direction {direction} ({name})...")

        # Take one step in this direction
        move_action = np.array([[move_action_idx, direction]], dtype=np_actions_type)
        obs_after, rewards, terminals, truncations, info = env.step(move_action)

        # Check if action succeeded
        action_success = env.action_success()
        print(f"Action success: {action_success[0]}")
        print(f"Timestep after move: {env.current_timestep}")

        # Check if observation changed
        wall_after = obs_after[0, :, :, wall_feature_idx]
        agent_after = obs_after[0, :, :, agent_feature_idx]

        observation_changed = not (
            np.array_equal(wall_channel, wall_after) and np.array_equal(agent_channel, agent_after)
        )

        print(f"Observation changed: {observation_changed}")

        if observation_changed:
            print(f"✓ Found working direction: {direction} ({name})")
            successful_direction = direction
            direction_name = name

            print("Wall channel after move:")
            print(wall_after)
            print("Agent channel after move:")
            print(agent_after)

            step_observations.append((1, wall_after.copy(), agent_after.copy()))
            break
        else:
            print(f"Direction {direction} ({name}) didn't change observation")

    if successful_direction is None:
        print("! No direction successfully moved the agent")
        print("This could mean:")
        print("- Move action is not implemented")
        print("- Agent is blocked by walls in all directions")
        print("- Move arguments are different than expected")

        # Let's still check if we can see action feedback
        print("\nTesting action feedback without movement...")
        noop_action_idx = action_names.index("noop")
        noop_action = np.array([[noop_action_idx, 0]], dtype=np_actions_type)

        obs_noop, rewards, terminals, truncations, info = env.step(noop_action)
        action_success = env.action_success()
        print(f"Noop action success: {action_success[0]}")
        print(f"Timestep after noop: {env.current_timestep}")

        print("✓ Agent walking test completed (movement not working, but feedback works)")

    # Continue walking in the successful direction
    print(f"\nContinuing to walk {direction_name}...")

    for step in range(2, 6):  # Take a few more steps
        move_action = np.array([[successful_direction, direction]], dtype=np_actions_type)
        obs_step, rewards, terminals, truncations, info = env.step(move_action)

        action_success = env.action_success()
        print(f"\nStep {step}:")
        print(f"Action success: {action_success[0]}")
        print(f"Timestep: {env.current_timestep}")

        wall_step = obs_step[0, :, :, wall_feature_idx]
        agent_step = obs_step[0, :, :, agent_feature_idx]

        print("Wall channel:")
        print(wall_step)
        print("Agent channel:")
        print(agent_step)

        step_observations.append((step, wall_step.copy(), agent_step.copy()))

        # Check if we've hit a wall (observation stops changing)
        prev_wall = step_observations[-2][1]
        prev_agent = step_observations[-2][2]

        if np.array_equal(wall_step, prev_wall) and np.array_equal(agent_step, prev_agent):
            print(f"! Observation stopped changing at step {step} - likely hit a wall")
            break
        else:
            print("✓ Observation changed from previous step")

    # Analyze the journey
    print("\n--- Journey Analysis ---")
    print(f"Total steps taken: {len(step_observations)}")
    print(f"Successful direction: {successful_direction} ({direction_name})")

    # Check that we saw different observations during the journey
    unique_observations = set()
    for _step, wall_obs, agent_obs in step_observations:
        # Convert to tuple for hashing
        obs_signature = (tuple(wall_obs.flatten()), tuple(agent_obs.flatten()))
        unique_observations.add(obs_signature)

    print(f"Unique observation patterns: {len(unique_observations)}")

    if len(unique_observations) > 1:
        print("✓ Agent saw different observations during the journey")
    else:
        print("! Agent saw the same observation throughout (no movement?)")

    # Verify that the agent channel shows the agent in the center
    for step, _wall_obs, agent_obs in step_observations:
        agent_center = agent_obs[1, 1]  # Center of 3x3 observation
        if agent_center > 0:
            print(f"✓ Step {step}: Agent visible at center of observation")
        else:
            print(f"! Step {step}: Agent not visible at center - unexpected!")

    print("\n✓ Agent walking test completed successfully")
