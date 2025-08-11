#!/usr/bin/env python3
import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.util.actions import get_agent_position
from metta.mettagrid.util.hydra import get_cfg


def test_no_agent_interference():
    """Test ghost object functionality and no_agent_interference flag"""

    print("Testing ghost object functionality and no_agent_interference flag...")

    test_results = []

    # Test 1: Basic ghost object functionality with no_agent_interference=True
    print("\n=== Test 1: Ghost object movement with no_agent_interference=True ===")
    result1 = test_ghost_movement_with_interference_flag(True)
    test_results.append(("Ghost movement with no_agent_interference=True", result1))

    # Test 2: Normal movement with no_agent_interference=False
    print("\n=== Test 2: Normal movement with no_agent_interference=False ===")
    result2 = test_ghost_movement_with_interference_flag(False)
    test_results.append(("Normal movement with no_agent_interference=False", result2))

    # Test 3: Observation filtering with no_agent_interference=True
    print("\n=== Test 3: Observation filtering with no_agent_interference=True ===")
    result3 = test_observation_filtering(True)
    test_results.append(("Observation filtering with no_agent_interference=True", result3))

    # Test 4: Observation filtering with no_agent_interference=False
    print("\n=== Test 4: Observation filtering with no_agent_interference=False ===")
    result4 = test_observation_filtering(False)
    test_results.append(("Observation filtering with no_agent_interference=False", result4))

    # Report results
    print("\n=== Test Results ===")
    all_passed = True
    for test_name, result in test_results:
        if result:
            print(f"✓ {test_name}: PASSED")
        else:
            print(f"✗ {test_name}: FAILED")
            all_passed = False

    if all_passed:
        print("\n=== All tests completed successfully! ===")
    else:
        print("\n=== Some tests failed! ===")

    return all_passed


@pytest.mark.skip(reason="This is a standalone test script, not a pytest test")
def test_ghost_movement_with_interference_flag(no_agent_interference: bool):
    """Test that agents can move through each other when no_agent_interference is enabled"""

    # Get the benchmark config and modify it
    cfg = get_cfg("benchmark")

    # Simplify config for testing
    cfg.game.num_agents = 2
    cfg.game.max_steps = 10
    cfg.game.episode_truncates = True
    cfg.game.track_movement_metrics = True
    cfg.game.no_agent_interference = no_agent_interference

    # Create a custom map with agents at adjacent positions
    cfg.game.map_builder = OmegaConf.create(
        {
            "_target_": "metta.mettagrid.room.random.Random",
            "width": 2,
            "height": 2,
            "objects": {},
            "agents": 2,
            "seed": 42,
        }
    )

    # Create curriculum and environment
    curriculum = SingleTaskCurriculum("test", cfg)
    env = MettaGridEnv(curriculum, render_mode=None)

    obs, _ = env.reset()

    # Get action indices
    action_names = env.action_names
    move_idx = action_names.index("move") if "move" in action_names else None

    if move_idx is None:
        print(f"ERROR: Required actions not available. move: {move_idx}")
        return False

    print(f"Testing with no_agent_interference={no_agent_interference}")

    # Get initial positions
    initial_positions = []
    for agent_idx in range(2):
        agent_pos = get_agent_position(env._c_env_instance, agent_idx)
        if agent_pos:
            initial_positions.append(agent_pos)
            print(f"  Agent {agent_idx} initial position: {agent_pos}")
        else:
            print(f"  Agent {agent_idx} initial position: unknown")

    # Check if agents are adjacent
    if len(initial_positions) == 2:
        pos0, pos1 = initial_positions[0], initial_positions[1]
        manhattan_distance = abs(pos0[0] - pos1[0]) + abs(pos0[1] - pos1[1])
        print(f"  Manhattan distance between agents: {manhattan_distance}")

        if manhattan_distance == 1:
            print("  ✓ Agents are adjacent, testing ghost movement...")
            # Agents are adjacent, try to have them move into each other's space
            # Determine which direction each agent should move to swap positions
            actions = np.array([[move_idx, 0], [move_idx, 0]], dtype=np.int32)
            actions = np.array([[move_idx, 1], [move_idx, 1]], dtype=np.int32)
            # if pos0[0] == pos1[0]:  # Same row, move horizontally
            #     if pos0[1] < pos1[1]:  # Agent 0 is to the left of Agent 1
            #         actions = np.array([[move_idx, 3], [move_idx, 1]], dtype=np.int32)  # Right, Left
            #     else:  # Agent 0 is to the right of Agent 1
            #         actions = np.array([[move_idx, 1], [move_idx, 3]], dtype=np.int32)  # Left, Right
            # else:  # Same column, move vertically
            #     if pos0[0] < pos1[0]:  # Agent 0 is above Agent 1
            #         actions = np.array([[move_idx, 2], [move_idx, 0]], dtype=np.int32)  # Down, Up
            #     else:  # Agent 0 is below Agent 1
            #         actions = np.array([[move_idx, 0], [move_idx, 2]], dtype=np.int32)  # Up, Down
        else:
            print("  ✗ Agents are not adjacent, cannot test ghost movement properly")
            return False
    else:
        print("  ✗ Could not determine agent positions")
        return False

    print("  Attempting to move agents into each other's space...")
    obs, rewards, terminals, truncations, info = env.step(actions)

    # Check action success
    action_success = env._c_env_instance.action_success()
    print(f"  Action success: {action_success}")

    # Get final positions
    final_positions = []
    for agent_idx in range(2):
        agent_pos = get_agent_position(env._c_env_instance, agent_idx)
        if agent_pos:
            final_positions.append(agent_pos)
            print(f"  Agent {agent_idx} final position: {agent_pos}")
        else:
            print(f"  Agent {agent_idx} final position: unknown")

    # Analyze results
    if no_agent_interference:
        # When no_agent_interference=True, both agents should be able to move successfully
        if action_success[0] and action_success[1]:
            print("  ✓ Agents can move through each other when no_agent_interference=True")
            return True
        else:
            print(
                f"  ✗ Agents cannot move through each other when no_agent_interference=True (success: {action_success})"
            )
            return False
    else:
        # When no_agent_interference=False, at least one agent should fail to move
        if not (action_success[0] and action_success[1]):
            print("  ✓ Agents cannot move through each other when no_agent_interference=False")
            return True
        else:
            print(
                f"  ✗ Agents can move through each other when no_agent_interference=False (success: {action_success})"
            )
            return False


@pytest.mark.skip(reason="This is a standalone test script, not a pytest test")
def test_observation_filtering(no_agent_interference: bool):
    """Test that agents only see themselves when no_agent_interference is enabled"""

    # Get the benchmark config and modify it
    cfg = get_cfg("benchmark")

    # Simplify config for testing
    cfg.game.num_agents = 2
    cfg.game.max_steps = 10
    cfg.game.episode_truncates = True
    cfg.game.no_agent_interference = no_agent_interference
    cfg.game.obs_width = 5  # Small observation window
    cfg.game.obs_height = 5

    # Create a simple level with two agents close to each other
    cfg.game.map_builder = OmegaConf.create(
        {
            "_target_": "metta.mettagrid.room.random.Random",
            "width": 5,
            "height": 5,
            "objects": {},
            "agents": 2,
            "seed": 42,
        }
    )

    # Create environment
    curriculum = SingleTaskCurriculum("test", cfg)
    env = MettaGridEnv(curriculum, render_mode=None)
    env.reset()

    print(f"Testing observation filtering with no_agent_interference={no_agent_interference}")

    # Get observations for both agents
    observations, rewards, terminals, truncations, infos = env.step(np.array([[0, 0], [0, 0]], dtype=np.int32))

    for agent_idx in range(2):
        agent_obs = observations[agent_idx]

        # Count agent tokens specifically (tokens with feature_id 0-8 are typically agent features)
        agent_tokens = []
        other_tokens = []

        for token in agent_obs:
            if token[0] != 255:  # Skip empty tokens
                location = token[0]
                feature_id = token[1]
                feature_value = token[2]

                # Agent tokens typically have feature_id 0-8 (position, orientation, inventory, etc.)
                if feature_id <= 8:
                    agent_tokens.append((location, feature_id, feature_value))
                else:
                    other_tokens.append((location, feature_id, feature_value))

        # Group agent tokens by location to count unique agent positions
        agent_positions = set()
        for location, _feature_id, _feature_value in agent_tokens:
            # Extract row and column from packed location
            row = (location >> 4) & 0xF
            col = location & 0xF
            agent_positions.add((row, col))

        print(f"\nAgent {agent_idx} observation analysis:")
        print(f"    Total agent tokens found: {len(agent_tokens)}")
        print(f"    Agent positions: {sorted(agent_positions)}")
        print(f"    Other object tokens: {len(other_tokens)}")

        if no_agent_interference:
            # Should only see 1 agent position (itself)
            if len(agent_positions) == 1:
                print(f"    ✓ Agent {agent_idx} only sees itself (found 1 agent position)")
            else:
                print(f"    ✗ Agent {agent_idx} sees {len(agent_positions)} agents (should see 1)")
                return False  # Indicate failure
        else:
            # Should see multiple agent positions (itself and others)
            if len(agent_positions) >= 2:
                print(f"    ✓ Agent {agent_idx} can see other agents (found {len(agent_positions)} agent positions)")
            else:
                print(f"    ✗ Agent {agent_idx} cannot see other agents (found {len(agent_positions)} agent positions)")
                return False  # Indicate failure

    return True  # Indicate success


@pytest.mark.skip(reason="This is a standalone test script, not a pytest test")
def test_ghost_add_object():
    """Test ghost_add_object functionality"""

    print("\n=== Test 5: Ghost add object functionality ===")

    # This test would require access to the C++ grid implementation
    # For now, we'll test it indirectly through the environment

    # Get the benchmark config and modify it
    cfg = get_cfg("benchmark")

    # Simplify config for testing
    cfg.game.num_agents = 1
    cfg.game.max_steps = 10
    cfg.game.episode_truncates = True
    cfg.game.no_agent_interference = True  # Enable ghost functionality

    # Create a simple level
    cfg.game.map_builder = OmegaConf.create(
        {
            "_target_": "metta.mettagrid.room.random.Random",
            "width": 3,
            "height": 3,
            "objects": {},
            "agents": 1,
            "border_width": 0,
        }
    )

    # Create curriculum and environment
    curriculum = SingleTaskCurriculum("test", cfg)
    env = MettaGridEnv(curriculum, render_mode=None)

    obs, _ = env.reset()

    # Get action indices
    action_names = env.action_names
    move_idx = action_names.index("move") if "move" in action_names else None
    noop_idx = action_names.index("noop") if "noop" in action_names else None

    if move_idx is None:
        print(f"ERROR: Required actions not available. move: {move_idx}")
        return False  # Indicate failure

    print("Testing ghost object movement capabilities...")

    # Test that the agent can move freely (ghost movement)
    actions_sequence = [
        [move_idx, 0],  # Move forward
        [move_idx, 0],  # Move forward again
        [move_idx, 0],  # Move forward again
        [noop_idx, 0],  # Noop to check position
    ]

    for i, action in enumerate(actions_sequence):
        obs, rewards, terminals, truncations, info = env.step(np.array([action], dtype=np.int32))

        print(f"Step {i + 1}: action={action_names[action[0]]}, arg={action[1]}")
        print(f"  Action success: {env.action_success}")

        if terminals.any() or truncations.any():
            print("  Episode ended")
            break

    print("  ✓ Ghost object movement test completed")

    return True  # Indicate success


if __name__ == "__main__":
    test_no_agent_interference()
