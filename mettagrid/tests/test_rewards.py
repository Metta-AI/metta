import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.mettagrid_env import (
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from metta.mettagrid.util.actions import (
    Orientation,
    get_agent_position,
    move,
    rotate,
)

NUM_AGENTS = 1
OBS_HEIGHT = 3
OBS_WIDTH = 3
NUM_OBS_TOKENS = 100
OBS_TOKEN_SIZE = 3


def create_heart_reward_test_env(max_steps=50, num_agents=NUM_AGENTS):
    """Helper function to create a MettaGrid environment with heart collection for reward testing."""

    # Create a simple map with agent, altar, and walls
    game_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "altar", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = {
        "max_steps": max_steps,
        "num_agents": num_agents,
        "obs_width": OBS_WIDTH,
        "obs_height": OBS_HEIGHT,
        "num_observation_tokens": NUM_OBS_TOKENS,
        "inventory_item_names": ["laser", "armor", "heart"],
        "actions": {
            "noop": {"enabled": True},
            "get_items": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "put_items": {"enabled": True},
            "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
            "swap": {"enabled": True},
            "change_color": {"enabled": True},
            "change_glyph": {"enabled": False, "number_of_glyphs": 4},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1},
            "altar": {
                "type_id": 8,
                "output_resources": {"heart": 1},
                "initial_resource_count": 5,  # Start with some hearts
                "max_output": 50,
                "conversion_ticks": 1,  # Faster conversion
                "cooldown": 10,
            },
        },
        "agent": {
            "default_resource_limit": 10,
            "rewards": {
                "inventory": {
                    "heart": 1.0  # This gives 1.0 reward per heart collected
                },
                "stats": {
                    "action.move.success": 0.1,  # 0.1 reward per successful move
                    "action.attack.success": 1.0,  # 1.0 reward per successful attack
                    "action.attack.success_max": 5.0,  # Max 5.0 total reward from attacks
                },
            },
        },
    }

    return MettaGrid(from_mettagrid_config(game_config), game_map, 42)


def create_combat_test_env(max_steps=50, num_agents=2):
    """Helper function to create a MettaGrid environment with multiple agents for combat testing."""

    # Create a simple map with two agents facing each other
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "agent.blue", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = {
        "max_steps": max_steps,
        "num_agents": num_agents,
        "obs_width": OBS_WIDTH,
        "obs_height": OBS_HEIGHT,
        "num_observation_tokens": NUM_OBS_TOKENS,
        "inventory_item_names": ["laser", "armor", "heart"],
        "actions": {
            "noop": {"enabled": True},
            "get_items": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "put_items": {"enabled": True},
            "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
            "swap": {"enabled": True},
            "change_color": {"enabled": True},
            "change_glyph": {"enabled": False, "number_of_glyphs": 4},
        },
        "groups": {
            "red": {
                "id": 0,
                "props": {
                    "rewards": {
                        "inventory": {
                            "heart": 1.0
                        },
                        "stats": {
                            "action.move.success": 0.1,  # 0.1 reward per successful move
                            "action.attack.success": 1.0,  # 1.0 reward per successful attack
                            "action.attack.success_max": 5.0,  # Max 5.0 total reward from attacks
                        },
                    }
                }
            },
            "blue": {"id": 1, "props": {}},
        },
        "objects": {
            "wall": {"type_id": 1},
        },
        "agent": {
            "default_resource_limit": 10,
            "initial_inventory": {"laser": 5, "armor": 2},  # Start with resources for combat
        },
    }

    return MettaGrid(from_mettagrid_config(game_config), game_map, 42)


def perform_action(env, action_name, arg=0, agent_idx=0, num_agents=None):
    """Perform a single action and return results."""
    available_actions = env.action_names()
    
    if num_agents is None:
        num_agents = NUM_AGENTS

    if action_name not in available_actions:
        raise ValueError(f"Unknown action '{action_name}'. Available actions: {available_actions}")

    action_idx = available_actions.index(action_name)
    action = np.zeros((num_agents, 2), dtype=dtype_actions)
    action[agent_idx] = [action_idx, arg]
    obs, rewards, terminals, truncations, info = env.step(action)
    return obs, float(rewards[agent_idx]), env.action_success()[agent_idx]


def wait_for_heart_production(env, steps=5):
    """Wait for altar to produce hearts by performing noop actions."""
    for _ in range(steps):
        perform_action(env, "noop")


def collect_heart_from_altar(env):
    """Move agent to altar (if needed) and collect a heart. Returns (success, reward)."""
    agent_pos = get_agent_position(env, 0)
    _altar_pos = (1, 3)  # Known altar position
    target_pos = (1, 2)  # Adjacent position to altar

    # Only move if not already in the correct position
    if agent_pos != target_pos:
        move_result = move(env, Orientation.RIGHT, agent_idx=0)
        if not move_result["success"]:
            return False, 0.0

    # Rotate to face right (towards altar at (1,3))
    rotate_result = rotate(env, Orientation.RIGHT, agent_idx=0)
    if not rotate_result["success"]:
        return False, 0.0

    # Collect heart
    obs, reward, success = perform_action(env, "get_output", 0)
    return success, reward


class TestRewards:
    def test_step_rewards_initialization(self):
        """Test that step rewards are properly initialized to zero."""
        env = create_heart_reward_test_env()

        # Create buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(NUM_AGENTS, dtype=dtype_terminals)
        truncations = np.zeros(NUM_AGENTS, dtype=dtype_truncations)
        rewards = np.zeros(NUM_AGENTS, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Check that rewards start at zero
        assert np.all(rewards == 0), f"Rewards should start at zero, got {rewards}"

        # Take a step with noop actions
        noop_action_idx = env.action_names().index("noop")
        actions = np.full((NUM_AGENTS, 2), [noop_action_idx, 0], dtype=dtype_actions)

        obs, step_rewards, terminals, truncations, info = env.step(actions)

        # Check that step rewards are accessible and match buffer
        assert np.array_equal(step_rewards, rewards), "Step rewards should match buffer rewards"
        print(f"✅ Step rewards properly initialized: {step_rewards}")

    def test_heart_collection_rewards(self):
        """Test that collecting hearts generates real rewards."""
        env = create_heart_reward_test_env()

        # Create buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(NUM_AGENTS, dtype=dtype_terminals)
        truncations = np.zeros(NUM_AGENTS, dtype=dtype_truncations)
        rewards = np.zeros(NUM_AGENTS, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Collect heart and verify rewards
        success, reward = collect_heart_from_altar(env)

        assert success, "Heart collection should succeed"
        assert reward > 0, f"Heart collection should give positive reward, got {reward}"

        # Check episode rewards
        episode_rewards = env.get_episode_rewards()
        assert episode_rewards[0] > 0, f"Episode rewards should be positive, got {episode_rewards[0]}"

        print(f"✅ Heart collection successful! Reward: {reward}, Episode total: {episode_rewards[0]}")

    def test_multiple_heart_collections(self):
        """Test collecting multiple hearts and verifying cumulative rewards."""
        env = create_heart_reward_test_env()

        # Create buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(NUM_AGENTS, dtype=dtype_terminals)
        truncations = np.zeros(NUM_AGENTS, dtype=dtype_truncations)
        rewards = np.zeros(NUM_AGENTS, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # First collection
        success1, reward1 = collect_heart_from_altar(env)
        episode_rewards_1 = env.get_episode_rewards()[0]

        # Wait and collect again
        wait_for_heart_production(env, steps=10)
        success2, reward2 = collect_heart_from_altar(env)
        episode_rewards_2 = env.get_episode_rewards()[0]

        # Verify both collections worked
        assert success1, "First collection should succeed"
        assert success2, "Second collection should succeed"
        assert reward1 > 0, f"First collection should give positive reward, got {reward1}"
        assert reward2 > 0, f"Second collection should give positive reward, got {reward2}"

        # Verify episode rewards accumulate
        assert episode_rewards_2 > episode_rewards_1, "Episode rewards should accumulate"
        expected_total = episode_rewards_1 + reward2
        assert abs(episode_rewards_2 - expected_total) < 1e-6, (
            f"Episode rewards should accumulate correctly: {episode_rewards_2} vs {expected_total}"
        )

        print("✅ Multiple collections successful!")
        print(f"   Collection 1: reward={reward1}, episode_total={episode_rewards_1}")
        print(f"   Collection 2: reward={reward2}, episode_total={episode_rewards_2}")

    def test_move_stat_rewards(self):
        """Test that agents receive rewards for successful moves."""
        env = create_heart_reward_test_env()

        # Create buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(NUM_AGENTS, dtype=dtype_terminals)
        truncations = np.zeros(NUM_AGENTS, dtype=dtype_truncations)
        rewards = np.zeros(NUM_AGENTS, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Rotate to face down first
        perform_action(env, "rotate", 1)  # Face down

        # Move down (should succeed and give 0.1 reward)
        obs, reward, success = perform_action(env, "move", 0)

        assert success, "Move should succeed"
        assert reward == 0.1, f"Move should give 0.1 reward, got {reward}"

        # Move down again
        obs, reward2, success2 = perform_action(env, "move", 0)

        assert success2, "Second move should succeed"
        assert reward2 == 0.1, f"Second move should give 0.1 reward, got {reward2}"

        # Check cumulative rewards
        episode_rewards = env.get_episode_rewards()
        assert episode_rewards[0] == 0.2, f"Total episode rewards should be 0.2, got {episode_rewards[0]}"

        print("✅ Move stat rewards working correctly!")
        print(f"   Move 1: reward={reward}, success={success}")
        print(f"   Move 2: reward={reward2}, success={success2}")
        print(f"   Total episode rewards: {episode_rewards[0]}")

    def test_attack_stat_rewards(self):
        """Test that agents receive rewards for successful attacks."""
        env = create_combat_test_env()

        # Create buffers for 2 agents
        observations = np.zeros((2, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(2, dtype=dtype_terminals)
        truncations = np.zeros(2, dtype=dtype_truncations)
        rewards = np.zeros(2, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Red agent (index 0) is at (1,1) facing right, blue agent (index 1) is at (1,3)
        # Red agent needs to rotate to face right and attack
        perform_action(env, "rotate", 2, agent_idx=0, num_agents=2)  # Face right

        # Attack (should succeed and give 1.0 reward)
        obs, reward, success = perform_action(env, "attack", 0, agent_idx=0, num_agents=2)

        assert success, "Attack should succeed"
        assert reward == 1.0, f"Attack should give 1.0 reward, got {reward}"

        # Attack again
        obs, reward2, success2 = perform_action(env, "attack", 0, agent_idx=0, num_agents=2)

        assert success2, "Second attack should succeed"
        assert reward2 == 1.0, f"Second attack should give 1.0 reward, got {reward2}"

        # Check cumulative rewards
        episode_rewards = env.get_episode_rewards()
        assert episode_rewards[0] == 2.0, f"Red agent total rewards should be 2.0, got {episode_rewards[0]}"

        print("✅ Attack stat rewards working correctly!")
        print(f"   Attack 1: reward={reward}, success={success}")
        print(f"   Attack 2: reward={reward2}, success={success2}")
        print(f"   Red agent total rewards: {episode_rewards[0]}")

    def test_attack_rewards_max_limit(self):
        """Test that attack rewards respect the maximum limit."""
        env = create_combat_test_env()

        # Create buffers for 2 agents
        observations = np.zeros((2, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(2, dtype=dtype_terminals)
        truncations = np.zeros(2, dtype=dtype_truncations)
        rewards = np.zeros(2, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Rotate to face right
        perform_action(env, "rotate", 2, agent_idx=0, num_agents=2)

        total_attack_reward = 0.0
        attacks = 0

        # Keep attacking until we hit the limit (5.0 max reward)
        for i in range(10):
            obs, reward, success = perform_action(env, "attack", 0, agent_idx=0, num_agents=2)
            if success and reward > 0:
                total_attack_reward += reward
                attacks += 1

        # Should not exceed the max of 5.0
        assert total_attack_reward <= 5.0, f"Total attack rewards should not exceed 5.0, got {total_attack_reward}"
        assert total_attack_reward >= 4.0, f"Total attack rewards should be close to 5.0, got {total_attack_reward}"

        print("✅ Attack rewards max limit working correctly!")
        print(f"   Total attacks: {attacks}")
        print(f"   Total attack rewards: {total_attack_reward}")

    def test_combined_stat_and_inventory_rewards(self):
        """Test that stat rewards and inventory rewards work together."""
        env = create_heart_reward_test_env()

        # Create buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(NUM_AGENTS, dtype=dtype_terminals)
        truncations = np.zeros(NUM_AGENTS, dtype=dtype_truncations)
        rewards = np.zeros(NUM_AGENTS, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Move down (0.1 reward)
        perform_action(env, "rotate", 1)  # Face down
        obs, move_reward, success = perform_action(env, "move", 0)
        assert move_reward == 0.1, f"Move should give 0.1 reward, got {move_reward}"

        # Collect heart (1.0 reward)
        success, heart_reward = collect_heart_from_altar(env)
        assert heart_reward == 1.0, f"Heart collection should give 1.0 reward, got {heart_reward}"

        # Move right (0.1 reward)
        perform_action(env, "rotate", 2)  # Face right
        obs, move_reward2, success = perform_action(env, "move", 0)
        assert move_reward2 == 0.1, f"Second move should give 0.1 reward, got {move_reward2}"

        # Check total rewards
        episode_rewards = env.get_episode_rewards()
        expected_total = 0.1 + 1.0 + 0.1  # move + heart + move
        assert abs(episode_rewards[0] - expected_total) < 1e-6, (
            f"Total rewards should be {expected_total}, got {episode_rewards[0]}"
        )

        print("✅ Combined stat and inventory rewards working correctly!")
        print(f"   Move reward: {move_reward}")
        print(f"   Heart reward: {heart_reward}")
        print(f"   Move reward 2: {move_reward2}")
        print(f"   Total episode rewards: {episode_rewards[0]}")

    def test_failed_move_no_reward(self):
        """Test that failed moves (against walls) don't give rewards."""
        env = create_heart_reward_test_env()

        # Create buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(NUM_AGENTS, dtype=dtype_terminals)
        truncations = np.zeros(NUM_AGENTS, dtype=dtype_truncations)
        rewards = np.zeros(NUM_AGENTS, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Agent starts at (1,1), try to move up into wall (should fail)
        perform_action(env, "rotate", 0)  # Face up
        obs, reward, success = perform_action(env, "move", 0)

        assert not success, "Move into wall should fail"
        assert reward == 0.0, f"Failed move should give 0 reward, got {reward}"

        # Try to move left into wall (should fail)
        perform_action(env, "rotate", 3)  # Face left
        obs, reward2, success2 = perform_action(env, "move", 0)

        assert not success2, "Move into wall should fail"
        assert reward2 == 0.0, f"Failed move should give 0 reward, got {reward2}"

        # Check that no rewards were accumulated
        episode_rewards = env.get_episode_rewards()
        assert episode_rewards[0] == 0.0, f"No rewards should be accumulated for failed moves, got {episode_rewards[0]}"

        print("✅ Failed moves correctly give no rewards!")
        print(f"   Move up into wall: reward={reward}, success={success}")
        print(f"   Move left into wall: reward={reward2}, success={success2}")
        print(f"   Total episode rewards: {episode_rewards[0]}")

    def test_failed_attack_no_reward(self):
        """Test that failed attacks (no target or no resources) don't give rewards."""
        env = create_combat_test_env()

        # Create buffers for 2 agents
        observations = np.zeros((2, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(2, dtype=dtype_terminals)
        truncations = np.zeros(2, dtype=dtype_truncations)
        rewards = np.zeros(2, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Attack in wrong direction (no target)
        perform_action(env, "rotate", 0, agent_idx=0, num_agents=2)  # Face up
        obs, reward, success = perform_action(env, "attack", 0, agent_idx=0, num_agents=2)

        assert not success, "Attack with no target should fail"
        assert reward == 0.0, f"Failed attack should give 0 reward, got {reward}"

        # Use up all laser resources
        perform_action(env, "rotate", 2, agent_idx=0, num_agents=2)  # Face right
        for i in range(5):  # Use all 5 lasers
            perform_action(env, "attack", 0, agent_idx=0, num_agents=2)

        # Try to attack without resources
        obs, reward_no_resource, success_no_resource = perform_action(env, "attack", 0, agent_idx=0, num_agents=2)

        assert not success_no_resource, "Attack without resources should fail"
        assert reward_no_resource == 0.0, f"Attack without resources should give 0 reward, got {reward_no_resource}"

        print("✅ Failed attacks correctly give no rewards!")
        print(f"   Attack with no target: reward={reward}, success={success}")
        print(f"   Attack without resources: reward={reward_no_resource}, success={success_no_resource}")

    def test_move_rewards_with_max_limit(self):
        """Test move rewards with a max limit configuration."""
        # Create environment with move max limit
        game_map = [
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "empty", "empty", "empty", "empty", "empty", "wall"],
            ["wall", "empty", "empty", "empty", "empty", "empty", "empty", "wall"],
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = {
            "max_steps": 100,
            "num_agents": 1,
            "obs_width": OBS_WIDTH,
            "obs_height": OBS_HEIGHT,
            "num_observation_tokens": NUM_OBS_TOKENS,
            "inventory_item_names": [],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
            },
            "groups": {"red": {"id": 0, "props": {}}},
            "objects": {
                "wall": {"type_id": 1},
            },
            "agent": {
                "default_resource_limit": 10,
                "rewards": {
                    "stats": {
                        "action.move.success": 0.2,  # 0.2 reward per successful move
                        "action.move.success_max": 1.0,  # Max 1.0 total reward from moves
                    },
                },
            },
        }

        env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

        # Create buffers
        observations = np.zeros((1, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(1, dtype=dtype_terminals)
        truncations = np.zeros(1, dtype=dtype_truncations)
        rewards = np.zeros(1, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        total_move_reward = 0.0
        moves = 0

        # Keep moving right until we hit the limit
        perform_action(env, "rotate", 2)  # Face right
        for i in range(10):  # Try many moves
            obs, reward, success = perform_action(env, "move", 0)
            if success and reward > 0:
                total_move_reward += reward
                moves += 1

        # Should not exceed the max of 1.0
        assert total_move_reward <= 1.0, f"Total move rewards should not exceed 1.0, got {total_move_reward}"
        assert total_move_reward >= 0.8, f"Total move rewards should be close to 1.0, got {total_move_reward}"

        print("✅ Move rewards max limit working correctly!")
        print(f"   Total moves: {moves}")
        print(f"   Total move rewards: {total_move_reward}")

    def test_stat_tracking_accuracy(self):
        """Test that action stats are accurately tracked in the environment."""
        env = create_combat_test_env()

        # Create buffers for 2 agents
        observations = np.zeros((2, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(2, dtype=dtype_terminals)
        truncations = np.zeros(2, dtype=dtype_truncations)
        rewards = np.zeros(2, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Track moves and attacks for agent 0
        successful_moves = 0
        successful_attacks = 0

        # Perform some moves
        perform_action(env, "rotate", 1, agent_idx=0, num_agents=2)  # Face down
        obs, reward, success = perform_action(env, "move", 0, agent_idx=0, num_agents=2)
        if success:
            successful_moves += 1

        perform_action(env, "rotate", 2, agent_idx=0, num_agents=2)  # Face right  
        obs, reward, success = perform_action(env, "move", 0, agent_idx=0, num_agents=2)
        if success:
            successful_moves += 1

        # Perform some attacks
        for i in range(3):
            obs, reward, success = perform_action(env, "attack", 0, agent_idx=0, num_agents=2)
            if success:
                successful_attacks += 1

        # Get episode stats
        episode_rewards = env.get_episode_rewards()
        
        # Calculate expected rewards
        expected_move_rewards = successful_moves * 0.1
        expected_attack_rewards = min(successful_attacks * 1.0, 5.0)  # Capped at 5.0
        expected_total = expected_move_rewards + expected_attack_rewards

        assert abs(episode_rewards[0] - expected_total) < 1e-6, (
            f"Episode rewards should match expected total: {episode_rewards[0]} vs {expected_total}"
        )

        print("✅ Stat tracking is accurate!")
        print(f"   Successful moves: {successful_moves}")
        print(f"   Successful attacks: {successful_attacks}")
        print(f"   Expected move rewards: {expected_move_rewards}")
        print(f"   Expected attack rewards: {expected_attack_rewards}")
        print(f"   Total episode rewards: {episode_rewards[0]}")

    def test_different_stat_reward_values(self):
        """Test stat rewards with different configuration values."""
        # Create environment with different reward values
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "empty", "agent.blue", "wall"],
            ["wall", "empty", "empty", "empty", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = {
            "max_steps": 50,
            "num_agents": 2,
            "obs_width": OBS_WIDTH,
            "obs_height": OBS_HEIGHT,
            "num_observation_tokens": NUM_OBS_TOKENS,
            "inventory_item_names": ["laser"],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": True, "consumed_resources": {"laser": 1}},
            },
            "groups": {
                "red": {
                    "id": 0,
                    "props": {
                        "rewards": {
                            "stats": {
                                "action.move.success": 0.5,  # Higher move reward
                                "action.attack.success": 2.0,  # Higher attack reward
                                "action.attack.success_max": 10.0,  # Higher max
                            },
                        }
                    }
                },
                "blue": {"id": 1, "props": {}},
            },
            "objects": {
                "wall": {"type_id": 1},
            },
            "agent": {
                "initial_inventory": {"laser": 10},
            },
        }

        env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

        # Create buffers for 2 agents
        observations = np.zeros((2, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(2, dtype=dtype_terminals)
        truncations = np.zeros(2, dtype=dtype_truncations)
        rewards = np.zeros(2, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Test move with higher reward value
        perform_action(env, "rotate", 1, agent_idx=0, num_agents=2)  # Face down
        obs, move_reward, success = perform_action(env, "move", 0, agent_idx=0, num_agents=2)
        assert move_reward == 0.5, f"Move should give 0.5 reward, got {move_reward}"

        # Test attack with higher reward value
        perform_action(env, "rotate", 0, agent_idx=0, num_agents=2)  # Face up
        perform_action(env, "move", 0, agent_idx=0, num_agents=2)  # Move back up
        perform_action(env, "rotate", 2, agent_idx=0, num_agents=2)  # Face right
        obs, attack_reward, success = perform_action(env, "attack", 0, agent_idx=0, num_agents=2)
        assert attack_reward == 2.0, f"Attack should give 2.0 reward, got {attack_reward}"

        print("✅ Different stat reward values working correctly!")
        print(f"   Move reward (0.5 configured): {move_reward}")
        print(f"   Attack reward (2.0 configured): {attack_reward}")
