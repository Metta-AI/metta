#!/usr/bin/env python3
"""
Example usage of Tribal Environment Python bindings

This example demonstrates how to use the tribal environment bindings
for reinforcement learning training and evaluation.
"""

import sys
from pathlib import Path

import numpy as np

# Add the generated bindings to the path
SCRIPT_DIR = Path(__file__).parent.parent
BINDINGS_DIR = SCRIPT_DIR / "bindings" / "generated"
sys.path.insert(0, str(BINDINGS_DIR))

try:
    import tribal
except ImportError as e:
    print(f"❌ Failed to import tribal bindings: {e}")
    print("Run 'nimble bindings' from mettascope2/ directory first")
    sys.exit(1)


def basic_usage_example():
    """Basic example of using the tribal environment"""
    print("🎮 Basic Tribal Environment Usage")

    # Create environment
    env = tribal.TribalEnv(1000)  # Max 1000 steps
    print(f"✅ Created environment with {tribal.MAP_AGENTS} agents")

    # Reset environment
    env.reset_env()
    print("✅ Environment reset")

    # Single step with noop actions
    actions = tribal.SeqInt()
    for _agent_id in range(tribal.MAP_AGENTS):
        actions.append(0)  # NOOP action
        actions.append(0)  # No argument

    success = env.step(actions)
    print(f"✅ Step completed: {success}")

    # Get observations
    obs = env.get_observations()
    print(f"✅ Observations: {len(obs)} elements")

    # Get rewards
    rewards = env.get_rewards()
    total_reward = sum([rewards[i] for i in range(len(rewards))])
    print(f"✅ Rewards: {total_reward}")

    del env


def rl_training_example():
    """Example of using the environment for RL training"""
    print("\n🤖 RL Training Example")

    env = tribal.TribalEnv(100)
    env.reset_env()

    episode_data = {"observations": [], "actions": [], "rewards": [], "terminals": []}

    for step in range(20):
        # Get current observations
        obs = env.get_observations()
        obs_array = np.array([obs[i] for i in range(len(obs))])
        obs_reshaped = obs_array.reshape(15, 19, 11, 11)  # [agents, layers, height, width]
        episode_data["observations"].append(obs_reshaped.copy())

        # Generate actions (simple policy: move randomly, then try to gather)
        actions = tribal.SeqInt()
        action_list = []

        for _agent_id in range(tribal.MAP_AGENTS):
            if step < 10:
                # First half: explore by moving
                action_type = 1  # MOVE
                argument = np.random.randint(0, 4)  # N, S, W, E
            else:
                # Second half: try to gather resources
                action_type = 3  # GET
                argument = np.random.randint(0, 8)  # All 8 directions

            actions.append(action_type)
            actions.append(argument)
            action_list.append([action_type, argument])

        episode_data["actions"].append(action_list)

        # Step environment
        success = env.step(actions)
        if not success:
            print("❌ Step failed")
            break

        # Get rewards and terminals
        rewards = env.get_rewards()
        terminated = env.get_terminated()

        reward_array = np.array([rewards[i] for i in range(len(rewards))])
        terminal_array = np.array([terminated[i] for i in range(len(terminated))])

        episode_data["rewards"].append(reward_array.copy())
        episode_data["terminals"].append(terminal_array.copy())

        # Print progress
        step_reward = np.sum(reward_array)
        if step % 5 == 0 or step_reward > 0:
            print(f"  Step {step + 1}: Reward = {step_reward:.4f}")

    # Episode summary
    total_reward = sum(np.sum(rewards) for rewards in episode_data["rewards"])
    print(f"✅ Episode completed: {len(episode_data['observations'])} steps, {total_reward:.4f} total reward")

    # Data shapes for ML
    print("✅ Data shapes:")
    print(f"  Observations: {len(episode_data['observations'])} × {episode_data['observations'][0].shape}")
    print(f"  Actions: {len(episode_data['actions'])} × {len(episode_data['actions'][0])} agents")
    print(f"  Rewards: {len(episode_data['rewards'])} × {episode_data['rewards'][0].shape}")

    del env
    return episode_data


def multi_environment_example():
    """Example of running multiple environments for vectorized training"""
    print("\n🔄 Multi-Environment Example")

    num_envs = 3
    envs = []

    # Create multiple environments
    for _i in range(num_envs):
        env = tribal.TribalEnv(50)
        env.reset_env()
        envs.append(env)

    print(f"✅ Created {num_envs} environments")

    # Run parallel episodes
    all_rewards = []
    for step in range(10):
        step_rewards = []

        for env_id, env in enumerate(envs):
            # Different strategy for each environment
            actions = tribal.SeqInt()
            for agent_id in range(tribal.MAP_AGENTS):
                if env_id == 0:
                    # Env 0: Always move north
                    actions.append(1)  # MOVE
                    actions.append(0)  # North
                elif env_id == 1:
                    # Env 1: Move in circle
                    actions.append(1)  # MOVE
                    actions.append(step % 4)  # N, S, W, E
                else:
                    # Env 2: Try to gather
                    actions.append(3)  # GET
                    actions.append(agent_id % 8)  # Different directions

            success = env.step(actions)
            if success:
                rewards = env.get_rewards()
                env_reward = sum([rewards[i] for i in range(len(rewards))])
                step_rewards.append(env_reward)
            else:
                step_rewards.append(0.0)

        all_rewards.append(step_rewards)

        if step % 3 == 0:
            print(f"  Step {step + 1}: Rewards = {step_rewards}")

    # Summary
    total_rewards = [sum(rewards[i] for rewards in all_rewards) for i in range(num_envs)]
    print(f"✅ Final rewards: {total_rewards}")

    # Cleanup
    for env in envs:
        del env


def environment_inspection_example():
    """Example of inspecting the environment state"""
    print("\n🔍 Environment Inspection Example")

    env = tribal.TribalEnv(200)
    env.reset_env()

    # Take a few steps to create interesting state
    for _ in range(5):
        actions = tribal.SeqInt()
        for _agent_id in range(tribal.MAP_AGENTS):
            actions.append(1)  # MOVE
            actions.append(np.random.randint(0, 4))  # Random direction
        env.step(actions)

    # Inspect text rendering
    render = env.render_text()
    lines = render.split("\n")
    print(f"✅ Map visualization ({len(lines)} lines):")
    for i, line in enumerate(lines[:8]):  # Show first 8 lines
        print(f"  {i:2d}: {line[:80]}")  # Truncate long lines

    # Inspect observations for agent 0
    obs = env.get_observations()
    obs_array = np.array([obs[i] for i in range(len(obs))])
    obs_reshaped = obs_array.reshape(15, 19, 11, 11)

    agent_0_obs = obs_reshaped[0]  # Agent 0's observations
    print(f"✅ Agent 0 observations shape: {agent_0_obs.shape}")

    # Show what agent 0 can see in different layers
    layer_names = [
        "Agents",
        "Food",
        "Wood",
        "Stone",
        "Sword",
        "Hat",
        "Armor",
        "Altar",
        "Mine",
        "Wall",
        "Water",
        "Trees",
        "Spawner",
        "Clippy",
        "Chest",
        "Loom",
        "Armory",
        "Forge",
        "Workshop",
    ]

    for layer_id in range(min(10, len(layer_names))):  # First 10 layers
        layer_data = agent_0_obs[layer_id]
        visible_count = np.sum(layer_data > 0)
        if visible_count > 0:
            print(f"  Layer {layer_id:2d} ({layer_names[layer_id]:8s}): {visible_count} visible objects")

    del env


if __name__ == "__main__":
    print("🌟 Tribal Environment Python Bindings Examples")
    print(f"📊 Environment specs: {tribal.MAP_AGENTS} agents, {tribal.MAP_WIDTH}×{tribal.MAP_HEIGHT} map")
    print()

    try:
        # Run all examples
        basic_usage_example()
        episode_data = rl_training_example()
        multi_environment_example()
        environment_inspection_example()

        print("\n🎉 All examples completed successfully!")
        print("\n💡 Next steps:")
        print("  - Integrate with your RL framework (PyTorch, TensorFlow, etc.)")
        print("  - Implement custom reward shaping")
        print("  - Create curriculum learning scenarios")
        print("  - Use vectorized environments for faster training")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
