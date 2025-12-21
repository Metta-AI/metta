#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "gymnasium",
#     "pettingzoo>=1.24",
#     "omegaconf",
#     "typing-extensions",
#     "pydantic",
#     "stable-baselines3>=2.0",
#     "sb3-contrib",
# ]
# ///

"""PettingZoo Demo - Pure PettingZoo ecosystem integration.

This demo shows how to use MettaGridPettingZooEnv with ONLY PettingZoo
and external multi-agent libraries, without any Metta training infrastructure.

Run with: uv run python packages/mettagrid/demos/demo_train_pettingzoo.py (from project root)
"""

import time

import numpy as np
from gymnasium import spaces
from pettingzoo.test import parallel_api_test

# PettingZoo adapter imports
from mettagrid.builder.envs import make_arena
from mettagrid.envs.pettingzoo_env import MettaGridPettingZooEnv
from mettagrid.simulator import Simulator


def demo_pettingzoo_api():
    """Demonstrate PettingZoo API compliance and basic usage."""
    print("PETTINGZOO API DEMO")
    print("=" * 60)

    # Create simulator and config
    simulator = Simulator()
    config = make_arena(num_agents=24)

    # Create PettingZoo environment
    env = MettaGridPettingZooEnv(
        simulator=simulator,
        cfg=config,
    )

    print("PettingZoo environment created")
    print(f"   - Possible agents: {env.possible_agents}")
    print(f"   - Max agents: {env.max_num_agents}")

    observations, _ = env.reset(seed=42)
    print(f"   - Reset successful: {len(observations)} observations")

    print("   - Running PettingZoo API compliance test...")
    parallel_api_test(env, num_cycles=2)
    print("PettingZoo API compliance passed!")


def demo_random_rollout():
    """Demonstrate random policy rollout in PettingZoo environment."""
    print("\nRANDOM ROLLOUT DEMO")
    print("=" * 60)

    # Create simulator and config
    simulator = Simulator()
    config = make_arena(num_agents=24)

    # Create PettingZoo environment
    env = MettaGridPettingZooEnv(
        simulator=simulator,
        cfg=config,
    )

    print("Running random policy rollout...")
    print(f"   - Agents: {env.possible_agents}")

    _, _ = env.reset(seed=42)
    total_reward = {agent: 0 for agent in env.possible_agents}
    steps = 0
    max_steps = 100  # Small for CI

    while steps < max_steps and env.agents:
        actions = {}
        for agent in env.agents:
            action_space = env.action_space(agent)
            assert isinstance(action_space, spaces.Discrete)
            actions[agent] = int(action_space.sample())

        _, rewards, terminations, truncations, _ = env.step(actions)

        for agent, reward in rewards.items():
            total_reward[agent] += reward

        steps += 1

        if all(terminations.values()) or all(truncations.values()):
            print(f"   Episode ended at step {steps}")
            _, _ = env.reset()
            total_reward = {agent: 0 for agent in env.possible_agents}

    print(f"Completed {steps} steps")
    for agent, reward in total_reward.items():
        print(f"   - {agent}: {reward:.2f} total reward")

    assert steps > 0, "Expected at least one step to be taken"
    assert all(isinstance(r, (int, float)) for r in total_reward.values()), "Rewards must be numeric"

    env.close()


def demo_simple_marl_training():
    """Demonstrate simple multi-agent training with PettingZoo."""
    print("\nSIMPLE MULTI-AGENT TRAINING DEMO")
    print("=" * 60)

    # Create simulator and config
    simulator = Simulator()
    config = make_arena(num_agents=24)

    # Create PettingZoo environment
    env = MettaGridPettingZooEnv(
        simulator=simulator,
        cfg=config,
    )

    print("Running simple multi-agent training...")
    print(f"   - Agents: {env.possible_agents}")
    print("   - Training for 300 steps")

    policies = {}
    for agent in env.possible_agents:
        action_space = env.action_space(agent)
        assert isinstance(action_space, spaces.Discrete)
        policies[agent] = np.ones(action_space.n) / action_space.n

    _, _ = env.reset(seed=42)
    total_rewards = {agent: 0 for agent in env.possible_agents}
    steps = 0
    max_steps = 300  # Reduced for faster CI
    episodes = 0

    while steps < max_steps:
        actions = {}

        for agent in env.agents:
            if agent in policies:
                action_space = env.action_space(agent)
                assert isinstance(action_space, spaces.Discrete)
                probs = policies[agent]
                probs = probs / probs.sum()
                actions[agent] = int(np.random.choice(action_space.n, p=probs))
            else:
                action_space = env.action_space(agent)
                assert isinstance(action_space, spaces.Discrete)
                actions[agent] = int(action_space.sample())

        _, rewards, terminations, truncations, _ = env.step(actions)

        for agent, reward in rewards.items():
            if agent in policies and agent in actions and reward > 0:
                action_taken = int(actions[agent])
                action_space = env.action_space(agent)
                assert isinstance(action_space, spaces.Discrete)
                policies[agent][action_taken] *= 1.1
                policies[agent] /= policies[agent].sum()

            total_rewards[agent] += reward

        steps += 1

        if all(terminations.values()) or all(truncations.values()):
            episodes += 1
            _, _ = env.reset()

    print(f"Training completed: {steps} steps, {episodes} episodes")
    for agent, total_reward in total_rewards.items():
        avg_reward = total_reward / steps if steps > 0 else 0
        print(f"   - {agent}: {avg_reward:.3f} avg reward/step")

    assert steps > 0, "Expected at least one training step"
    assert all(not np.isnan(r) for r in total_rewards.values()), "Rewards contain NaN"

    env.close()


def main():
    """Run PettingZoo adapter demo."""
    print("PETTINGZOO ADAPTER DEMO")
    print("=" * 80)
    print("This demo shows MettaGridPettingZooEnv integration with")
    print("the PettingZoo multi-agent ecosystem (no internal training code).")
    print()

    try:
        start_time = time.time()

        # Run pure PettingZoo demos
        demo_pettingzoo_api()
        demo_random_rollout()
        demo_simple_marl_training()

        # Summary
        duration = time.time() - start_time
        print("\n" + "=" * 80)
        print("PETTINGZOO DEMO COMPLETED")
        print("=" * 80)
        print("PettingZoo API compliance: Passed")
        print("Random rollout: Successful")
        print("Multi-agent training: Completed")
        print(f"\nTotal demo time: {duration:.1f} seconds")
        print("\nNext steps:")
        print("   - Use SuperSuit for advanced wrappers")
        print("   - Integrate with MARLlib, RLlib, or Tianshou")
        print("   - Build custom multi-agent algorithms")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
