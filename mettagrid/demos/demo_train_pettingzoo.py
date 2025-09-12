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

Run with: uv run python mettagrid/demos/demo_train_pettingzoo.py (from project root)
"""

import time

import numpy as np

# Demo configuration
from demo_config import DEFAULT_CONFIG as config
from gymnasium import spaces
from pettingzoo.test import parallel_api_test

# PettingZoo adapter imports
from metta.mettagrid import MettaGridPettingZooEnv


def demo_pettingzoo_api():
    """Demonstrate PettingZoo API compliance and basic usage."""
    print("PETTINGZOO API DEMO")
    print("=" * config.separator_short)

    # Create PettingZoo environment with shared config
    env = MettaGridPettingZooEnv(
        mg_config=config.get_pettingzoo_config(),
        render_mode=config.render_mode,
    )

    print("PettingZoo environment created")
    print(f"   - Possible agents: {env.possible_agents}")
    print(f"   - Max agents: {env.max_num_agents}")

    observations, _ = env.reset(seed=config.seed)
    print(f"   - Reset successful: {len(observations)} observations")

    print("   - Running PettingZoo API compliance test...")
    parallel_api_test(env, num_cycles=config.pettingzoo_api_test_cycles)
    print("PettingZoo API compliance passed!")


def demo_random_rollout():
    """Demonstrate random policy rollout in PettingZoo environment."""
    print("\nRANDOM ROLLOUT DEMO")
    print("=" * config.separator_short)

    # Create PettingZoo environment with shared config
    env = MettaGridPettingZooEnv(
        mg_config=config.get_pettingzoo_config(),
        render_mode=config.render_mode,
        is_training=config.pettingzoo_is_training,
    )

    print("Running random policy rollout...")
    print(f"   - Agents: {env.possible_agents}")

    _, _ = env.reset(seed=config.seed)
    total_reward = {agent: 0 for agent in env.possible_agents}
    steps = 0
    max_steps = config.max_steps_quick  # Small for CI

    while steps < max_steps and env.agents:
        actions = {}
        for agent in env.agents:
            action_space = env.action_space(agent)
            if isinstance(action_space, spaces.MultiDiscrete):
                actions[agent] = np.random.randint(
                    0, action_space.nvec, size=len(action_space.nvec), dtype=config.np_int32
                )
            else:
                actions[agent] = action_space.sample()

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
        print(f"   - {agent}: {reward:.{config.reward_precision}f} total reward")

    assert steps > 0, "Expected at least one step to be taken"
    assert all(isinstance(r, (int, float)) for r in total_reward.values()), "Rewards must be numeric"

    env.close()


def demo_simple_marl_training():
    """Demonstrate simple multi-agent training with PettingZoo."""
    print("\nSIMPLE MULTI-AGENT TRAINING DEMO")
    print("=" * config.separator_short)

    # Create PettingZoo environment with shared config
    env = MettaGridPettingZooEnv(
        mg_config=config.get_pettingzoo_config(),
        render_mode=config.render_mode,
        is_training=config.pettingzoo_is_training,
    )

    print("Running simple multi-agent training...")
    print(f"   - Agents: {env.possible_agents}")
    print(f"   - Training for {config.max_steps_rollout} steps")

    policies = {}
    for agent in env.possible_agents:
        action_space = env.action_space(agent)
        if isinstance(action_space, spaces.MultiDiscrete):
            # MultiDiscrete case
            policies[agent] = (
                np.ones((len(action_space.nvec), max(action_space.nvec))) / action_space.nvec[:, np.newaxis]
            )
        elif isinstance(action_space, spaces.Discrete):
            # Discrete case
            policies[agent] = np.ones(action_space.n) / action_space.n

    _, _ = env.reset(seed=config.seed)
    total_rewards = {agent: 0 for agent in env.possible_agents}
    steps = 0
    max_steps = config.max_steps_rollout  # Reduced for faster CI
    episodes = 0

    while steps < max_steps:
        actions = {}

        for agent in env.agents:
            if agent in policies:
                action_space = env.action_space(agent)
                if isinstance(action_space, spaces.MultiDiscrete):
                    # MultiDiscrete case
                    action = []
                    for i, nvec in enumerate(action_space.nvec):
                        probs = policies[agent][i, :nvec]
                        probs = probs / probs.sum()
                        action.append(np.random.choice(nvec, p=probs))
                    actions[agent] = np.array(action, dtype=config.np_int32)
                elif isinstance(action_space, spaces.Discrete):
                    # Discrete case
                    probs = policies[agent] / policies[agent].sum()
                    actions[agent] = np.random.choice(action_space.n, p=probs)
            else:
                actions[agent] = env.action_space(agent).sample()

        _, rewards, terminations, truncations, _ = env.step(actions)

        for agent, reward in rewards.items():
            if agent in policies and agent in actions and reward > 0:
                action_taken = actions[agent]
                action_space = env.action_space(agent)
                if isinstance(action_space, spaces.MultiDiscrete):
                    # MultiDiscrete case
                    for i, a in enumerate(action_taken):
                        policies[agent][i, a] *= config.pettingzoo_learning_rate
                        policies[agent][i] /= policies[agent][i].sum()
                elif isinstance(action_space, spaces.Discrete):
                    # Discrete case
                    policies[agent][action_taken] *= config.pettingzoo_learning_rate
                    policies[agent] /= policies[agent].sum()

            total_rewards[agent] += reward

        steps += 1

        if all(terminations.values()) or all(truncations.values()):
            episodes += 1
            _, _ = env.reset()

    print(f"Training completed: {steps} steps, {episodes} episodes")
    for agent, total_reward in total_rewards.items():
        avg_reward = total_reward / steps if steps > 0 else 0
        print(f"   - {agent}: {avg_reward:.{config.avg_reward_precision}f} avg reward/step")

    assert steps > 0, "Expected at least one training step"
    assert all(not np.isnan(r) for r in total_rewards.values()), "Rewards contain NaN"

    env.close()


def main():
    """Run PettingZoo adapter demo."""
    print("PETTINGZOO ADAPTER DEMO")
    print("=" * config.separator_long)
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
        print("\n" + "=" * config.separator_long)
        print("PETTINGZOO DEMO COMPLETED")
        print("=" * config.separator_long)
        print("PettingZoo API compliance: Passed")
        print("Random rollout: Successful")
        print("Multi-agent training: Completed")
        print(f"\nTotal demo time: {duration:.{config.time_precision}f} seconds")
        print("\nNext steps:")
        print("   - Use SuperSuit for advanced wrappers")
        print("   - Integrate with MARLlib, RLlib, or Tianshou")
        print("   - Build custom multi-agent algorithms")
        print("=" * config.separator_long)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
