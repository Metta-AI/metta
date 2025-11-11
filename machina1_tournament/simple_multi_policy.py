#!/usr/bin/env python
"""Simple example: 4 different policies on machina_1.

This is a minimal example showing the core mechanics without CLI parsing.
"""

import sys
from pathlib import Path

# Add cogames to path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "cogames" / "src"))

from cogames.cli.mission import get_mission
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.utils import initialize_or_load_policy
from mettagrid.simulator.rollout import Rollout


def main():
    """Run machina_1 with 4 different policies."""

    # 1. Load the mission config
    _, env_cfg, _ = get_mission("machina_1")
    print(f"Loaded machina_1: {env_cfg.game.num_agents} agents")

    # 2. Create PolicyEnvInterface (needed for policy initialization)
    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

    # 3. Define what policies you want for each agent
    # Format: (class_path, checkpoint_path_or_None, description)
    policy_configs = [
        ("mettagrid.policy.random.RandomMultiAgentPolicy", None, "Random Agent"),
        ("mettagrid.policy.random.RandomMultiAgentPolicy", None, "Random Agent"),
        (
            "cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
            None,
            "Baseline Scripted",
        ),
        (
            "cogames.policy.scripted_agent.unclipping_agent.UnclippingPolicy",
            None,
            "Unclipping Scripted",
        ),
        # To use a trained policy:
        # ("metta.agent.policies.fast.FastPolicy", "./checkpoints/my_policy.pt", "Trained Agent"),
    ]

    # 4. Load/initialize each policy
    policy_instances = []
    for class_path, data_path, description in policy_configs:
        print(f"Loading: {description} ({class_path})")
        policy = initialize_or_load_policy(policy_env_info, class_path, data_path)
        policy_instances.append(policy)

    # 5. Create agent_policies list - one per agent
    agent_policies = []
    for agent_id in range(env_cfg.game.num_agents):
        policy = policy_instances[agent_id]
        agent_policy = policy.agent_policy(agent_id)
        agent_policies.append(agent_policy)
        print(f"Agent {agent_id} -> {policy_configs[agent_id][2]}")

    # 6. Create rollout with the agent_policies list
    print("\nStarting rollout...")
    rollout = Rollout(
        env_cfg,
        agent_policies,  # This is the key: list of AgentPolicy objects
        max_action_time_ms=10000,
        render_mode="gui",  # Set to None for headless
        seed=42,
        pass_sim_to_policies=True,
    )

    # 7. Run the episode
    step_count = 0
    max_steps = 1000
    while not rollout.is_done() and step_count < max_steps:
        rollout.step()
        step_count += 1

    # 8. Print results
    print("\n=== Episode Complete ===")
    print(f"Steps: {rollout._sim.current_step}")
    print(f"Episode rewards: {rollout._sim.episode_rewards}")
    print("\nPer-agent breakdown:")
    for agent_id, reward in enumerate(rollout._sim.episode_rewards):
        print(f"  Agent {agent_id} ({policy_configs[agent_id][2]}): {reward:.2f}")


if __name__ == "__main__":
    main()
