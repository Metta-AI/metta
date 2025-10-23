"""Runner to evaluate the ScriptedAgentPolicy on a training facility mission.

Usage: run this module with the workspace Python environment. It will create a small
episode, run the scripted agents, and log actions and rewards to stdout.
"""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path

from cogames.cogs_vs_clips.missions import make_game
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv, dtype_actions


logger = logging.getLogger("cogames.run_scripted")


def run_episode(num_cogs: int = 2, max_steps: int = 3, seed: int = 1) -> None:
    """Run a single episode using the scripted policy and log progress."""
    env_cfg = make_game(num_cogs=num_cogs)

    # Create env with no rendering (headless)
    env = MettaGridEnv(env_cfg=env_cfg, render_mode="gui")

    policy = ScriptedAgentPolicy(env)

    # Build per-agent policies
    agent_policies = [policy.agent_policy(i) for i in range(env.num_agents)]

    # Reset env
    obs, _ = env.reset(seed=seed)

    total_rewards = np.zeros(env.num_agents, dtype=float)
    step = 0

    # Action name lookup
    action_names = list(env.action_names)

    logger.info(f"Starting episode: agents={env.num_agents}, max_steps={max_steps}")

    while step < max_steps:
        actions = np.zeros(env.num_agents, dtype=dtype_actions)

        # Ask each agent for an action
        for agent_id in range(env.num_agents):
            a = agent_policies[agent_id].step(obs[agent_id])
            # Ensure scalar int
            actions[agent_id] = int(a)

        obs, rewards, dones, truncated, info = env.step(actions)

        total_rewards += rewards

        # Log actions and rewards for first few steps and periodically
        act_strings = [action_names[int(a)] if int(a) < len(action_names) else str(int(a)) for a in actions]
        logger.info(f"step={step} actions={act_strings} rewards={rewards} total_rewards={total_rewards}")
        print(f"step={step} actions={act_strings} rewards={rewards} total_rewards={total_rewards}")
        step += 1

        if all(dones) or all(truncated):
            break

    logger.info(f"Episode finished steps={step} total_rewards={total_rewards} sum={float(total_rewards.sum()):.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_episode(num_cogs=2, max_steps=100, seed=1)
