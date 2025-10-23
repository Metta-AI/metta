"""Runner to evaluate the ScriptedAgentPolicy on a training facility mission.

Usage: run this module with the workspace Python environment. It will create a small
episode, run the scripted agents, and log actions and rewards to stdout.

Examples:
  # Headless
  uv run python -u packages/cogames/scripts/run_scripted.py --map training_facility_tight_4.map --cogs 2 --steps 1000

  # GUI
  METTA_RENDER=gui uv run python -u packages/cogames/scripts/run_scripted.py --map training_facility_tight_4.map --cogs 2 --steps 1000
"""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
import argparse
import os

from cogames.cogs_vs_clips.missions import make_game
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv, dtype_actions


logger = logging.getLogger("cogames.run_scripted")


def run_episode(num_cogs: int = 1, max_steps: int = 500, seed: int = 1, map_name: str | None = None) -> None:
    """Run a single episode using the scripted policy and log progress."""
    env_cfg = make_game(num_cogs=num_cogs, map_name=(map_name or "training_facility_open_1.map"))

    # Read render mode from env var
    render_mode_env = os.getenv("METTA_RENDER", "none")
    if render_mode_env not in ("gui", "unicode", "none"):
        render_mode_env = "none"
    render_mode = render_mode_env

    # Create env
    env = MettaGridEnv(env_cfg=env_cfg, render_mode=render_mode)

    policy = ScriptedAgentPolicy(env)

    # Build per-agent policies
    agent_policies = [policy.agent_policy(i) for i in range(env.num_agents)]

    # Reset env
    obs, _ = env.reset(seed=seed)

    total_rewards = np.zeros(env.num_agents, dtype=float)
    step = 0

    # Action name lookup
    action_names = list(env.action_names)

    logger.info(f"Starting episode: agents={env.num_agents}, max_steps={max_steps}, render={render_mode}")

    while step < max_steps:
        # Check if renderer wants to continue (for GUI mode)
        if not env._renderer.should_continue():
            break

        # Render the environment (updates GUI display)
        env.render()

        actions = np.zeros(env.num_agents, dtype=dtype_actions)

        # Ask each agent for an action
        for agent_id in range(env.num_agents):
            a = agent_policies[agent_id].step(obs[agent_id])
            # Ensure scalar int
            actions[agent_id] = int(a)

        obs, rewards, dones, truncated, info = env.step(actions)

        total_rewards += rewards

        # Log actions and rewards for first few steps and periodically
        if step < 30 or step % 50 == 0:
            act_strings = [action_names[int(a)] if int(a) < len(action_names) else str(int(a)) for a in actions]
            logger.info(f"step={step} actions={act_strings} rewards={rewards} total_rewards={total_rewards}")

        step += 1

        if all(dones) or all(truncated):
            break

    logger.info(f"Episode finished steps={step} total_rewards={total_rewards} sum={float(total_rewards.sum()):.2f}")

    # Keep GUI open if rendering
    if render_mode == "gui":
        print("\n=== Episode Complete ===")
        print(f"Total rewards: {total_rewards}")
        print(f"Sum: {float(total_rewards.sum()):.2f}")
        print("\nPress Enter to close...")
        input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scripted agent on a chosen map")
    parser.add_argument("--map", dest="map_name", default=None, help="Map filename under cogames/maps")
    parser.add_argument("--cogs", dest="cogs", type=int, default=1, help="Number of agents")
    parser.add_argument("--steps", dest="steps", type=int, default=1000, help="Max steps")
    parser.add_argument("--seed", dest="seed", type=int, default=1, help="Seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_episode(num_cogs=args.cogs, max_steps=args.steps, seed=args.seed, map_name=args.map_name)
