#!/usr/bin/env -S uv run

"""
Demo showing how to create an MettaGridConfig and build a game map using the map builder.
"""

import argparse
import logging
from typing import get_args

from alo.pure_single_episode_runner import PureSingleEpisodeSpecJob, run_pure_single_episode_from_specs
from mettagrid.builder import building
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.random_map import RandomMapBuilder
from mettagrid.policy.policy import PolicySpec
from mettagrid.renderer.renderer import RenderMode

logger = logging.getLogger("mettagrid.demo")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MettaGrid demo - showcase map building and policy rollout")
    parser.add_argument(
        "--render",
        type=str,
        choices=get_args(RenderMode),
        default="log",
        help="Render mode: gui (mettascope), unicode (miniscope), log (logger), or none (headless)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=4,
        help="Number of agents in the simulation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Maximum number of steps to run",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=10,
        help="Map width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=10,
        help="Map height",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = MettaGridConfig()
    cfg.game.num_agents = args.agents
    cfg.game.max_steps = args.steps

    # Enable change_vibe for GUI control, but create a policy config without it for random actions
    cfg.game.actions.change_vibe.enabled = True
    from mettagrid.config.vibes import VIBES as ALL_VIBES

    cfg.game.actions.change_vibe.vibes = list(ALL_VIBES[:10])

    # Define objects used in the map
    cfg.game.objects = {
        "wall": building.wall,
        "assembler": building.assembler_assembler,
    }

    cfg.game.map_builder = RandomMapBuilder.Config(
        agents=args.agents,
        width=args.width,
        height=args.height,
        objects={"wall": 10, "assembler": 1},
        border_width=1,
        border_object="wall",
    )

    logger.info("=== MettaGrid Demo ===")
    logger.info(f"Render mode: {args.render}")
    logger.info(f"Agents: {args.agents}, Max steps: {args.steps}")
    logger.info(f"Map size: {args.width}x{args.height}")

    map_builder = cfg.game.map_builder.create()
    game_map = map_builder.build()

    logger.info("=== Map Generated ===")
    logger.debug(game_map.grid)

    policy_spec = PolicySpec(class_path="mettagrid.policy.random_agent.RandomMultiAgentPolicy")
    job = PureSingleEpisodeSpecJob(
        policy_specs=[policy_spec],
        assignments=[0] * cfg.game.num_agents,
        env=cfg,
        seed=42,
        max_action_time_ms=10000,
    )

    logger.info("\n=== Running simulation ===")
    results, _replay = run_pure_single_episode_from_specs(job, device="cpu", render_mode=args.render)
    logger.info("=== Simulation complete ===")
    logger.info(f"Total steps: {results.steps}")
    logger.info(f"Total rewards: {results.rewards}")
    logger.info(f"Total stats: {results.stats}")
    logger.info("Done: True")


if __name__ == "__main__":
    main()
