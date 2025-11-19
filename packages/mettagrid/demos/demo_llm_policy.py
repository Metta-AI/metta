#!/usr/bin/env python3
"""Demo of LLM policy playing MettaGrid."""

import argparse
import logging
import os
import signal
import sys

from dotenv import load_dotenv

from mettagrid.builder import building
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.policy.llm_policy import LLMAgentPolicy, LLMMultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout

load_dotenv()


logger = logging.getLogger("mettagrid.demos.llm_policy")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM Policy Demo - Watch an LLM play MettaGrid")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (defaults: gpt-4o-mini for OpenAI, claude-3-5-sonnet for Anthropic)",
    )
    parser.add_argument(
        "--render",
        type=str,
        choices=["gui", "unicode", "log", "none"],
        default="log",
        help="Render mode",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def print_cost_summary():
    """Print LLM API cost summary."""
    summary = LLMAgentPolicy.get_cost_summary()
    print("\n" + "="*60)
    print("LLM API COST SUMMARY")
    print("="*60)
    print(f"Total API calls:     {summary['total_calls']}")
    print(f"Total input tokens:  {summary['total_input_tokens']:,}")
    print(f"Total output tokens: {summary['total_output_tokens']:,}")
    print(f"Total tokens:        {summary['total_tokens']:,}")
    print(f"Total cost:          ${summary['total_cost']:.6f}")
    print("="*60 + "\n")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nInterrupted by user (Ctrl+C)")
    print_cost_summary()
    sys.exit(0)


def main():
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(levelname)s - %(name)s - %(message)s'
    )

    # Check for API key
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    if args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Create config
    cfg = MettaGridConfig()
    cfg.game.num_agents = 1  # Single agent for testing
    cfg.game.max_steps = args.steps

    # Define objects
    cfg.game.objects = {
        "wall": building.wall,
        "altar": building.assembler_altar,
        "mine_red": building.assembler_mine_red,
        "generator_red": building.assembler_generator_red,
    }

    # Create simple map
    cfg.game.map_builder = RandomMapBuilder.Config(
        agents=1,
        width=10,
        height=10,
        objects={
            "wall": 5,
            "altar": 1,
            "mine_red": 2,
            "generator_red": 1,
        },
        border_width=1,
        border_object="wall",
    )

    logger.info("=== LLM Policy Demo ===")
    logger.info(f"Provider: {args.provider}")
    logger.info(f"Model: {args.model or 'default'}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max steps: {args.steps}")

    # Create LLM policy
    policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)
    policy = LLMMultiAgentPolicy(
        policy_env_info,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    agent_policies = [policy.agent_policy(0)]

    # Run rollout
    logger.info("\n=== Starting simulation ===")
    rollout = Rollout(config=cfg, policies=agent_policies, render_mode=args.render)

    try:
        rollout.run_until_done()
    except KeyboardInterrupt:
        logger.info("\n=== Simulation interrupted ===")

    logger.info("\n=== Simulation complete ===")
    logger.info(f"Total steps: {rollout._sim.current_step}")
    logger.info(f"Total rewards: {rollout._sim.episode_rewards}")
    logger.info(f"Episode stats: {rollout._sim.episode_stats}")

    # Print cost summary
    print_cost_summary()


if __name__ == "__main__":
    main()
