#!/usr/bin/env -S uv run

"""Simulation driver for evaluating policies - Refactored example."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.typed_config import ConfigWithBuilder
from metta.eval.eval_service import evaluate_policy
from metta.rl.env_config import EnvConfig
from metta.rl.stats import process_policy_evaluator_stats
from metta.sim.simulation_config import SimulationSuiteConfig
from tools.utils import get_policy_store_from_cfg

logger = logging.getLogger(__name__)


class SimToolConfig(ConfigWithBuilder):
    """Configuration for the sim tool - ONLY contains other configs."""

    # Config objects only - no runtime data
    simulation_suite: SimulationSuiteConfig = Field(description="Simulation suite configuration")
    env: Optional[EnvConfig] = Field(default=None, description="Environment configuration")
    wandb: dict[str, Any] = Field(default_factory=dict, description="WandB configuration")


def _determine_run_name(policy_uri: str) -> str:
    """Determine run name from policy URI."""
    if policy_uri.startswith("file://"):
        # Extract checkpoint name from file path
        checkpoint_path = Path(policy_uri.replace("file://", ""))
        return f"eval_{checkpoint_path.stem}"
    elif policy_uri.startswith("wandb://"):
        # Extract artifact name from wandb URI
        # Format: wandb://entity/project/artifact:version
        artifact_part = policy_uri.split("/")[-1]
        return f"eval_{artifact_part.replace(':', '_')}"
    else:
        # Fallback to timestamp
        return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main():
    """Main simulation function."""
    parser = argparse.ArgumentParser(description="Evaluate Metta AI policies")

    # Configuration file (contains only Config objects)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")

    # Runtime parameters (not stored in config file)
    parser.add_argument("--run", type=str, default=None, help="Run name (auto-generated if not provided)")
    parser.add_argument("--policy-uri", type=str, required=True, help="Policy URI to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    # Evaluation job parameters (previously in SimJobConfig)
    parser.add_argument("--policy-uris", type=str, nargs="+", default=[], help="Additional policy URIs to evaluate")
    parser.add_argument(
        "--selector-type",
        type=str,
        default="top",
        choices=["top", "best", "latest", "all"],
        help="Policy selector type",
    )
    parser.add_argument("--stats-db-uri", type=str, required=True, help="Stats database URI")
    parser.add_argument("--register-missing-policies", action="store_true", help="Register missing policies")
    parser.add_argument("--stats-dir", type=str, default="./stats", help="Local directory for stats storage")
    parser.add_argument("--replay-dir", type=str, default="./replays", help="Directory for replay storage")

    # Runtime overrides for specific config fields
    parser.add_argument("--num-sims", type=int, default=None, help="Override number of simulations")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per simulation")

    args = parser.parse_args()

    # Load configuration from file (only Config objects)
    cfg = SimToolConfig.from_file(args.config)

    # Determine run name if not provided
    if not args.run:
        args.run = _determine_run_name(args.policy_uri)

    # Apply runtime overrides to config objects
    if args.num_sims and hasattr(cfg.simulation_suite, "num_sims"):
        cfg.simulation_suite.num_sims = args.num_sims

    if args.max_steps and hasattr(cfg.simulation_suite, "max_steps"):
        cfg.simulation_suite.max_steps = args.max_steps

    # Setup environment config if not provided
    if not cfg.env:
        cfg.env = EnvConfig(device=args.device)
    else:
        cfg.env.device = args.device

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info(f"Starting simulation with run: {args.run}")
    logger.info(f"Policy URI: {args.policy_uri}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Stats DB: {args.stats_db_uri}")
    logger.info(f"Config loaded from: {args.config}")

    # Combine primary and additional policy URIs
    all_policy_uris = [args.policy_uri] + args.policy_uris

    try:
        # Get policy store - create runtime context dict
        runtime_context = {
            "device": args.device,
            "wandb": cfg.wandb,
            "run": args.run,
        }

        policy_store = get_policy_store_from_cfg(runtime_context)

        # Setup stats client
        stats_client = StatsClient(args.stats_db_uri)

        # Process each policy
        for policy_uri in all_policy_uris:
            logger.info(f"Evaluating policy: {policy_uri}")

            # Get policy record
            policy_pr = policy_store.policy_record(policy_uri, args.selector_type)

            # Register policy if needed
            if args.register_missing_policies and not policy_pr:
                logger.info(f"Registering missing policy: {policy_uri}")
                policy_pr = policy_store.register_policy(policy_uri)

            if not policy_pr:
                logger.error(f"Policy not found: {policy_uri}")
                continue

            # Evaluate the policy
            results = evaluate_policy(
                policy_pr,
                cfg.simulation_suite,
                cfg.env,
                stats_dir=Path(args.stats_dir),
                replay_dir=Path(args.replay_dir),
            )

            # Process and store results
            process_policy_evaluator_stats(results, stats_client)

            logger.info(f"Completed evaluation for policy: {policy_uri}")

        logger.info("All simulations completed successfully")

    except Exception as e:
        logger.error(f"Simulation failed with error: {e}")
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
