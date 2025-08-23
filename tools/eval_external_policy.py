#!/usr/bin/env -S uv run
"""
Script to evaluate external PyTorch policies (e.g., from pufferlib).
Usage: python tools/eval_external_policy.py --policy-path train_dir/metta_7-23/metta.pt
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.eval.eval_service import evaluate_policy
from metta.sim.simulation_config import SimulationSuiteConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate external PyTorch policies")
    parser.add_argument("--policy-path", type=str, required=True, help="Path to the PyTorch policy file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--simulation-suite",
        type=str,
        default="laser_tag",
        help="Simulation suite to use (navigation_single, laser_tag, etc.)",
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes per task")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this evaluation run")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Directory for output files")
    parser.add_argument(
        "--pytorch-config", type=str, default=None, help="Path to YAML config for PyTorch policy architecture"
    )
    parser.add_argument("--save-replays", action="store_true", help="Save replay videos")
    args = parser.parse_args()

    # Set up paths
    policy_path = Path(args.policy_path).resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = f"eval_{policy_path.stem}_{args.simulation_suite}"

    # Load PyTorch config if provided
    pytorch_cfg = None
    if args.pytorch_config:
        pytorch_config_path = Path(args.pytorch_config)
        if pytorch_config_path.exists():
            pytorch_cfg = OmegaConf.load(pytorch_config_path)
            logger.info(f"Loaded PyTorch config from {pytorch_config_path}")
        else:
            logger.warning(f"PyTorch config file not found: {pytorch_config_path}")

    # Create configuration for PolicyStore
    cfg = DictConfig(
        {
            "run": args.run_name,
            "run_dir": str(output_dir / args.run_name),
            "device": args.device,
            "policy_cache_size": 10,
            "vectorization": "serial",
            "pytorch": pytorch_cfg,  # Pass PyTorch config to PolicyStore
            "wandb": {
                "enabled": False  # Disable wandb for simple evaluation
            },
        }
    )

    # Create directories
    os.makedirs(cfg.run_dir, exist_ok=True)
    stats_dir = os.path.join(cfg.run_dir, "stats")
    replay_dir = os.path.join(cfg.run_dir, "replays") if args.save_replays else None

    logger.info(f"Evaluating policy: {policy_path}")
    logger.info(f"Output directory: {cfg.run_dir}")

    # Create policy store and load policy
    policy_store = PolicyStore(cfg, wandb_run=None)

    # Load the policy using pytorch:// URI scheme
    policy_uri = f"pytorch://{policy_path}"
    logger.info(f"Loading policy from URI: {policy_uri}")

    try:
        policy_record = policy_store.load_from_uri(policy_uri)
        logger.info("Successfully loaded policy")
    except Exception as e:
        logger.error(f"Failed to load policy: {e}")
        raise

    # Create simulation suite configuration
    sim_suite_configs = {
        "navigation_single": {
            "name": "navigation_single",
            "simulations": {
                "nav/simple": {
                    "env": "/env/mettagrid/navigation/simple",
                    "num_episodes": args.num_episodes,
                },
            },
        },
        "laser_tag": {
            "name": "laser_tag",
            "simulations": {
                "laser_tag/simple": {
                    "env": "/env/mettagrid/laser_tag",
                    "num_episodes": args.num_episodes,
                },
            },
        },
        "full_eval": {
            "name": "full_eval",
            "simulations": {
                "nav/simple": {
                    "env": "/env/mettagrid/navigation/simple",
                    "num_episodes": args.num_episodes,
                },
                "nav/intermediate": {
                    "env": "/env/mettagrid/navigation/intermediate",
                    "num_episodes": args.num_episodes,
                },
                "laser_tag/simple": {
                    "env": "/env/mettagrid/laser_tag",
                    "num_episodes": args.num_episodes,
                },
            },
        },
    }

    if args.simulation_suite not in sim_suite_configs:
        raise ValueError(
            f"Unknown simulation suite: {args.simulation_suite}. Choose from: {list(sim_suite_configs.keys())}"
        )

    sim_config_dict = sim_suite_configs[args.simulation_suite]
    sim_suite_config = SimulationSuiteConfig(**sim_config_dict)

    logger.info(f"Running {args.simulation_suite} evaluation suite with {args.num_episodes} episodes per task")

    # Run evaluation
    device = torch.device(args.device)

    try:
        results = evaluate_policy(
            policy_record=policy_record,
            simulation_suite=sim_suite_config,
            stats_dir=stats_dir,
            replay_dir=replay_dir,
            device=device,
            vectorization="serial",
            export_stats_db_uri=None,
            policy_store=policy_store,
            stats_client=None,
            logger=logger,
            eval_task_id=None,
        )

        logger.info("\nEvaluation Results:")
        logger.info("=" * 50)
        logger.info(f"Average simulation score: {results.scores.avg_simulation_score:.3f}")
        logger.info(f"Average category score: {results.scores.avg_category_score:.3f}")

        logger.info("\nDetailed scores:")
        for category, sims in results.scores.simulation_scores.items():
            logger.info(f"\n{category}:")
            for sim_name, score in sims.items():
                logger.info(f"  {sim_name}: {score:.3f}")

        # Save results to file
        results_file = Path(cfg.run_dir) / "evaluation_results.yaml"
        results_dict = {
            "policy_path": str(policy_path),
            "simulation_suite": args.simulation_suite,
            "num_episodes": args.num_episodes,
            "device": args.device,
            "results": {
                "avg_simulation_score": float(results.scores.avg_simulation_score),
                "avg_category_score": float(results.scores.avg_category_score),
                "detailed_scores": {
                    category: {sim: float(score) for sim, score in sims.items()}
                    for category, sims in results.scores.simulation_scores.items()
                },
            },
        }

        OmegaConf.save(results_dict, results_file)
        logger.info(f"\nResults saved to: {results_file}")

        if args.save_replays and results.replay_urls:
            logger.info(f"\nReplay files saved to: {replay_dir}")
            for sim_name, url in results.replay_urls.items():
                logger.info(f"  {sim_name}: {url}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
