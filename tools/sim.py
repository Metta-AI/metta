#!/usr/bin/env -S uv run

"""Simulation driver for evaluating policies in the Metta environment using typer."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import typer
from omegaconf import DictConfig
from pydantic import Field

from cogworks.curriculum.core import Curriculum
from cogworks.curriculum.util import curriculum_from_config_path
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicySelectorType
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.util.typed_config import ConfigWithBuilder
from metta.eval.eval_service import evaluate_policy
from metta.rl.env_config import EnvConfig
from metta.rl.stats import process_policy_evaluator_stats
from metta.sim.simulation_config import SimulationSuiteConfig
from tools.utils import get_policy_store_from_cfg

logger = logging.getLogger(__name__)
app = typer.Typer()


class SimConfig(ConfigWithBuilder):
    """Configuration for the sim script."""

    # Run configuration
    run: str | None = Field(default=None, description="Run name/identifier")

    # Policy URI (single policy for this invocation)
    policy_uri: str = Field(description="Policy URI to evaluate")

    # Simulation configuration (folded from SimJobConfig)
    simulation_suite: SimulationSuiteConfig = Field(description="Simulation suite configuration")
    policy_uris: list[str] = Field(default_factory=list, description="Policy URIs to evaluate")
    selector_type: PolicySelectorType = Field(default="top", description="Policy selector type")
    stats_db_uri: str = Field(description="Stats database URI")
    register_missing_policies: bool = Field(default=False, description="Register missing policies")
    stats_dir: str = Field(default="", description="Local directory for stats storage")
    replay_dir: str = Field(default="", description="Directory for replay storage")

    # Optional config paths
    sim_suite_config_path: str | None = Field(default=None, description="Path to simulation suite config file")
    trainer_task_path: str | None = Field(default=None, description="Path to trainer task config file")
    eval_task_id: str | None = Field(default=None, description="Evaluation task ID")
    push_metrics_to_wandb: bool = Field(default=False, description="Whether to push metrics to wandb")

    # Environment configuration
    env: EnvConfig | None = Field(default=None, description="Environment configuration")
    device: str = Field(default="cuda", description="Device to use")

    # Other configurations
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


@app.command()
def main(
    config: str = typer.Option(..., "--config", help="Path to YAML configuration file"),
    policy_uri: Optional[str] = typer.Option(None, "--policy-uri", help="Override policy URI"),
    device: Optional[str] = typer.Option(None, "--device", help="Override device (cuda/cpu)"),
    run: Optional[str] = typer.Option(None, "--run", help="Override run name"),
):
    """Evaluate Metta AI policies."""
    # Load configuration from file
    cfg = SimConfig.from_file(config)

    # Apply command line overrides
    if policy_uri:
        cfg.policy_uri = policy_uri

    if device:
        cfg.device = device

    if run:
        cfg.run = run

    # Auto-generate run name if not provided
    if not cfg.run:
        cfg.run = _determine_run_name(cfg.policy_uri)
        logger.info(f"Auto-generated run name: {cfg.run}")

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info(f"Starting simulation with config: {cfg.run}")
    logger.info(f"Policy URI: {cfg.policy_uri}")
    logger.info(f"Device: {cfg.device}")

    # Set up directories based on run name
    eval_dir = Path("train_dir") / cfg.run
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.stats_dir:
        cfg.stats_dir = str(eval_dir / "stats")
    if not cfg.replay_dir:
        cfg.replay_dir = str(eval_dir / "replays")

    # Add the single policy URI to the list
    if cfg.policy_uri not in cfg.policy_uris:
        cfg.policy_uris = [cfg.policy_uri]

    # Create directories
    Path(cfg.stats_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.replay_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Sim config: {cfg.run}")
    training_curriculum: Curriculum | None = None

    if cfg.sim_suite_config_path:
        with open(cfg.sim_suite_config_path, "r") as f:
            sim_suite_config_dict = json.load(f)
        cfg.simulation_suite = SimulationSuiteConfig.model_validate(sim_suite_config_dict)
        logger.info(f"Sim suite config:\n{cfg.simulation_suite}")

    if cfg.trainer_task_path:
        logger.info(f"Loading trainer task from {cfg.trainer_task_path}")
        with open(cfg.trainer_task_path, "r") as f:
            trainer_task_dict = json.load(f)
        logger.info(f"Trainer task:\n{trainer_task_dict}")
        if curriculum_name := trainer_task_dict.get("curriculum"):
            training_curriculum = curriculum_from_config_path(
                curriculum_name, DictConfig(trainer_task_dict.get("env_overrides", {}))
            )
            logger.info(f"Training curriculum:\n{training_curriculum}")
    # Create env config
    if cfg.env:
        env_cfg = cfg.env
    else:
        env_cfg = EnvConfig(device=cfg.device)

    # Get policy store - create a simple config dict for compatibility
    config_dict = {"device": cfg.device, **cfg.model_dump()}
    policy_store = get_policy_store_from_cfg(config_dict)

    # Setup stats client
    stats_client: StatsClient | None = get_stats_client(config_dict, logger)
    if stats_client:
        stats_client.validate_authenticated()

    # Get policy records
    policy_records_by_uri: dict[str, list[PolicyRecord]] = {
        policy_uri: policy_store.policy_records(
            uri_or_config=policy_uri,
            selector_type=cfg.selector_type,
            n=1,
            metric=cfg.simulation_suite.name + "_score",
        )
        for policy_uri in cfg.policy_uris
    }

    all_results = {"simulation_suite": cfg.simulation_suite.name, "policies": []}
    device = torch.device(cfg.device)

    # Get eval_task_id from config if provided
    eval_task_id = None
    if cfg.eval_task_id:
        eval_task_id = uuid.UUID(cfg.eval_task_id)

    for policy_uri, policy_prs in policy_records_by_uri.items():
        results = {"policy_uri": policy_uri, "checkpoints": []}
        for pr in policy_prs:
            eval_results = evaluate_policy(
                policy_record=pr,
                simulation_suite=cfg.simulation_suite,
                stats_dir=cfg.stats_dir,
                replay_dir=f"{cfg.replay_dir}/{pr.run_name}",
                device=device,
                vectorization=env_cfg.vectorization,
                export_stats_db_uri=cfg.stats_db_uri,
                policy_store=policy_store,
                stats_client=stats_client,
                logger=logger,
                eval_task_id=eval_task_id,
                training_curriculum=training_curriculum,
            )
            if cfg.push_metrics_to_wandb:
                try:
                    process_policy_evaluator_stats(pr, eval_results)
                except Exception as e:
                    logger.error(f"Error logging evaluation results to wandb: {e}")

            results["checkpoints"].append(
                {
                    "name": pr.run_name,
                    "uri": pr.uri,
                    "metrics": {
                        "reward_avg": eval_results.scores.avg_simulation_score,
                        "reward_avg_category_normalized": eval_results.scores.avg_category_score,
                        "detailed": eval_results.scores.to_wandb_metrics_format(),
                    },
                    "replay_url": next(iter(eval_results.replay_urls.values())) if eval_results.replay_urls else None,
                }
            )
        all_results["policies"].append(results)

    # Always output JSON results to stdout
    # Ensure all logging is flushed before printing JSON
    sys.stderr.flush()
    sys.stdout.flush()

    # Print JSON with a marker for easier extraction
    print("===JSON_OUTPUT_START===")
    print(json.dumps(all_results, indent=2))
    print("===JSON_OUTPUT_END===")

    return 0


if __name__ == "__main__":
    app()
