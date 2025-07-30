"""Policy evaluation functionality."""

import logging
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from metta.eval.eval_request_config import EvalRewardSummary
from metta.eval.eval_service import evaluate_policy as eval_service_evaluate_policy
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.rl.util.evaluation import generate_replay
from metta.rl.util.stats import StatsTracker
from metta.rl.wandb import upload_replay_html
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig


def should_evaluate(epoch: int, evaluate_interval: int, is_master: bool = True) -> bool:
    """Check if evaluation should run at this epoch.

    Args:
        epoch: Current training epoch
        evaluate_interval: How often to evaluate (0 means never)
        is_master: Whether this is the master process

    Returns:
        True if evaluation should run
    """
    if not is_master:
        return False

    if evaluate_interval <= 0:
        return False

    return epoch % evaluate_interval == 0


def evaluate_policy(
    policy_record: Any,
    sim_suite_config: SimulationSuiteConfig,
    curriculum: Any,
    stats_client: Any | None,
    stats_tracker: StatsTracker,
    agent_step: int,
    epoch: int,
    device: torch.device,
    vectorization: str,
    replay_dir: str,
    wandb_policy_name: str | None,
    policy_store: Any,
    cfg: Any,
    wandb_run: Any | None,
    logger: logging.Logger,
) -> EvalRewardSummary:
    """Evaluate policy using the new eval service."""
    # Create an extended simulation suite that includes the training task
    extended_suite_config = SimulationSuiteConfig(
        name=sim_suite_config.name,
        simulations=dict(sim_suite_config.simulations),
        env_overrides=sim_suite_config.env_overrides,
        num_episodes=sim_suite_config.num_episodes,
    )

    # Add training task to the suite
    # Pass the config as _pre_built_env_config to avoid Hydra loading
    task_cfg = curriculum.get_task().env_cfg()
    training_task_config = SingleEnvSimulationConfig(
        env="eval/training_task",  # Just a descriptive name
        num_episodes=1,
        env_overrides={"_pre_built_env_config": task_cfg},
    )
    extended_suite_config.simulations["eval/training_task"] = training_task_config

    logger.info("Simulating policy with extended config including training task")

    # Use the eval service evaluate_policy function
    evaluation_results = eval_service_evaluate_policy(
        policy_record=policy_record,
        simulation_suite=extended_suite_config,
        device=device,
        vectorization=vectorization,
        replay_dir=replay_dir,
        stats_epoch_id=stats_tracker.stats_epoch_id,
        wandb_policy_name=wandb_policy_name,
        policy_store=policy_store,
        stats_client=stats_client,
        logger=logger,
    )

    logger.info("Simulation complete")

    # Set policy metadata score for sweep_eval.py
    target_metric = getattr(cfg, "sweep", {}).get("metric", "reward")  # fallback to reward
    category_scores = list(evaluation_results.scores.category_scores.values())
    if category_scores and policy_record:
        policy_record.metadata["score"] = float(np.mean(category_scores))
        logger.info(f"Set policy metadata score to {policy_record.metadata['score']} using {target_metric} metric")

    # Upload replay HTML if we have wandb
    if wandb_run is not None and evaluation_results.replay_urls:
        upload_replay_html(
            replay_urls=evaluation_results.replay_urls,
            agent_step=agent_step,
            epoch=epoch,
            wandb_run=wandb_run,
        )

    return evaluation_results.scores


def generate_policy_replay(
    policy_record: Any,
    policy_store: Any,
    trainer_cfg: Any,
    epoch: int,
    device: torch.device,
    vectorization: str,
    wandb_run: Any | None,
) -> str | None:
    """Generate a replay for the policy."""
    # Get curriculum from trainer config
    curriculum = curriculum_from_config_path(trainer_cfg.curriculum_or_env, DictConfig(trainer_cfg.env_overrides))

    replay_url = generate_replay(
        policy_record=policy_record,
        policy_store=policy_store,
        curriculum=curriculum,
        epoch=epoch,
        device=device,
        vectorization=vectorization,
        replay_dir=trainer_cfg.simulation.replay_dir,
        wandb_run=wandb_run,
    )

    return replay_url
