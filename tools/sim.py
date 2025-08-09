#!/usr/bin/env -S uv run
"""
Simulation driver for evaluating policies in the Metta environment.

 ▸ For every requested *policy URI*
   ▸ choose the checkpoint(s) according to selector/metric
   ▸ run the configured `SimulationSuite`
   ▸ export the merged stats DB if an output URI is provided
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicySelectorType
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.config import Config
from metta.common.util.stats_client_cfg import get_stats_client
from metta.eval.eval_service import evaluate_policy
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.rl.env_config import create_env_config
from metta.rl.stats import process_policy_evaluator_stats
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.metta_script import metta_script
from tools.utils import get_policy_store_from_cfg

# --------------------------------------------------------------------------- #
# Config objects                                                              #
# --------------------------------------------------------------------------- #


class SimJob(Config):
    __init__ = Config.__init__
    simulation_suite: SimulationSuiteConfig
    policy_uris: list[str]
    selector_type: PolicySelectorType = "top"
    stats_db_uri: str
    register_missing_policies: bool = False
    stats_dir: str  # The (local) directory where stats should be stored
    replay_dir: str  # where to store replays


def _determine_run_name(policy_uri: str) -> str:
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


# --------------------------------------------------------------------------- #
# CLI entry-point                                                             #
# --------------------------------------------------------------------------- #


def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("tools.sim")
    if not cfg.get("run"):
        cfg.run = _determine_run_name(cfg.policy_uri)
        logger.info(f"Auto-generated run name: {cfg.run}")

    sim_job = SimJob(cfg.sim_job)
    training_curriculum: Curriculum | None = None
    if cfg.sim_suite_config:
        logger.info(f"Using sim_suite_config: {cfg.sim_suite_config}")
        sim_job.simulation_suite = SimulationSuiteConfig.model_validate(cfg.sim_suite_config)
    if (
        cfg.trainer_task
        and (parsed := json.loads(cfg.trainer_task))
        and (curriculum_name := parsed.get("curriculum"))
        and (env_overrides := parsed.get("env_overrides"))
    ):
        logger.info(f"Using trainer_task: {curriculum_name} with overrides: {env_overrides}")
        training_curriculum = curriculum_from_config_path(curriculum_name, DictConfig(env_overrides))
    logger.info(f"Sim job config:\n{OmegaConf.to_yaml(sim_job, resolve=True)}")

    # Create env config
    env_cfg = create_env_config(cfg)

    policy_store = get_policy_store_from_cfg(cfg)
    stats_client: StatsClient | None = get_stats_client(cfg, logger)
    if stats_client:
        stats_client.validate_authenticated()

    policy_records_by_uri: dict[str, list[PolicyRecord]] = {
        policy_uri: policy_store.policy_records(
            uri_or_config=policy_uri,
            selector_type=sim_job.selector_type,
            n=1,
            metric=sim_job.simulation_suite.name + "_score",
        )
        for policy_uri in sim_job.policy_uris
    }

    all_results = {"simulation_suite": sim_job.simulation_suite.name, "policies": []}
    device = torch.device(cfg.device)

    # Get eval_task_id from config if provided
    eval_task_id = None
    if cfg.get("eval_task_id"):
        eval_task_id = uuid.UUID(cfg.eval_task_id)
    for policy_uri, policy_prs in policy_records_by_uri.items():
        results = {"policy_uri": policy_uri, "checkpoints": []}
        for pr in policy_prs:
            eval_results = evaluate_policy(
                policy_record=pr,
                simulation_suite=sim_job.simulation_suite,
                stats_dir=sim_job.stats_dir,
                replay_dir=f"{sim_job.replay_dir}/{pr.run_name}",
                device=device,
                vectorization=env_cfg.vectorization,
                export_stats_db_uri=sim_job.stats_db_uri,
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


metta_script(main, "sim_job")
