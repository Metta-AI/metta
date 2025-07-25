#!/usr/bin/env -S uv run
"""
Simulation driver for evaluating policies in the Metta environment.

 ▸ For every requested *policy URI*
   ▸ choose the checkpoint(s) according to selector/metric
   ▸ run the configured `SimulationSuite`
   ▸ export the merged stats DB if an output URI is provided

With --remote flag:
 ▸ Create evaluation tasks in the backend instead of running locally
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List

import torch
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest
from metta.app_backend.stats_client import StatsClient
from metta.common.util.config import Config
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.wandb.wandb_context import WandbConfigOn
from metta.common.wandb.wandb_runs import find_training_jobs, get_artifacts_for_runs, register_with_stats_server
from metta.eval.eval_service import evaluate_policy
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.metta_script import metta_script

# --------------------------------------------------------------------------- #
# Config objects                                                              #
# --------------------------------------------------------------------------- #


class SimJob(Config):
    __init__ = Config.__init__
    simulation_suite: SimulationSuiteConfig
    policy_uris: List[str]
    selector_type: str = "top"
    stats_db_uri: str
    register_missing_policies: bool = False
    stats_dir: str  # The (local) directory where stats should be stored
    replay_dir: str  # where to store replays
    git_hash: str | None = None  # Git hash for remote mode
    env_overrides: dict[str, Any] = {}  # Environment overrides for remote mode


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


def _create_remote_eval_tasks(
    policy_records_by_uri: dict[str, list[PolicyRecord]],
    sim_job: SimJob,
    stats_client: StatsClient,
    logger: logging.Logger,
    wandb_cfg: WandbConfigOn,
) -> None:
    stats_client.validate_authenticated()
    all_policy_records = [pr for prs in policy_records_by_uri.values() for pr in prs]
    policy_ids_by_name = stats_client.get_policy_ids([pr.run_name for pr in all_policy_records]).policy_ids
    logger.info(f"Found {len(policy_ids_by_name)} policies in stats database")
    missing_policy_records = [pr for pr in all_policy_records if not policy_ids_by_name.get(pr.run_name)]
    if missing_policy_records:
        if not sim_job.register_missing_policies:
            logger.warning(f"Policy {missing_policy_records} not found in stats database")
        else:
            logger.info(f"Attempting to register {len(missing_policy_records)} policies in stats database...")

            runs = find_training_jobs(
                entity=wandb_cfg.entity,
                project=wandb_cfg.project,
                run_names=[pr.run_name for pr in missing_policy_records],
            )
            run_infos = get_artifacts_for_runs(runs)
            register_with_stats_server(run_infos, stats_client)
            policy_ids_by_name.update(
                stats_client.get_policy_ids([pr.run_name for pr in missing_policy_records]).policy_ids
            )

    logger.info(f"Creating {len(missing_policy_records)} evaluation tasks")
    for policy_name, policy_id in policy_ids_by_name.items():
        request = TaskCreateRequest(
            policy_id=policy_id,
            git_hash=sim_job.git_hash,
            env_overrides=sim_job.env_overrides,
            sim_suite=sim_job.simulation_suite.name,
        )
        response = stats_client.create_task(request)
        logger.info(f"Created task {response.id} for policy {policy_name} ({policy_id})")


# --------------------------------------------------------------------------- #
# CLI entry-point                                                             #
# --------------------------------------------------------------------------- #


def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("tools.sim")
    if not cfg.get("run"):
        cfg.run = _determine_run_name(cfg.policy_uri)
        logger.info(f"Auto-generated run name: {cfg.run}")

    logger.info(f"Sim job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    sim_job = SimJob(cfg.sim_job)

    policy_store = PolicyStore(cfg, None)
    stats_client: StatsClient | None = get_stats_client(cfg, logger)
    if stats_client is not None:
        stats_client.validate_authenticated()

    policy_records_by_uri: dict[str, list[PolicyRecord]] = {}
    for policy_uri in sim_job.policy_uris:
        # TODO: institutionalize this better?
        metric = sim_job.simulation_suite.name + "_score"  # TODO: make this configurable
        policy_records_by_uri[policy_uri] = policy_store.policy_records(
            policy_uri, sim_job.selector_type, n=1, metric=metric
        )

    # Check if we're in remote mode
    if cfg.remote:
        if not stats_client:
            raise ValueError("Not configured to use stats server. Please provide a stats_server_uri")
        if not cfg.wandb.enabled:
            raise ValueError("Wandb is not enabled. Please provide a wandb_api_key")
        logger.info("Running in remote mode - creating evaluation tasks")
        _create_remote_eval_tasks(policy_records_by_uri, sim_job, stats_client, logger, cfg.wandb)
    else:
        _run_local_simulations(cfg, policy_records_by_uri, sim_job, stats_client, policy_store, logger)


def _run_local_simulations(
    cfg: DictConfig,
    policy_records_by_uri: dict[str, list[PolicyRecord]],
    sim_job: SimJob,
    stats_client: StatsClient | None,
    policy_store: PolicyStore,
    logger: logging.Logger,
) -> None:
    if cfg.git_hash:
        raise ValueError("git_hash is not supported in local mode")
    all_results = {"simulation_suite": sim_job.simulation_suite.name, "policies": []}
    device = torch.device(cfg.device)

    # Get eval_task_id from config if provided
    eval_task_id = None
    if cfg.get("eval_task_id"):
        eval_task_id = uuid.UUID(cfg.eval_task_id)
    for policy_uri, policy_prs in policy_records_by_uri.items():
        results = {"policy_uri": policy_uri, "checkpoints": []}
        for pr in policy_prs:
            policy_results = evaluate_policy(
                policy_record=pr,
                simulation_suite=sim_job.simulation_suite,
                stats_dir=sim_job.stats_dir,
                replay_dir=f"{sim_job.replay_dir}/{pr.run_name}",
                device=device,
                vectorization=cfg.vectorization,
                export_stats_db_uri=sim_job.stats_db_uri,
                policy_store=policy_store,
                stats_client=stats_client,
                logger=logger,
                eval_task_id=eval_task_id,
            )
            results["checkpoints"].append(
                {
                    "name": pr.run_name,
                    "uri": pr.uri,
                    "metrics": {
                        "reward_avg": policy_results.scores.avg_simulation_score,
                        "reward_avg_category_normalized": policy_results.scores.avg_category_score,
                        "detailed": policy_results.scores.to_wandb_metrics_format(),
                    },
                    "replay_url": next(iter(policy_results.replay_urls.values()))
                    if policy_results.replay_urls
                    else None,
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
