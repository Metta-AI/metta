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

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicySelectorType
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.config import Config
from metta.common.util.stats_client_cfg import get_stats_client
from metta.eval.eval_service import evaluate_policy
from metta.rl.stats import process_policy_evaluator_stats
from metta.sim.simulation_config import SimulationConfig
from metta.util.metta_script import metta_script
from tools.utils import get_policy_store_from_cfg

# --------------------------------------------------------------------------- #
# Config objects                                                              #
# --------------------------------------------------------------------------- #


class SimToolConfig(Config):
    simulations: list[SimulationConfig]
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


def main(cfg: SimToolConfig) -> None:
    logger = logging.getLogger("tools.sim")

    policy_store = get_policy_store_from_cfg(cfg)
    stats_client: StatsClient | None = get_stats_client(cfg, logger)
    if stats_client:
        stats_client.validate_authenticated()

    policy_records_by_uri: dict[str, list[PolicyRecord]] = {
        policy_uri: policy_store.policy_records(
            uri_or_config=policy_uri,
            selector_type=cfg.selector_type,
            n=1,
            metric=cfg.simulations[0].name + "_score",
        )
        for policy_uri in cfg.policy_uris
    }

    all_results = {"simulations": [sim.name for sim in cfg.simulations], "policies": []}
    device = torch.device(cfg.system.device)

    # Get eval_task_id from config if provided
    eval_task_id = None
    if cfg.eval_task_id:
        eval_task_id = uuid.UUID(cfg.eval_task_id)

    for policy_uri, policy_prs in policy_records_by_uri.items():
        results = {"policy_uri": policy_uri, "checkpoints": []}
        for pr in policy_prs:
            eval_results = evaluate_policy(
                policy_record=pr,
                simulations=cfg.simulations,
                stats_dir=cfg.stats_dir,
                replay_dir=f"{cfg.replay_dir}/{pr.run_name}",
                device=device,
                vectorization=cfg.system.vectorization,
                export_stats_db_uri=cfg.stats_db_uri,
                policy_store=policy_store,
                stats_client=stats_client,
                logger=logger,
                eval_task_id=eval_task_id,
            )
            if cfg.wandb_run:
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
