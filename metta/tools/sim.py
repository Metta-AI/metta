"""
Simulation driver for evaluating policies in the Metta environment.

 ▸ For every requested *policy URI*
   ▸ choose the checkpoint(s) according to selector/metric
   ▸ run the configured `SimulationSuite`
   ▸ export the merged stats DB if an output URI is provided
"""

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicySelectorType, PolicyStore
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig, WandbConfigOff
from metta.eval.eval_service import evaluate_policy
from metta.rl.stats import process_policy_evaluator_stats
from metta.sim.simulation_config import SimulationConfig

logger = logging.getLogger(__name__)


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


class SimTool(Tool):
    # required params:
    simulations: Sequence[SimulationConfig]  # list of simulations to run
    policy_uris: Sequence[str]  # list of policy uris to evaluate
    replay_dir: str  # where to store replays

    wandb: WandbConfig = WandbConfigOff()

    selector_type: PolicySelectorType = "top"
    stats_dir: str | None = None  # The (local) directory where stats should be stored
    stats_db_uri: str | None = None  # If set, export stats to this url (local path, wandb:// or s3://)
    stats_server_uri: str | None = None  # If set, send stats to this http server
    register_missing_policies: bool = False
    eval_task_id: str | None = None
    push_metrics_to_wandb: bool = False

    def invoke(self) -> None:
        # TODO(daveey): #dehydration
        policy_store = PolicyStore.create(
            device=self.system.device,
            wandb_config=self.wandb,
            data_dir=self.system.data_dir,
            wandb_run=None,
        )
        stats_client: StatsClient | None = None
        if self.stats_server_uri is not None:
            stats_client = StatsClient.create(self.stats_server_uri)

        policy_records_by_uri: dict[str, list[PolicyRecord]] = {
            policy_uri: policy_store.policy_records(
                uri_or_config=policy_uri,
                selector_type=self.selector_type,
                n=1,
                metric=self.simulations[0].name + "_score",
            )
            for policy_uri in self.policy_uris
        }

        all_results = {"simulations": [sim.name for sim in self.simulations], "policies": []}
        device = torch.device(self.system.device)

        # Get eval_task_id from config if provided
        eval_task_id = None
        if self.eval_task_id:
            eval_task_id = uuid.UUID(self.eval_task_id)

        for policy_uri, policy_prs in policy_records_by_uri.items():
            eval_run_name = _determine_run_name(policy_uri)
            results = {"policy_uri": policy_uri, "checkpoints": []}
            for pr in policy_prs:
                eval_results = evaluate_policy(
                    policy_record=pr,
                    simulations=self.simulations,
                    stats_dir=self.stats_dir,
                    replay_dir=f"{self.replay_dir}/{eval_run_name}/{pr.run_name}",
                    device=device,
                    vectorization=self.system.vectorization,
                    export_stats_db_uri=self.stats_db_uri,
                    policy_store=policy_store,
                    stats_client=stats_client,
                    logger=logger,
                    eval_task_id=eval_task_id,
                )
                if self.push_metrics_to_wandb:
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
                        "replay_url": next(iter(eval_results.replay_urls.values()))
                        if eval_results.replay_urls
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
