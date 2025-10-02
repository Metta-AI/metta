import json
import logging
import sys
import uuid
from datetime import datetime
from typing import ClassVar, Sequence

import torch
from pydantic import Field

from metta.app_backend.clients.stats_client import HttpStatsClient, StatsClient
from metta.common.tool import Tool
from metta.common.util.constants import SOFTMAX_S3_BASE
from metta.common.wandb.context import WandbConfig, WandbContext
from metta.eval.eval_service import evaluate_policy
from metta.rl import stats as rl_stats
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_wandb_config
from metta.utils.uri import ParsedURI

logger = logging.getLogger(__name__)


def _determine_run_name(policy_uri: str) -> str:
    parsed = ParsedURI.parse(policy_uri)
    if parsed.scheme == "file" and parsed.local_path is not None:
        return f"eval_{parsed.local_path.stem}"
    if parsed.scheme == "wandb" and parsed.wandb is not None:
        artifact_part = parsed.wandb.artifact_path.split("/")[-1]
        return f"eval_{artifact_part.replace(':', '_')}"
    return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class EvalTool(Tool):
    tool_name: ClassVar[str] = "eval"
    # required params:
    simulations: Sequence[SimulationConfig]  # list of simulations to run
    policy_uris: str | Sequence[str] | None = None  # list of policy uris to evaluate
    replay_dir: str = Field(default=f"{SOFTMAX_S3_BASE}/replays/{str(uuid.uuid4())}")

    wandb: WandbConfig = auto_wandb_config()

    stats_dir: str | None = None  # The (local) directory where stats should be stored
    stats_db_uri: str | None = None  # If set, export stats to this url (local path, wandb:// or s3://)
    stats_server_uri: str | None = None  # If set, send stats to this http server
    register_missing_policies: bool = False
    eval_task_id: str | None = None
    push_metrics_to_wandb: bool = False

    def invoke(self, args: dict[str, str]) -> int | None:
        if self.policy_uris is None:
            raise ValueError("policy_uris is required")

        if isinstance(self.policy_uris, str):
            self.policy_uris = [self.policy_uris]

        for uri in self.policy_uris:
            parsed_uri = ParsedURI.parse(uri)
            if parsed_uri.scheme == "wandb":
                raise ValueError(
                    "Policy artifacts must be stored on local disk or S3. "
                    "Download the checkpoint and re-run with a file:// or s3:// URI."
                )

        stats_client: StatsClient | None = None
        if self.stats_server_uri is not None:
            stats_client = HttpStatsClient.create(self.stats_server_uri)

        all_results = {"simulations": [sim.name for sim in self.simulations], "policies": []}
        device = torch.device(self.system.device)

        wandb_run = None
        wandb_context = None
        if self.wandb and self.wandb.enabled:
            wandb_context = WandbContext(self.wandb, self)
            wandb_context.__enter__()
            wandb_run = wandb_context.run
            logger.info(f"Initialized wandb run: {wandb_run.id if wandb_run else 'None'}")
        # Get eval_task_id from config if provided
        eval_task_id = None
        if self.eval_task_id:
            eval_task_id = uuid.UUID(self.eval_task_id)

        for policy_uri in self.policy_uris:
            # Normalize the URI using CheckpointManager
            normalized_uri = CheckpointManager.normalize_uri(policy_uri)

            # Verify the checkpoint exists
            try:
                agent = CheckpointManager.load_from_uri(normalized_uri, device="cpu")
                metadata = CheckpointManager.get_policy_metadata(normalized_uri)
                del agent
            except Exception as e:
                logger.warning(f"Failed to load policy from {normalized_uri}: {e}")
                continue

            eval_run_name = _determine_run_name(normalized_uri)
            results = {"policy_uri": normalized_uri, "checkpoints": []}

            eval_results = evaluate_policy(
                checkpoint_uri=normalized_uri,
                simulations=list(self.simulations),
                stats_dir=self.stats_dir,
                replay_dir=f"{self.replay_dir}/{eval_run_name}/{metadata.get('run_name', 'unknown')}",
                device=device,
                vectorization=self.system.vectorization,
                export_stats_db_uri=self.stats_db_uri,
                stats_client=stats_client,
                eval_task_id=eval_task_id,
            )
            if self.push_metrics_to_wandb:
                try:
                    rl_stats.process_policy_evaluator_stats(policy_uri, eval_results)
                except Exception as e:
                    logger.error(f"Error logging evaluation results to wandb: {e}")
            results["checkpoints"].append(
                {
                    "name": metadata.get("run_name", "unknown"),
                    "uri": normalized_uri,
                    "metrics": {
                        "reward_avg": eval_results.scores.avg_simulation_score,
                        "reward_avg_category_normalized": eval_results.scores.avg_category_score,
                        "detailed": eval_results.scores.to_wandb_metrics_format(),
                    },
                    "replay_url": eval_results.replay_urls,
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
