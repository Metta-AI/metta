import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Sequence

import torch
from pydantic import Field

from metta.app_backend.clients.stats_client import HttpStatsClient, StatsClient
from metta.common.tool import Tool
from metta.common.util.constants import SOFTMAX_S3_BASE
from metta.common.wandb.context import WandbContext
from metta.eval.eval_request_config import EvalResults
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
    return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class SimTool(Tool):
    """Tool for running policy evaluations on simulation suites.

    Can evaluate policies specified either by:
    - run: Training run name (automatically resolves to latest S3 checkpoint)
    - policy_uris: Explicit list of policy URIs (file://, s3://, etc.)

    Usage examples:
        # Evaluate latest checkpoint from a training run
        SimTool(simulations=my_sims, run="my_experiment_2024")

        # Evaluate specific policy URIs
        SimTool(simulations=my_sims, policy_uris=["s3://bucket/path/policy:v10.pt"])

        # Can also be invoked with run parameter
        tool.invoke({"run": "my_experiment_2024"})
    """

    # required params:
    simulations: Sequence[SimulationConfig]  # list of simulations to run
    policy_uris: str | Sequence[str] | None = None  # list of policy uris to evaluate
    replay_dir: str = Field(default=f"{SOFTMAX_S3_BASE}/replays/{str(uuid.uuid4())}")

    group: str | None = None  # Separate group parameter like in train.py

    stats_dir: str | None = None  # The (local) directory where stats should be stored
    stats_db_uri: str | None = None  # If set, export stats to this url (local path, wandb:// or s3://)
    stats_server_uri: str | None = None  # If set, send stats to this http server
    register_missing_policies: bool = False
    eval_task_id: str | None = None
    push_metrics_to_wandb: bool = False

    def _log_to_wandb(self, policy_uri: str, eval_results: EvalResults, stats_client: StatsClient | None):
        if stats_client is None:
            logger.info("Stats client is not set, skipping wandb logging")
            return

        if not self.push_metrics_to_wandb:
            logger.info("Push metrics to wandb is not set, skipping wandb logging")
            return

        run_name = CheckpointManager.get_policy_metadata(policy_uri).get("run_name")
        if run_name is None:
            logger.info("Could not determine run name, skipping wandb logging")
            return

        # Resume the existing training run without overriding its group
        wandb = auto_wandb_config()
        wandb.run_id = run_name  # resume existing
        # Only override group if explicitly provided
        if self.group:
            wandb.group = self.group

        if not wandb.enabled:
            logger.info("WandB is not enabled, skipping wandb logging")
            return

        wandb_context = WandbContext(wandb, self)
        with wandb_context as wandb_run:
            if not wandb_run:
                logger.info("Failed to initialize wandb run, skipping wandb logging")
                return

            logger.info(f"Initialized wandb run: {wandb_run.id}")

            try:
                (epoch, attributes) = stats_client.sql_query(
                    f"""SELECT e.end_training_epoch, e.attributes
                          FROM policies p join epochs e ON p.epoch_id = e.id
                          WHERE p.url = '{policy_uri}'"""
                ).rows[0]
                agent_step = attributes.get("agent_step")
                if agent_step is None:
                    logger.info("Agent step is not set, skipping wandb logging")
                    return

                rl_stats.process_policy_evaluator_stats(policy_uri, eval_results, wandb_run, epoch, agent_step, False)
            except IndexError:
                # No rows returned; log with fallback step/epoch
                logger.info(
                    "No epoch metadata for %s in stats DB; logging eval metrics to WandB with default step/epoch=0",
                    policy_uri,
                )
                try:
                    rl_stats.process_policy_evaluator_stats(policy_uri, eval_results, wandb_run, 0, 0, False)
                except Exception as e:
                    logger.error("Fallback WandB logging failed: %s", e)
            except Exception as e:
                logger.error(f"Error logging evaluation results to wandb: {e}")
                # Best-effort fallback logging with default indices
                try:
                    rl_stats.process_policy_evaluator_stats(policy_uri, eval_results, wandb_run, 0, 0, False)
                except Exception as e2:
                    logger.error("Fallback WandB logging failed: %s", e2)

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

        all_results = {"simulations": [sim.full_name for sim in self.simulations], "policies": []}
        device = torch.device(self.system.device)

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
                logger.warning(f"Failed to load policy from {policy_uri}: {e}")
                continue

            eval_run_name = _determine_run_name(policy_uri)
            results = {"policy_uri": policy_uri, "checkpoints": []}

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

            self._log_to_wandb(normalized_uri, eval_results, stats_client)

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

        # Output JSON results to stdout
        # Ensure all logging is flushed before printing JSON
        sys.stderr.flush()
        sys.stdout.flush()

        # Print JSON with a marker for easier extraction
        print("===JSON_OUTPUT_START===")
        print(json.dumps(all_results, indent=2))
        print("===JSON_OUTPUT_END===")
