import contextlib
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
from metta.common.util.constants import SOFTMAX_S3_BASE, SOFTMAX_S3_POLICY_PREFIX
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


def _get_s3_policy_uri_for_run(run_name: str) -> str:
    """Build S3 policy URI for a run name using the :latest selector.

    Args:
        run_name: Name of the training run

    Returns:
        S3 URI pointing to the latest checkpoint for the run

    Example:
        _get_s3_policy_uri_for_run("my_experiment")
        -> "s3://softmax-public/policies/my_experiment/my_experiment:latest.mpt"
    """
    return f"{SOFTMAX_S3_POLICY_PREFIX}/{run_name}/{run_name}:latest.mpt"


class SimTool(Tool):
    """Tool for running policy evaluations on simulation suites.

    Can evaluate policies specified either by:
    - run: Training run name (automatically resolves to latest S3 checkpoint)
    - policy_uris: Explicit list of policy URIs (file://, s3://, etc.)

    Usage examples:
        # Evaluate latest checkpoint from a training run
        SimTool(simulations=my_sims, run="my_experiment_2024")

        # Evaluate specific policy URIs
        SimTool(simulations=my_sims, policy_uris=["s3://bucket/path/policy:v10.mpt"])

        # Can also be invoked with run parameter
        tool.invoke({"run": "my_experiment_2024"})
    """

    # required params:
    simulations: Sequence[SimulationConfig]  # list of simulations to run
    policy_uris: str | Sequence[str] | None = None  # list of policy uris to evaluate
    run: str | None = None  # run name to evaluate (alternative to policy_uris)
    replay_dir: str = Field(default=f"{SOFTMAX_S3_BASE}/replays/{str(uuid.uuid4())}")

    wandb: WandbConfig = WandbConfig.Unconfigured()
    group: str | None = None  # Separate group parameter like in train.py

    stats_dir: str | None = None  # The (local) directory where stats should be stored
    stats_db_uri: str | None = None  # If set, export stats to this url (local path, wandb:// or s3://)
    stats_server_uri: str | None = None  # If set, send stats to this http server
    register_missing_policies: bool = False
    eval_task_id: str | None = None
    push_metrics_to_wandb: bool = False

    def invoke(self, args: dict[str, str]) -> int | None:
        # Handle run parameter from args
        if "run" in args:
            if self.run is not None:
                raise ValueError("run cannot be set via args if already provided in config")
            self.run = args["run"]

        # Determine policy URIs: either from run name or explicit URIs
        if self.run is not None and self.policy_uris is not None:
            raise ValueError("Cannot specify both 'run' and 'policy_uris' parameters")

        if self.run is not None:
            # Convert run name to S3 policy URI
            logger.info(f"Evaluating run '{self.run}' using S3 checkpoint with :latest selector")
            self.policy_uris = [_get_s3_policy_uri_for_run(self.run)]
        elif self.policy_uris is None:
            raise ValueError("Either 'run' or 'policy_uris' is required")

        # Configure wandb using run name if unconfigured, preserving existing settings
        if self.run:
            self.wandb = auto_wandb_config(self.run)
            if self.group:
                self.wandb.group = self.group

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

        # Build WandB context manager (like train.py does)
        wandb_manager = self._build_wandb_manager()

        with wandb_manager as wandb_run:
            if wandb_run:
                logger.info(f"Initialized wandb run: {wandb_run.id}")

            # Get eval_task_id from config if provided
            eval_task_id = None
            if self.eval_task_id:
                eval_task_id = uuid.UUID(self.eval_task_id)

            for policy_uri in self.policy_uris:
                # Normalize the URI using CheckpointManager
                normalized_uri = CheckpointManager.normalize_uri(policy_uri)

                # Verify the checkpoint exists
                try:
                    artifact = CheckpointManager.load_from_uri(normalized_uri, device="cpu")
                    if artifact.state_dict is None:
                        raise ValueError("Policy artifact contained no weights")
                    metadata = CheckpointManager.get_policy_metadata(normalized_uri)
                    del artifact
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

                # Log to WandB if configured
                if self.push_metrics_to_wandb and wandb_run:
                    try:
                        rl_stats.process_policy_evaluator_stats(policy_uri, eval_results, wandb_run)
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

        # Output JSON results to stdout
        # Ensure all logging is flushed before printing JSON
        sys.stderr.flush()
        sys.stdout.flush()

        # Print JSON with a marker for easier extraction
        print("===JSON_OUTPUT_START===")
        print(json.dumps(all_results, indent=2))
        print("===JSON_OUTPUT_END===")

    def _build_wandb_manager(self):
        """Build WandB context manager, consistent with train.py pattern."""
        if self.wandb and self.wandb.enabled:
            return WandbContext(self.wandb, self)
        return contextlib.nullcontext(None)
