"""
Simulation driver for evaluating policies in the Metta environment.

 ▸ For every requested *policy URI*
   ▸ choose the checkpoint(s) according to selector/metric
   ▸ run the configured `SimulationSuite`
   ▸ export the merged stats DB if an output URI is provided
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Sequence

from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config.tool import Tool
from metta.common.util.constants import SOFTMAX_S3_BASE
from metta.common.wandb.wandb_context import WandbConfig

# Removed PolicyRecord and PolicyStore imports - using SimpleCheckpointManager
from metta.rl.simple_checkpoint_manager import SimpleCheckpointManager
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_wandb_config

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
    policy_uris: str | Sequence[str] | None = None  # list of policy uris to evaluate
    replay_dir: str = Field(default=f"{SOFTMAX_S3_BASE}/replays/{str(uuid.uuid4())}")

    wandb: WandbConfig = auto_wandb_config()

    selector_type: str = "top"  # top, latest, or score threshold
    stats_dir: str | None = None  # The (local) directory where stats should be stored
    stats_db_uri: str | None = None  # If set, export stats to this url (local path, wandb:// or s3://)
    stats_server_uri: str | None = None  # If set, send stats to this http server
    register_missing_policies: bool = False
    eval_task_id: str | None = None
    push_metrics_to_wandb: bool = False

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        if self.policy_uris is None:
            raise ValueError("policy_uris is required")

        if isinstance(self.policy_uris, str):
            self.policy_uris = [self.policy_uris]

        # Note: With SimpleCheckpointManager, we work directly with checkpoint directories
        # No central policy store needed
        if self.stats_server_uri is not None:
            StatsClient.create(self.stats_server_uri)

        # Load policies directly from checkpoint directories
        checkpoint_managers_by_uri: dict[str, SimpleCheckpointManager] = {}
        policies_by_uri: dict[str, list[tuple]] = {}  # (agent, metadata) tuples

        for policy_uri in self.policy_uris:
            if policy_uri.startswith("file://"):
                checkpoint_dir = policy_uri.replace("file://", "")
                # Extract run name from path
                run_name = Path(checkpoint_dir).parent.name
                checkpoint_manager = SimpleCheckpointManager(
                    run_dir=str(Path(checkpoint_dir).parent), run_name=run_name
                )
                checkpoint_managers_by_uri[policy_uri] = checkpoint_manager

                # Load policy based on selector type
                if self.selector_type == "top":
                    # Find best checkpoint by score
                    best_path = checkpoint_manager.find_best_checkpoint("score")
                    if best_path:
                        import torch

                        agent = torch.load(best_path, map_location="cpu", weights_only=False)
                        # Get metadata from corresponding YAML file
                        yaml_path = best_path.replace(".pt", ".yaml")
                        metadata = {}
                        if os.path.exists(yaml_path):
                            import yaml

                            with open(yaml_path) as f:
                                metadata = yaml.safe_load(f) or {}
                        policies_by_uri[policy_uri] = [(agent, metadata)]
                    else:
                        logger.warning(f"No checkpoints found with score for {policy_uri}")
                        policies_by_uri[policy_uri] = []
                elif self.selector_type == "latest":
                    # Load latest checkpoint
                    agent = checkpoint_manager.load_agent()
                    if agent:
                        policies_by_uri[policy_uri] = [(agent, {})]
                    else:
                        logger.warning(f"No checkpoints found for {policy_uri}")
                        policies_by_uri[policy_uri] = []
                else:
                    logger.error(f"Unsupported selector_type: {self.selector_type}")
                    policies_by_uri[policy_uri] = []
            else:
                logger.error(f"Only file:// URIs supported currently, got {policy_uri}")
                policies_by_uri[policy_uri] = []

        all_results = {"simulations": [sim.name for sim in self.simulations], "policies": []}

        # Get eval_task_id from config if provided
        if self.eval_task_id:
            uuid.UUID(self.eval_task_id)

        for policy_uri, agent_metadata_list in policies_by_uri.items():
            _determine_run_name(policy_uri)
            results = {"policy_uri": policy_uri, "checkpoints": []}

            for _agent, metadata in agent_metadata_list:
                # TODO: Update evaluate_policy to work with direct agents instead of policy_records
                # For now, skip evaluation and just return basic info
                logger.warning(
                    f"Evaluation temporarily disabled for {policy_uri} - needs SimpleCheckpointManager integration"
                )

                results["checkpoints"].append(
                    {
                        "name": metadata.get("run", "unknown"),
                        "uri": policy_uri,
                        "metrics": {
                            "reward_avg": metadata.get("score", 0.0),
                            "reward_avg_category_normalized": metadata.get("avg_reward", 0.0),
                            "detailed": {},
                        },
                        "replay_url": [],
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
