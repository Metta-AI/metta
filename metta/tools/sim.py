import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
import yaml
from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config.tool import Tool
from metta.common.util.constants import SOFTMAX_S3_BASE
from metta.common.wandb.wandb_context import WandbConfig
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


def _determine_run_name(policy_uri: str) -> str:
    if policy_uri.startswith("file://"):
        checkpoint_path = Path(policy_uri.replace("file://", ""))
        return f"eval_{checkpoint_path.stem}"
    elif policy_uri.startswith("wandb://"):
        artifact_part = policy_uri.split("/")[-1]
        return f"eval_{artifact_part.replace(':', '_')}"
    else:
        return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class SimTool(Tool):
    # required params:
    simulations: Sequence[SimulationConfig]  # list of simulations to run
    policy_uris: str | Sequence[str] | None = None  # list of policy uris to evaluate
    replay_dir: str = Field(default=f"{SOFTMAX_S3_BASE}/replays/{str(uuid.uuid4())}")

    wandb: WandbConfig = auto_wandb_config()

    selector_type: str = "top"  # top, latest, all, or best_score
    selector_count: int = 1  # number of checkpoints to select
    selector_metric: str = "score"  # metric to use for selection
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

        # Note: With CheckpointManager, we work directly with checkpoint directories
        # No central policy store needed
        if self.stats_server_uri is not None:
            StatsClient.create(self.stats_server_uri)

        # Load policies directly from checkpoint directories
        checkpoint_managers_by_uri: dict[str, CheckpointManager] = {}
        policies_by_uri: dict[str, list[tuple]] = {}  # (agent, metadata, checkpoint_path) tuples

        for policy_uri in self.policy_uris:
            if policy_uri.startswith("file://"):
                checkpoint_dir = policy_uri.replace("file://", "")
                checkpoint_path = Path(checkpoint_dir)

                # Extract run name from path
                run_name = checkpoint_path.parent.name
                run_dir = str(checkpoint_path.parent.parent)
                checkpoint_manager = CheckpointManager(run_name=run_name, run_dir=run_dir)
                checkpoint_managers_by_uri[policy_uri] = checkpoint_manager

                # Select checkpoints using the new selection system
                strategy_map = {"top": "best_score", "latest": "latest", "best_score": "best_score", "all": "all"}
                strategy = strategy_map.get(self.selector_type, "latest")

                selected_paths = checkpoint_manager.select_checkpoints(
                    strategy=strategy, count=self.selector_count, metric=self.selector_metric
                )

                logger.info(f"Selected {len(selected_paths)} checkpoints for {policy_uri} using strategy '{strategy}'")

                policies_by_uri[policy_uri] = []
                for checkpoint_path in selected_paths:
                    try:
                        agent = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

                        yaml_path = checkpoint_path.with_suffix(".yaml")
                        metadata = {}
                        if yaml_path.exists():
                            with open(yaml_path) as f:
                                metadata = yaml.safe_load(f) or {}

                        policies_by_uri[policy_uri].append((agent, metadata, checkpoint_path))
                        logger.info(
                            f"Loaded checkpoint {checkpoint_path.name} with score {metadata.get('score', 'N/A')}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")

                if not policies_by_uri[policy_uri]:
                    logger.warning(f"No valid checkpoints loaded for {policy_uri}")
            else:
                logger.error(f"Only file:// URIs supported currently, got {policy_uri}")
                policies_by_uri[policy_uri] = []

        all_results = {"simulations": [sim.name for sim in self.simulations], "policies": []}

        for policy_uri, agent_metadata_list in policies_by_uri.items():
            _determine_run_name(policy_uri)
            results = {"policy_uri": policy_uri, "checkpoints": []}

            for _agent, metadata, checkpoint_path in agent_metadata_list:
                # Evaluation with CheckpointManager - direct agent evaluation
                # For now, skip evaluation and just return basic info
                logger.info(f"Processing checkpoint {checkpoint_path.name} from {policy_uri}")

                results["checkpoints"].append(
                    {
                        "name": metadata.get("run", checkpoint_path.stem),
                        "uri": f"file://{checkpoint_path}",
                        "epoch": metadata.get("epoch", 0),
                        "checkpoint_path": str(checkpoint_path),
                        "metrics": {
                            "reward_avg": metadata.get("score", 0.0),
                            "reward_avg_category_normalized": metadata.get("avg_reward", 0.0),
                            "agent_step": metadata.get("agent_step", 0),
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
