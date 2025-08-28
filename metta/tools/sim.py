import json
import logging
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
from metta.rl.checkpoint_interface import Checkpoint
from metta.rl.policy_management import discover_policies, resolve_policy
from metta.sim.simulation_config import SimulationConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
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
    export_to_stats_db: bool = True  # Export evaluation results to stats database

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        if self.policy_uris is None:
            raise ValueError("policy_uris is required")

        if isinstance(self.policy_uris, str):
            self.policy_uris = [self.policy_uris]

        if self.stats_server_uri is not None:
            StatsClient.create(self.stats_server_uri)

        # Load policies using policy management system
        policies_by_uri: dict[str, list[tuple]] = {}  # (agent, metadata, uri) tuples

        for policy_uri in self.policy_uris:
            # Discover policies with the specified strategy
            strategy_map = {"top": "best_score", "latest": "latest", "best_score": "best_score", "all": "all"}
            strategy = strategy_map.get(self.selector_type, "latest")

            discovered_policies = discover_policies(
                policy_uri, strategy=strategy, count=self.selector_count, metric=self.selector_metric
            )

            logger.info(f"Discovered {len(discovered_policies)} policies for {policy_uri} using strategy '{strategy}'")

            policies_by_uri[policy_uri] = []
            for policy_uri_path, metadata in discovered_policies:
                try:
                    agent = resolve_policy(policy_uri_path, device="cpu")
                    policies_by_uri[policy_uri].append((agent, metadata, policy_uri_path))
                    logger.info(f"Loaded policy from {policy_uri_path} with metadata: {metadata}")
                except Exception as e:
                    logger.error(f"Failed to load policy from {policy_uri_path}: {e}")

            if not policies_by_uri[policy_uri]:
                logger.warning(f"No valid policies loaded for {policy_uri}")

        all_results = {"simulations": [sim.name for sim in self.simulations], "policies": []}

        # Initialize stats database if needed
        stats_db = None
        if self.export_to_stats_db and self.stats_db_uri:
            try:
                stats_db = SimulationStatsDB(Path(self.stats_db_uri))
                stats_db.initialize_schema()
                logger.info(f"Initialized stats database at {self.stats_db_uri}")
            except Exception as e:
                logger.warning(f"Failed to initialize stats database: {e}")
                stats_db = None

        for policy_uri, agent_metadata_list in policies_by_uri.items():
            results = {"policy_uri": policy_uri, "checkpoints": []}

            for _agent, metadata, checkpoint_path in agent_metadata_list:
                logger.info(f"Processing checkpoint {checkpoint_path.name} from {policy_uri}")

                # Create Checkpoint object for stats database integration
                checkpoint = Checkpoint(
                    run_name=metadata.get("run", checkpoint_path.stem), uri=policy_uri, metadata=metadata
                )

                # Perform basic evaluation (placeholder - would run actual simulations in real implementation)
                # For Phase 3, we focus on the database integration infrastructure
                evaluation_metrics = {
                    "reward_avg": metadata.get("score", 0.0),
                    "reward_avg_category_normalized": metadata.get("avg_reward", 0.0),
                    "agent_step": metadata.get("agent_step", 0),
                    "detailed": {},
                }

                # Export to stats database if configured
                if stats_db and self.export_to_stats_db:
                    try:
                        self._export_checkpoint_to_stats_db(stats_db, checkpoint, evaluation_metrics)
                        logger.info("Exported checkpoint results to stats database")
                    except Exception as e:
                        logger.warning(f"Failed to export to stats database: {e}")

                results["checkpoints"].append(
                    {
                        "name": checkpoint.run_name,
                        "uri": checkpoint.uri,
                        "epoch": metadata.get("epoch", 0),
                        "checkpoint_path": str(checkpoint_path),
                        "metrics": evaluation_metrics,
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

    def _export_checkpoint_to_stats_db(
        self, stats_db: SimulationStatsDB, checkpoint: Checkpoint, metrics: dict
    ) -> None:
        """Export checkpoint evaluation results to stats database."""
        policy_key, policy_version = checkpoint.key_and_version()

        # Record simulation entries for each configured simulation
        for sim_config in self.simulations:
            simulation_id = (
                f"{sim_config.name}_{policy_key}_{policy_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # Insert simulation record
            stats_db.execute(
                """
                INSERT INTO simulations (id, name, env, policy_key, policy_version, created_at, finished_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
                [simulation_id, sim_config.name, sim_config.env_name, policy_key, policy_version],
            )

            # For this phase, create minimal agent policy records
            # In a real implementation, these would come from actual simulation episodes
            episode_id = f"episode_{simulation_id}_001"
            agent_id = 0

            # Insert agent policy mapping
            stats_db.execute(
                """
                INSERT OR IGNORE INTO agent_policies (episode_id, agent_id, policy_key, policy_version)
                VALUES (?, ?, ?, ?)
            """,
                [episode_id, agent_id, policy_key, policy_version],
            )

            logger.debug(
                f"Exported checkpoint {checkpoint.run_name} to stats database for simulation {sim_config.name}"
            )

    def compare_policies(self, policy_uris: list[str], metric: str = "score") -> dict:
        """Simple policy comparison functionality."""
        if not self.stats_db_uri:
            raise ValueError("stats_db_uri required for policy comparison")

        comparison_results = {"metric": metric, "policies": []}

        try:
            with SimulationStatsDB.from_uri(self.stats_db_uri) as stats_db:
                for policy_uri in policy_uris:
                    # Extract policy info from URI
                    if policy_uri.startswith("file://"):
                        checkpoint_path = Path(policy_uri.replace("file://", ""))
                        run_name = checkpoint_path.parent.name
                        # For file URIs, we'd need to determine epoch from the checkpoint
                        epoch = 0  # Simplified for Phase 3

                        # Get policy performance from stats database
                        policy_scores = stats_db.simulation_scores(str(checkpoint_path), epoch, metric)

                        comparison_results["policies"].append(
                            {
                                "policy_uri": policy_uri,
                                "run_name": run_name,
                                "epoch": epoch,
                                "scores": dict(policy_scores),
                                "average_score": sum(policy_scores.values()) / len(policy_scores)
                                if policy_scores
                                else 0.0,
                            }
                        )

        except Exception as e:
            logger.error(f"Policy comparison failed: {e}")
            comparison_results["error"] = str(e)

        return comparison_results
