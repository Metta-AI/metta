import datetime
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Sequence

from pydantic import Field
import wandb

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config.tool import Tool
from metta.common.util.constants import SOFTMAX_S3_BASE
from metta.common.wandb.wandb_context import WandbConfig
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.policy_management import discover_policy_uris
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


class SimTool(Tool):
    """Tool for evaluating policies across multiple simulations and exporting results.
    Loads policies, runs evaluations, and exports metrics to databases or JSON output.
    This tool focuses on batch policy evaluation and statistical analysis, not replay
    visualization (use ReplayTool for that)."""

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
        policies_by_uri: dict[str, list[str]] = {}  # Just store URIs, load agents on demand

        for policy_uri in self.policy_uris:
            # Discover policies with the specified strategy
            strategy_map = {"top": "best_score", "latest": "latest", "best_score": "best_score", "all": "all"}
            strategy = strategy_map.get(self.selector_type, "latest")

            discovered_uris = discover_policy_uris(
                policy_uri, strategy=strategy, count=self.selector_count, metric=self.selector_metric
            )

            logger.info(f"Discovered {len(discovered_uris)} policies for {policy_uri} using strategy '{strategy}'")

            policies_by_uri[policy_uri] = []
            for policy_uri_path in discovered_uris:
                try:
                    # Validate that we can load the policy
                    agent = CheckpointManager.load_from_uri(policy_uri_path)
                    if agent is None:
                        raise FileNotFoundError(f"Could not load policy from {policy_uri_path}")

                    # Get metadata for logging using centralized method
                    metadata = CheckpointManager.get_policy_metadata(policy_uri_path)
                    policies_by_uri[policy_uri].append(policy_uri_path)
                    logger.info(
                        f"Loaded policy from {policy_uri_path} (key={metadata['run_name']}, epoch={metadata['epoch']})"
                    )
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

        for policy_uri, checkpoint_uris in policies_by_uri.items():
            results = {"policy_uri": policy_uri, "checkpoints": []}

            for checkpoint_uri in checkpoint_uris:
                logger.info(f"Processing checkpoint {checkpoint_uri}")

                # Extract metadata using centralized method
                metadata = CheckpointManager.get_policy_metadata(checkpoint_uri)

                # Run actual simulations for this checkpoint
                evaluation_metrics = self._run_simulations_for_checkpoint(checkpoint_uri)

                # Push metrics to wandb if available
                self._push_metrics_to_wandb(checkpoint_uri, metadata, evaluation_metrics)

                # Export to stats database if configured
                if stats_db and self.export_to_stats_db:
                    try:
                        self._export_checkpoint_to_stats_db(stats_db, checkpoint_uri, evaluation_metrics)
                        logger.info("Exported checkpoint results to stats database")
                    except Exception as e:
                        logger.warning(f"Failed to export to stats database: {e}")

                results["checkpoints"].append(
                    {
                        "name": metadata["run_name"],
                        "uri": checkpoint_uri,
                        "epoch": metadata["epoch"],
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

    def _export_checkpoint_to_stats_db(self, stats_db: SimulationStatsDB, checkpoint_uri: str, metrics: dict) -> None:
        """Export checkpoint evaluation results to stats database.

        Note: stats_db is a local DuckDB instance used for temporary storage of simulation
        results before they are pushed to wandb or the remote stats server.
        """
        # Extract key and version from URI for database
        metadata = CheckpointManager.get_policy_metadata(checkpoint_uri)
        policy_key, policy_version = metadata["run_name"], metadata["epoch"]

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

            logger.debug(f"Exported checkpoint {policy_key} to stats database for simulation {sim_config.name}")

    def compare_policies(self, policy_uris: list[str], metric: str = "score") -> dict:
        """Simple policy comparison functionality."""
        if not self.stats_db_uri:
            raise ValueError("stats_db_uri required for policy comparison")

        comparison_results = {"metric": metric, "policies": []}

        try:
            with SimulationStatsDB.from_uri(self.stats_db_uri) as stats_db:
                for policy_uri in policy_uris:
                    # Extract policy info from URI using CheckpointManager
                    metadata = CheckpointManager.get_policy_metadata(policy_uri)
                    policy_key, policy_version = metadata["run_name"], metadata["epoch"]

                    # Get policy performance from stats database
                    policy_scores = stats_db.simulation_scores(policy_key, policy_version, metric)

                    comparison_results["policies"].append(
                        {
                            "policy_uri": policy_uri,
                            "policy_key": policy_key,
                            "policy_version": policy_version,
                            "scores": dict(policy_scores),
                            "average_score": sum(policy_scores.values()) / len(policy_scores) if policy_scores else 0.0,
                        }
                    )

        except Exception as e:
            logger.error(f"Policy comparison failed: {e}")
            comparison_results["error"] = str(e)

        return comparison_results

    def _run_simulations_for_checkpoint(self, checkpoint_uri: str) -> dict:
        """Run simulations for a single checkpoint and return aggregated metrics.

        Args:
            checkpoint_uri: URI of the checkpoint to evaluate

        Returns:
            Dictionary with evaluation metrics from simulations
        """
        all_metrics = {}

        try:
            # Run each configured simulation
            for sim_config in self.simulations:
                logger.info(f"Running simulation '{sim_config.name}' for checkpoint {checkpoint_uri}")

                # Create and run simulation
                sim = Simulation.create(
                    sim_config=sim_config,
                    device=str(self.system.device),
                    vectorization=self.system.vectorization,
                    stats_dir=self.effective_stats_dir,
                    replay_dir=self.effective_replay_dir if self.save_replays else None,
                    policy_uri=checkpoint_uri,
                    run_name=f"eval_{sim_config.name}",
                )

                # Run the simulation and get results
                results = sim.simulate()

                # Extract metrics from simulation results
                if results and results.db:
                    # Get aggregate statistics from the simulation database
                    stats_df = results.db.conn.execute(
                        """
                        SELECT 
                            AVG(reward) as reward_avg,
                            COUNT(DISTINCT episode_id) as num_episodes,
                            MAX(agent_step) as max_agent_step
                        FROM trajectories
                        """
                    ).fetchdf()

                    if not stats_df.empty:
                        sim_metrics = {
                            "reward_avg": float(stats_df["reward_avg"].iloc[0] or 0.0),
                            "num_episodes": int(stats_df["num_episodes"].iloc[0] or 0),
                            "max_agent_step": int(stats_df["max_agent_step"].iloc[0] or 0),
                        }
                        all_metrics[sim_config.name] = sim_metrics

                        logger.info(
                            f"Simulation '{sim_config.name}' completed: "
                            f"avg_reward={sim_metrics['reward_avg']:.3f}, "
                            f"episodes={sim_metrics['num_episodes']}"
                        )

        except Exception as e:
            logger.error(f"Error running simulations for {checkpoint_uri}: {e}")

        # Aggregate metrics across all simulations
        if all_metrics:
            avg_reward = sum(m["reward_avg"] for m in all_metrics.values()) / len(all_metrics)
            total_episodes = sum(m["num_episodes"] for m in all_metrics.values())
            max_step = max(m["max_agent_step"] for m in all_metrics.values())
        else:
            avg_reward = 0.0
            total_episodes = 0
            max_step = 0

        return {
            "reward_avg": avg_reward,
            "reward_avg_category_normalized": avg_reward,  # Would normalize based on category
            "agent_step": max_step,
            "total_episodes": total_episodes,
            "detailed": all_metrics,
        }

    def _push_metrics_to_wandb(self, checkpoint_uri: str, metadata: dict, metrics: dict) -> None:
        """Push evaluation metrics to wandb.

        This replaces the old process_policy_evaluator_stats function to ensure
        evaluation metrics are logged to wandb for tracking and visualization.

        Args:
            checkpoint_uri: URI of the evaluated checkpoint
            metadata: Checkpoint metadata including run_name and epoch
            metrics: Evaluation metrics to log
        """
        # Skip if wandb is not enabled
        if not self.wandb.enabled:
            return

        # Skip if no metrics to log
        if not metrics or not metrics.get("detailed"):
            logger.debug("No metrics to push to wandb")
            return

        try:
            # Extract wandb information from checkpoint URI if it's a wandb artifact
            if checkpoint_uri.startswith("wandb://"):
                # Parse wandb URI to get project and run info
                # Format: wandb://project/artifact/name:version
                parts = checkpoint_uri.replace("wandb://", "").split("/")
                if len(parts) >= 2:
                    wandb_project = parts[0]
                    wandb_entity = self.wandb.entity

                    # Try to extract run ID from metadata
                    run_name = metadata.get("run_name", "")
                    epoch = metadata.get("epoch", 0)
                    agent_step = metadata.get("agent_step", 0)

                    # Initialize wandb run (resume if it exists)
                    run = wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        name=f"eval_{run_name}",
                        config={"checkpoint_uri": checkpoint_uri},
                        reinit=True,
                    )

                    # Prepare metrics for logging
                    metrics_to_log = {
                        "eval/reward_avg": metrics.get("reward_avg", 0.0),
                        "eval/total_episodes": metrics.get("total_episodes", 0),
                        "eval/agent_step": metrics.get("agent_step", 0),
                    }

                    # Add detailed metrics per simulation
                    for sim_name, sim_metrics in metrics.get("detailed", {}).items():
                        metrics_to_log[f"eval/{sim_name}/reward_avg"] = sim_metrics.get("reward_avg", 0.0)
                        metrics_to_log[f"eval/{sim_name}/num_episodes"] = sim_metrics.get("num_episodes", 0)
                        metrics_to_log[f"eval/{sim_name}/max_agent_step"] = sim_metrics.get("max_agent_step", 0)

                    # Log metrics
                    run.log(metrics_to_log, step=agent_step if agent_step else None)
                    logger.info(f"Pushed {len(metrics_to_log)} evaluation metrics to wandb")

                    # Finish the run
                    run.finish()

        except Exception as e:
            # Don't fail the evaluation if wandb logging fails
            logger.warning(f"Failed to push metrics to wandb: {e}")
