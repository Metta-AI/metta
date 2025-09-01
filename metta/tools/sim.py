import datetime
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Sequence

from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config.tool import Tool
from metta.common.util.constants import SOFTMAX_S3_BASE
from metta.common.wandb.wandb_context import WandbConfig, WandbRun
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.evaluate import upload_replay_html
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


def process_policy_evaluator_stats(policy_result: dict, eval_results: dict, wandb_run: WandbRun | None = None) -> None:
    """Process evaluation results and push metrics to wandb."""
    if not wandb_run:
        logger.debug("No wandb run available, skipping metric logging")
        return

    metrics = eval_results.get("metrics", {})
    checkpoint_info = policy_result.get("checkpoints", [])

    if not checkpoint_info:
        logger.warning("No checkpoint information in policy results")
        return

    latest_checkpoint = checkpoint_info[-1] if checkpoint_info else {}
    epoch = latest_checkpoint.get("epoch", 0)
    agent_step = metrics.get("agent_step", 0)

    wandb_metrics = {
        "eval/reward_avg": metrics.get("reward_avg", 0.0),
        "eval/reward_avg_category_normalized": metrics.get("reward_avg_category_normalized", 0.0),
        "eval/agent_step": agent_step,
        "eval/epoch": epoch,
    }

    detailed = metrics.get("detailed", {})
    for sim_name, sim_metrics in detailed.items():
        if isinstance(sim_metrics, dict):
            for metric_name, metric_value in sim_metrics.items():
                wandb_metrics[f"eval/{sim_name}/{metric_name}"] = metric_value
        else:
            wandb_metrics[f"eval/detailed/{sim_name}"] = sim_metrics

    wandb_run.log(wandb_metrics, step=agent_step if agent_step > 0 else epoch)

    replay_urls = latest_checkpoint.get("replay_url", [])
    if replay_urls:
        replay_dict = {}
        for i, url in enumerate(replay_urls):
            sim_name = f"sim_{i}"
            if sim_name not in replay_dict:
                replay_dict[sim_name] = []
            replay_dict[sim_name].append(url)

        upload_replay_html(
            replay_urls=replay_dict,
            agent_step=agent_step,
            epoch=epoch,
            wandb_run=wandb_run,
            step_metric_key="eval/agent_step",
            epoch_metric_key="eval/epoch",
        )

    logger.info(f"Successfully logged evaluation metrics to wandb for epoch {epoch}")


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
            # For wandb URIs and file URIs, we expect them to be fully versioned
            # No more strategy-based discovery
            normalized_uri = CheckpointManager.normalize_uri(policy_uri)

            agent = CheckpointManager.load_from_uri(normalized_uri, device=self.system.device)
            if agent is None:
                logger.error(f"Failed to load policy from {normalized_uri}")
                policies_by_uri[policy_uri] = []
                continue

            metadata = CheckpointManager.get_policy_metadata(normalized_uri)
            policies_by_uri[policy_uri] = [normalized_uri]
            logger.info(f"Loaded policy from {normalized_uri} (key={metadata['run_name']}, epoch={metadata['epoch']})")

        all_results = {"simulations": [sim.name for sim in self.simulations], "policies": []}

        wandb_run = None
        wandb_context = None
        if self.wandb and self.wandb.is_configured():
            from metta.common.wandb.wandb_context import WandbContext

            wandb_context = WandbContext(self.wandb)
            wandb_context.__enter__()
            wandb_run = wandb_context.run
            logger.info(f"Initialized wandb run: {wandb_run.id if wandb_run else 'None'}")

        stats_db = None
        if self.export_to_stats_db and self.stats_db_uri:
            stats_db = SimulationStatsDB(Path(self.stats_db_uri))
            stats_db.initialize_schema()
            logger.info(f"Initialized stats database at {self.stats_db_uri}")

        for policy_uri, checkpoint_uris in policies_by_uri.items():
            results = {"policy_uri": policy_uri, "checkpoints": []}

            for checkpoint_uri in checkpoint_uris:
                logger.info(f"Processing checkpoint {checkpoint_uri}")

                metadata = CheckpointManager.get_policy_metadata(checkpoint_uri)
                evaluation_metrics = self._run_simulations_for_checkpoint(checkpoint_uri)

                if stats_db and self.export_to_stats_db:
                    self._export_checkpoint_to_stats_db(stats_db, checkpoint_uri, evaluation_metrics)
                    logger.info("Exported checkpoint results to stats database")

                checkpoint_result = {
                    "name": metadata["run_name"],
                    "uri": checkpoint_uri,
                    "epoch": metadata["epoch"],
                    "metrics": evaluation_metrics,
                    "replay_url": [],
                }
                results["checkpoints"].append(checkpoint_result)

                if wandb_run:
                    process_policy_evaluator_stats(
                        policy_result={"checkpoints": [checkpoint_result]},
                        eval_results={"metrics": evaluation_metrics},
                        wandb_run=wandb_run,
                    )

            all_results["policies"].append(results)

        sys.stderr.flush()
        sys.stdout.flush()

        print("===JSON_OUTPUT_START===")
        print(json.dumps(all_results, indent=2))
        print("===JSON_OUTPUT_END===")

        if wandb_context:
            wandb_context.__exit__(None, None, None)

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

    def _run_simulations_for_checkpoint(self, checkpoint_uri: str) -> dict:
        """Run simulations for a single checkpoint and return aggregated metrics."""
        all_metrics = {}

        for sim_config in self.simulations:
            logger.info(f"Running simulation '{sim_config.name}' for checkpoint {checkpoint_uri}")

            # Create and run simulation
            sim = Simulation.create(
                sim_config=sim_config,
                device=str(self.system.device),
                vectorization=self.system.vectorization,
                stats_dir=self.stats_dir or "/tmp/stats",
                replay_dir=self.replay_dir if getattr(self, "save_replays", False) else None,
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

        # Aggregate metrics
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
            "reward_avg_category_normalized": avg_reward,
            "agent_step": max_step,
            "total_episodes": total_episodes,
            "detailed": all_metrics,
        }
