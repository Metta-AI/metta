"""Core sweep functionality using the Tool pattern."""

import logging
import time
from typing import Any, Callable, Optional, Sequence

import wandb

from cogweb.cogweb_client import CogwebClient
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.datastruct import flatten_config
from metta.common.util.heartbeat import record_heartbeat
from metta.common.wandb.wandb_context import WandbConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.protein_config import ProteinConfig
from metta.sweep.protein_metta import MettaProtein
from metta.sweep.wandb_utils import (
    fetch_protein_observations_from_wandb,
    record_protein_observation_to_wandb,
)
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

logger = logging.getLogger(__name__)


# TODO: Implement robustness to single failure
# TODO: Implement Distributed logic (meeting with @relh tomorrow)
def sweep(
    sweep_name: str,
    protein_config: ProteinConfig,
    train_tool_factory: Callable[[str], TrainTool],
    wandb_cfg: WandbConfig,
    num_trials: int,
    evaluation_simulations: Sequence[SimulationConfig],
    sweep_server_uri: str = "https://api.observatory.softmax-research.net",
    max_observations_to_load: int = 250,
    stats_client: Optional[StatsClient] = None,
) -> None:
    """Execute a hyperparameter sweep using Protein optimization.

    Args:
        sweep_name: Unique identifier for this sweep
        protein_config: Configuration for the Protein optimizer
        train_tool_factory: Factory function that creates a TrainTool given a run name
        wandb_cfg: Weights & Biases configuration
        num_trials: Number of trials to run
        sweep_server_uri: Cogweb server URI for sweep coordination
        max_observations_to_load: Maximum number of previous observations to load from WandB
        stats_client: Optional stats client for remote monitoring
        evaluation_simulations: Optional simulations to run after each trial
    """
    logger.info(f"Starting sweep '{sweep_name}' for {num_trials} trials")

    # Initialize sweep with Cogweb
    cogweb_client = CogwebClient.get_client(base_url=sweep_server_uri)
    sweep_client = cogweb_client.sweep_client()

    # Register sweep if it doesn't exist
    sweep_info = sweep_client.get_sweep(sweep_name)
    if not sweep_info.exists:
        logger.info(f"Registering sweep {sweep_name}")
        sweep_client.create_sweep(sweep_name, wandb_cfg.project, wandb_cfg.entity, sweep_name)

    # Initialize Protein optimizer and load previous observations
    # Run trials
    for trial_idx in range(num_trials):
        record_heartbeat()
        logger.info(f"Starting trial {trial_idx + 1}/{num_trials}")
        protein = MettaProtein(protein_config)
        previous_observations = fetch_protein_observations_from_wandb(
            wandb_entity=wandb_cfg.entity,
            wandb_project=wandb_cfg.project,
            sweep_name=sweep_name,
            max_observations=max_observations_to_load,
        )
        logger.info(f"Loaded {len(previous_observations)} previous observations")
        for obs in previous_observations:
            protein.observe(obs["suggestion"], obs["objective"], obs["cost"], obs.get("is_failure", False))

        # Generate suggestion and get run name
        protein_suggestion, _ = protein.suggest()
        run_name = cogweb_client.sweep_client().get_next_run_id(sweep_name)

        # Create and configure TrainTool
        train_tool = train_tool_factory(run_name)
        train_tool = apply_suggestion_to_tool(train_tool, protein_suggestion)
        train_tool.wandb = wandb_cfg
        train_tool.wandb.group = sweep_name
        train_tool.wandb.name = run_name
        train_tool.wandb.run_id = run_name
        train_tool.wandb.tags.append("sweep")

        train_start_time = time.time()

        try:
            # Train
            train_tool.invoke(args={}, overrides=[])
            train_time = time.time() - train_start_time

            # Run evaluation
            sim_tool = SimTool(
                simulations=evaluation_simulations,
                policy_uris=[f"wandb://run/{run_name}"],
                selector_type="latest",
                stats_dir=f"/tmp/stats/{run_name}",
                wandb=wandb_cfg,
                push_metrics_to_wandb=True,
                stats_server_uri=sweep_server_uri if stats_client else None,
                system=train_tool.system,
            )

            eval_start_time = time.time()
            sim_tool.invoke(args={}, overrides=[])
            eval_time = time.time() - eval_start_time

            # Fetch metrics from wandb
            api = wandb.Api()
            wandb_run = api.run(f"{wandb_cfg.entity}/{wandb_cfg.project}/{run_name}")
            assert wandb_run is not None, f"Error fetching run: {run_name}"

            summary = dict(wandb_run.summary)

            # Get score and cost
            eval_score = summary.get(f"evaluator/{protein_config.metric}/score")

            if eval_score is None:
                raise ValueError(f"Error during evaluation, got score: None. Check metric: {protein_config.metric}")

            if "monitor/cost/accrued_total" in summary:
                cost_hours = summary["monitor/cost/accrued_total"]
            elif "total_time" in summary:
                cost_hours = summary["total_time"] / 3600.0
            else:
                cost_hours = (train_time + eval_time) / 3600.0

            logger.info(f"Trial {trial_idx + 1}: score={eval_score:.4f}, cost={cost_hours:.4f}h")

            # Record to wandb for sweep tracking
            wandb_run.config.update(
                {
                    "sweep_name": sweep_name,
                    "protein_suggestion": protein_suggestion,
                }
            )

            record_protein_observation_to_wandb(
                wandb_run,
                protein_suggestion,
                eval_score,
                cost_hours,
                is_failure=False,
            )

            wandb_run.summary.update(
                {
                    "trial_index": trial_idx,
                    "trial_score": eval_score,
                    "trial_cost_hours": cost_hours,
                    protein_config.metric: eval_score,
                }
            )

        except Exception as e:
            logger.error(f"Trial {trial_idx + 1} failed: {e}")
            # Let the error propagate - we want hard failures
            raise

    logger.info(f"Sweep '{sweep_name}' completed {num_trials} trials")


def apply_suggestion_to_tool(train_tool: TrainTool, suggestion: dict[str, Any]) -> TrainTool:
    """Apply Protein suggestion to a TrainTool instance.

    Args:
        train_tool: TrainTool instance to modify
        suggestion: Nested dict of parameter values from Protein

    Returns:
        Modified TrainTool instance
    """
    # Apply suggestions by using the Tool's override method
    for key_path, value in flatten_config(suggestion).items():
        # The override method expects dot-separated paths
        train_tool = train_tool.override(key_path, str(value))
        logger.debug(f"Applied {key_path} = {value}")

    return train_tool
