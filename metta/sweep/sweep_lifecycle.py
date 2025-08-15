import logging
import os
import time
from typing import Any

from omegaconf import DictConfig, OmegaConf

from cogweb.cogweb_client import CogwebClient
from metta.common.wandb.wandb_context import WandbContext, WandbRun
from metta.eval.eval_stats_db import EvalStatsDB
from metta.rl.system_config import create_system_config
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.sweep.protein_metta import MettaProtein
from metta.sweep.wandb_utils import (
    fetch_protein_observations_from_wandb,
    record_protein_observation_to_wandb,
)
from tools.utils import get_policy_store_from_cfg

logger = logging.getLogger(__name__)


def initialize_sweep(sweep_job_cfg: DictConfig, logger: logging.Logger) -> None:
    """Initialize a new sweep or skip if it already exists."""
    # Initialize Cogweb client
    cogweb_client = CogwebClient.get_client(base_url=sweep_job_cfg.settings.sweep_server_uri)
    sweep_client = cogweb_client.sweep_client()

    # Check if sweep exists
    sweep_info = sweep_client.get_sweep(sweep_job_cfg.sweep_name)
    logger.info(f"Sweep exists: {sweep_info.exists}")

    if not sweep_info.exists:
        # Register the sweep in the centralized DB
        # Pass sweep_name as wandb_sweep_id for now to maintain API compatibility
        logger.info(f"Registering sweep {sweep_job_cfg.sweep_name} in the centralized DB")
        sweep_client.create_sweep(
            sweep_job_cfg.sweep_name, sweep_job_cfg.wandb.project, sweep_job_cfg.wandb.entity, sweep_job_cfg.sweep_name
        )  # TODO: Remove sweep_name from the sweep_client.create_sweep callin place of sweep_id
    else:
        logger.info(f"Found existing sweep {sweep_job_cfg.sweep_name} in the centralized DB")


def prepare_sweep_run(sweep_job_cfg: DictConfig, logger: logging.Logger) -> tuple[str, dict[str, Any]]:
    """Generate a new sweep run configuration with Protein suggestions and run name."""

    previous_observations = fetch_protein_observations_from_wandb(
        wandb_entity=sweep_job_cfg.wandb.entity,
        wandb_project=sweep_job_cfg.wandb.project,
        sweep_name=sweep_job_cfg.sweep_name,
        max_observations=sweep_job_cfg.settings.max_observations_to_load,
    )

    if hasattr(sweep_job_cfg, "schedule"):
        total_runs = 0
        phase_overrides = {}
        for phase_idx, phase in enumerate(sweep_job_cfg.schedule.phases):
            total_runs += phase.get("num_runs", 0)
            if total_runs > len(previous_observations) or phase_idx == len(sweep_job_cfg.schedule.phases) - 1:
                phase_overrides = phase.sweep
                break
        default_sweep_config = OmegaConf.create(OmegaConf.to_yaml(sweep_job_cfg.sweep))
        sweep_phase_config = OmegaConf.merge(default_sweep_config, phase_overrides)
        protein = MettaProtein(sweep_phase_config)  # type: ignore[arg-type]
    else:
        protein = MettaProtein(sweep_job_cfg.sweep)

    logger.info(f"Loaded {len(previous_observations)} previous observations from WandB")
    for obs in previous_observations:
        protein.observe(obs["suggestion"], obs["objective"], obs["cost"], obs["is_failure"])

    # Generate new suggestion
    protein_suggestion, _ = protein.suggest()

    # Get next available run name from central DB
    cogweb_client = CogwebClient.get_client(base_url=sweep_job_cfg.settings.sweep_server_uri)
    run_name = cogweb_client.sweep_client().get_next_run_id(sweep_job_cfg.sweep_name)
    logger.info(f"Got next run name from Cogweb DB: {run_name}")

    return run_name, protein_suggestion


def evaluate_sweep_rollout(
    train_job_cfg: DictConfig,
    protein_suggestion: dict[str, Any],
    metric: str,
    sweep_name: str,
) -> dict[str, Any]:
    """Evaluate a completed sweep rollout and record results to WandB (rank 0 only)."""
    logger.info(f"Evaluating run: {train_job_cfg.run} (rank 0 only)")

    with WandbContext(train_job_cfg.wandb, train_job_cfg, timeout=120) as wandb_run:
        if wandb_run is None:
            logger.error("Failed to initialize WandB context for evaluation")
            raise RuntimeError("WandB initialization failed during evaluation")

        # Run evaluation
        # side-effect: updates last policy metadata and adds policy to wandb sweep
        eval_results = _run_policy_evaluation(
            wandb_run,
            metric,
            sweep_name,
            train_job_cfg,
        )

        # Record evaluation results in WandB
        wandb_run.summary.update(eval_results)  # type: ignore[attr-defined]
        logger.info(f"Evaluation results: {eval_results}")

        suggestion_cost = eval_results["time.total"]
        suggestion_score = eval_results[metric]
        record_protein_observation_to_wandb(
            wandb_run,
            protein_suggestion,
            suggestion_score,
            suggestion_cost,
            False,
        )
        # build obs dictionary for logging purposes only
        obs_dict = {
            "suggestion": protein_suggestion,
            "score": suggestion_score,
            "cost": suggestion_cost,
            "is_failure": False,
        }
        logger.info(f"Recorded protein observation to WandB: {obs_dict}")

        # Save results for all ranks to read
        OmegaConf.save(
            {"eval_metric": suggestion_score, "total_time": suggestion_cost},
            f"{train_job_cfg.run_dir}/sweep_eval_results.yaml",
        )

    return eval_results


def _run_policy_evaluation(
    wandb_run: WandbRun,
    sweep_metric: str,
    sweep_name: str,
    train_job_cfg: DictConfig,
) -> dict[str, Any]:
    """Execute policy evaluation and compute metrics including costs."""
    simulation_suite_cfg = SimulationSuiteConfig(**OmegaConf.to_container(train_job_cfg.sim, resolve=True))  # type: ignore[arg-type]

    # Create env config
    env_cfg = create_system_config(train_job_cfg)

    if not wandb_run.name:
        raise ValueError("WandB run has no name")
    policy_store = get_policy_store_from_cfg(train_job_cfg, wandb_run)
    policy_pr = policy_store.policy_record("wandb://run/" + wandb_run.name, selector_type="latest")

    # Load the policy record directly using its wandb URI
    # This will download the artifact and give us a local path
    if not policy_pr.uri:
        raise ValueError(f"Policy record has no URI for run {wandb_run.name}")
    policy_pr = policy_store.load_from_uri(policy_pr.uri)

    eval = SimulationSuite(
        config=simulation_suite_cfg,
        policy_pr=policy_pr,
        policy_store=policy_store,
        device=train_job_cfg.device,
        vectorization=env_cfg.vectorization,
    )

    # Start evaluation
    eval_start_time = time.time()
    results = eval.simulate()
    eval_time = time.time() - eval_start_time
    eval_stats_db = EvalStatsDB.from_sim_stats_db(results.stats_db)
    eval_metric = eval_stats_db.get_average_metric_by_filter(sweep_metric, policy_pr)
    if eval_metric is None:
        logger.error("Evaluation returned no metric; check eval configuration and DB contents.")
        raise RuntimeError("Evaluation metric is None")

    # Get training stats from metadata if available
    # Note: training saves this as "total_train_time" not "train_time"
    train_time = policy_pr.metadata.get("total_train_time", 0)
    agent_step = policy_pr.metadata.get("agent_step", 0)
    epoch = policy_pr.metadata.get("epoch", 0)

    # Get hourly cost from environment variable (set by infrastructure)
    hourly_cost = 0.0
    hourly_cost_str = os.environ.get("METTA_HOURLY_COST")
    if hourly_cost_str:
        try:
            hourly_cost = float(hourly_cost_str)
            logger.debug(f"Using hourly cost from environment: ${hourly_cost:.4f}/hr")
        except (ValueError, TypeError):
            logger.warning(f"Could not parse METTA_HOURLY_COST: {hourly_cost_str}")

    # Calculate costs based on time and hourly rate
    training_cost = (hourly_cost * train_time / 3600.0) if hourly_cost > 0 else 0
    eval_cost = (hourly_cost * eval_time / 3600.0) if hourly_cost > 0 else 0
    total_cost = training_cost + eval_cost

    # Update sweep stats with evaluation results
    eval_results = {
        "train.agent_step": agent_step,
        "train.epoch": epoch,
        "time.train": train_time,
        "time.eval": eval_time,
        "time.total": train_time + eval_time,
        "cost.hourly": hourly_cost,
        "cost.training": training_cost,
        "cost.eval": eval_cost,
        "cost.total": total_cost,
        "uri": policy_pr.uri,
        "score": eval_metric,
        "score.metric": sweep_metric,
        sweep_metric: eval_metric,  # TODO: Should this be here?
    }

    # Update lineage stats
    for stat in [
        "train.agent_step",
        "train.epoch",
        "time.train",
        "time.eval",
        "time.total",
        "cost.training",
        "cost.eval",
        "cost.total",
    ]:
        eval_results["lineage." + stat] = eval_results[stat] + policy_pr.metadata.get("lineage." + stat, 0)

    # Update policy metadata
    policy_pr.metadata.update(
        {
            **eval_results,
            "training_run": wandb_run.name,
        }
    )

    # Add policy to wandb sweep
    policy_store.add_to_wandb_sweep(sweep_name, policy_pr)
    return eval_results
