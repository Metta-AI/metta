import logging
import os
import time
from typing import Any

from omegaconf import DictConfig, OmegaConf

from cogweb.cogweb_client import CogwebClient
from metta.agent.policy_store import PolicyStore
from metta.common.wandb.wandb_context import WandbContext
from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.sweep.protein_metta import MettaProtein
from metta.sweep.protein_utils import apply_protein_suggestion, generate_protein_suggestion
from metta.sweep.wandb_utils import (
    create_wandb_run_for_sweep,
    fetch_protein_observations_from_wandb,
    record_protein_observation_to_wandb,
)

# 1 - Sweep Lifecycle Utils


def setup_sweep(sweep_job_cfg: DictConfig, logger: logging.Logger) -> str:
    """
    Setup a new sweep with the given name. If the sweep already exists, skip creation.
    Save the sweep configuration to sweep_dir/metadata.yaml.
    Returns the sweep name.
    """
    # Check if sweep already exists
    cogweb_client = CogwebClient.get_client(base_url=sweep_job_cfg.sweep_server_uri)
    sweep_client = cogweb_client.sweep_client()

    # Get sweep info
    sweep_info = sweep_client.get_sweep(sweep_job_cfg.sweep_name)

    # The sweep hasn't been registered with the centralized DB
    if not sweep_info.exists:
        logger.info(f"Creating sweep {sweep_job_cfg.sweep_name} in the centralized DB")

        # Create directories
        os.makedirs(sweep_job_cfg.runs_dir, exist_ok=True)

        # Register the sweep in the centralized DB
        # Pass sweep_name as wandb_sweep_id for now to maintain API compatibility
        logger.info(f"Registering sweep {sweep_job_cfg.sweep_name} in the centralized DB")
        sweep_client.create_sweep(
            sweep_job_cfg.sweep_name, sweep_job_cfg.wandb.project, sweep_job_cfg.wandb.entity, sweep_job_cfg.sweep_name
        )  # TODO: Remove sweep_name from the sweep_client.create_sweep callin place of sweep_id
    else:
        logger.info(f"Found existing sweep {sweep_job_cfg.sweep_name} in the centralized DB")

    # Save sweep metadata locally
    OmegaConf.save(
        {
            "sweep_name": sweep_job_cfg.sweep_name,
            "wandb_path": f"{sweep_job_cfg.wandb.entity}/{sweep_job_cfg.wandb.project}",
            "wandb_entity": sweep_job_cfg.wandb.entity,
            "wandb_project": sweep_job_cfg.wandb.project,
        },
        os.path.join(sweep_job_cfg.sweep_dir, "metadata.yaml"),
    )
    logger.info(f"Saved sweep metadata to {os.path.join(sweep_job_cfg.sweep_dir, 'metadata.yaml')}")

    return sweep_job_cfg.sweep_name


# Expects sweep_job configuration.
def prepare_sweep_run(
    sweep_job_cfg: DictConfig, logger: logging.Logger
) -> tuple[str, DictConfig, dict[str, Any], str | None]:
    """Prepare a sweep rollout - only runs on rank 0."""
    # Load previous protein suggestions from WandB
    protein = MettaProtein(sweep_job_cfg.sweep)
    previous_observations = fetch_protein_observations_from_wandb(
        wandb_entity=sweep_job_cfg.wandb.entity,
        wandb_project=sweep_job_cfg.wandb.project,
        sweep_name=sweep_job_cfg.sweep_name,
        max_observations=sweep_job_cfg.max_observations_to_load,
    )
    logger.info(f"Loaded {len(previous_observations)} previous observations from WandB")
    for obs in previous_observations:
        protein.observe(obs["suggestion"], obs["objective"], obs["cost"], obs["is_failure"])

    protein_suggestion = generate_protein_suggestion(sweep_job_cfg.trainer, protein)

    # Get next available run name from central DB
    # This enables parallel rollouts across workers
    cogweb_client = CogwebClient.get_client(base_url=sweep_job_cfg.sweep_server_uri)
    run_name = cogweb_client.sweep_client().get_next_run_id(sweep_job_cfg.sweep_name)
    logger.info(f"Got next run name from Cogweb DB: {run_name}")

    # Prepare the train job config
    train_job_cfg = _create_train_job_cfg_for_run(sweep_job_cfg, run_name)

    # Prepare the run directory
    os.makedirs(train_job_cfg.run_dir, exist_ok=True)

    # Create a new run in WandB
    # Returns wandb_run_id instead of creating dist_cfg.yaml file
    wandb_run_id = create_wandb_run_for_sweep(
        train_job_cfg=train_job_cfg,
        protein_suggestion=protein_suggestion,
        sweep_name=sweep_job_cfg.sweep_name,
    )
    logger.info(f"Created WandB run for sweep: {sweep_job_cfg.sweep_name} with run name: {run_name}")

    # Merge trainer overrides with protein suggestion and save to run_dir
    apply_protein_suggestion(train_job_cfg, protein_suggestion)
    trainer_cfg_override_path = os.path.join(train_job_cfg.run_dir, "train_config_overrides.yaml")
    OmegaConf.save(train_job_cfg, trainer_cfg_override_path)
    logger.info(f"Wrote trainer overrides to {trainer_cfg_override_path}")

    return run_name, train_job_cfg, protein_suggestion, wandb_run_id


def evaluate_rollout(
    train_job_cfg: DictConfig,
    protein_suggestion: dict[str, Any],
    metric: str,
    sweep_name: str,
    logger: Any,
) -> dict[str, Any]:
    """Evaluate the rollout - only runs on rank 0."""
    logger.info(f"Evaluating run: {train_job_cfg.run} (rank 0 only)")

    with WandbContext(train_job_cfg.wandb, train_job_cfg) as wandb_run:
        if wandb_run is None:
            logger.error("Failed to initialize WandB context for evaluation")
            raise RuntimeError("WandB initialization failed during evaluation")

        # Run evaluation
        # side-effect: updates last policy metadata and adds policy to wandb sweep
        eval_results = _evaluate_sweep_run(
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


# 2 - Private methods


def _evaluate_sweep_run(
    wandb_run: Any,
    sweep_metric: str,
    sweep_name: str,
    train_job_cfg: DictConfig,
) -> dict[str, Any]:
    simulation_suite_cfg = SimulationSuiteConfig(**OmegaConf.to_container(train_job_cfg.sim, resolve=True))  # type: ignore[arg-type]
    policy_store = PolicyStore(train_job_cfg, wandb_run)
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
        vectorization=train_job_cfg.vectorization,
    )

    # Start evaluation
    eval_start_time = time.time()
    results = eval.simulate()
    eval_time = time.time() - eval_start_time
    eval_stats_db = EvalStatsDB.from_sim_stats_db(results.stats_db)
    eval_metric = eval_stats_db.get_average_metric_by_filter(sweep_metric, policy_pr)

    # Get training stats from metadata if available
    train_time = policy_pr.metadata.get("train_time", 0)
    agent_step = policy_pr.metadata.get("agent_step", 0)
    epoch = policy_pr.metadata.get("epoch", 0)

    # Update sweep stats with evaluation results
    eval_results = {
        "train.agent_step": agent_step,
        "train.epoch": epoch,
        "time.train": train_time,
        "time.eval": eval_time,
        "time.total": train_time + eval_time,
        "uri": policy_pr.uri,
        "score": eval_metric,
        "score.metric": sweep_metric,
        sweep_metric: eval_metric,  # TODO: Should this be here?
    }

    # Update lineage stats
    for stat in ["train.agent_step", "train.epoch", "time.train", "time.eval", "time.total"]:
        eval_results["lineage." + stat] = eval_results[stat] + policy_pr.metadata.get("lineage." + stat, 0)

    # Update wandb summary
    wandb_run.summary.update(eval_results)

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


def _create_train_job_cfg_for_run(sweep_job_cfg: DictConfig, run_name: str) -> DictConfig:
    sweep_job_copy = OmegaConf.create(OmegaConf.to_container(sweep_job_cfg, resolve=False))

    # These must be set before resolving the config so that paths are properly interpolated
    sweep_job_copy.run = run_name
    sweep_job_copy.sweep_dir = f"{sweep_job_cfg.data_dir}/sweep/{sweep_job_cfg.sweep_name}"
    sweep_job_copy.data_dir = f"{sweep_job_copy.sweep_dir}/runs"
    resolved_sweep_job_copy = OmegaConf.create(OmegaConf.to_container(sweep_job_copy, resolve=True))
    train_job_cfg = resolved_sweep_job_copy.sweep_train_job
    assert isinstance(train_job_cfg, DictConfig)

    # Add sweep_name for evaluation
    train_job_cfg.sweep_name = sweep_job_cfg.sweep_name

    # TODO: This is a hack to get the wandb config to work.
    # Resolving early messes with the data_dir interpolation for wandb,
    # so we set it explicitly here.
    train_job_cfg.wandb.group = sweep_job_cfg.sweep_name
    train_job_cfg.wandb.name = run_name
    # train_job_cfg.wandb.data_dir = train_job_cfg.run_dir
    return train_job_cfg
