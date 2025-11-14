from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskResponse
from metta.common.wandb.context import WandbRun
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation_config import SimulationConfig
from metta.sim.utils import get_or_create_policy_ids

logger = logging.getLogger(__name__)


# Avoid circular import: evaluator.py → evaluate.py → evaluator.py
# EvaluatorConfig is only used as a type hint, never instantiated here
if TYPE_CHECKING:
    from metta.rl.training.evaluator import EvaluatorConfig


def evaluate_policy_remote_with_checkpoint_manager(
    policy_uri: str,
    simulations: list[SimulationConfig],
    stats_epoch_id: uuid.UUID | None,
    stats_client: StatsClient | None,
    wandb_run: WandbRun | None,
    evaluation_cfg: EvaluatorConfig | None,
) -> TaskResponse | None:
    """Create a remote evaluation task using a policy URI."""
    if not (wandb_run and stats_client and policy_uri):
        logger.warning("Remote evaluation requires wandb_run, stats_client, and policy_uri")
        return None

    # Normalize the policy URI
    normalized_uri = CheckpointManager.normalize_uri(policy_uri)

    # Process policy registration using the new format
    stats_server_policy_id = get_or_create_policy_ids(
        stats_client,
        [(normalized_uri, wandb_run.notes)],  # New format: (uri, description)
        stats_epoch_id,
    ).get(normalized_uri)

    if not stats_server_policy_id:
        logger.warning(f"Remote evaluation: failed to get or register policy ID for {normalized_uri}")
        return None

    # Create evaluation task
    task = stats_client.create_task(
        TaskCreateRequest(
            policy_id=stats_server_policy_id,
            sim_suite=simulations[0].name,
            attributes={
                "git_hash": (evaluation_cfg and evaluation_cfg.git_hash),
                "simulations": [sim.model_dump() for sim in simulations],
            },
        )
    )

    logger.info(f"Policy evaluator: created task {task.id} for {normalized_uri} on {simulations[0].name}")

    return task
