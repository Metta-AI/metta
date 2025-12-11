import uuid
from typing import Sequence

from pydantic import BaseModel

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import EvalTaskRow
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest
from metta.sim.runner import SimulationRunConfig


class SimulationList(BaseModel):
    simulations: Sequence[SimulationRunConfig]


def evaluate_remotely(
    simulations: Sequence[SimulationRunConfig],
    stats_client: StatsClient,
    policy_version_id: uuid.UUID | str | None = None,
    policy_uri: str | None = None,
    git_hash: str | None = None,
    push_metrics_to_wandb: bool = True,
) -> EvalTaskRow:
    if not bool(policy_version_id) ^ bool(policy_uri):
        raise ValueError("Exactly one of policy_version_id or policy_uri must be provided")
    command_parts = [
        "uv run tools/run.py recipes.experiment.remote_eval.eval",
        f"push_metrics_to_wandb={str(push_metrics_to_wandb).lower()}",
    ]
    if policy_version_id:
        command_parts.append(f"policy_version_id={str(policy_version_id)}")
    if policy_uri:
        command_parts.append(f"policy_uri={policy_uri}")
    command = " ".join(command_parts)
    request = TaskCreateRequest(
        command=command,
        data_file=SimulationList(simulations=simulations).model_dump(mode="json"),
        git_hash=git_hash,
        attributes={"parallelism": len(simulations)},
    )
    return stats_client.create_eval_task(request)
