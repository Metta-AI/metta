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
    policy_version_id: uuid.UUID,
    simulations: Sequence[SimulationRunConfig],
    stats_client: StatsClient,
    git_hash: str | None = None,
) -> EvalTaskRow:
    command_parts = [
        "uv run tools/run.py recipes.experiment.remote_eval.eval",
        f"policy_version_id={str(policy_version_id)}",
    ]
    command = " ".join(command_parts)
    request = TaskCreateRequest(
        command=command,
        data_file=SimulationList(simulations=simulations).model_dump(mode="json"),
        git_hash=git_hash,
    )
    return stats_client.create_eval_task(request)
