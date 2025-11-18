import uuid
from typing import Sequence

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import EvalTaskRow
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest
from metta.sim.runner import SimulationRunConfig


def evaluate_remotely(
    policy_version_id: uuid.UUID,
    simulations: Sequence[SimulationRunConfig],
    stats_client: StatsClient,
    git_hash: str | None = None,
) -> EvalTaskRow:
    simulations_json = [sim.model_dump(mode="json") for sim in simulations]
    command_parts = [
        "uv run tools/run.py recipes.experiment.remote_eval.eval",
        f"policy_version_id={str(policy_version_id)}",
        f"stats_server_uri={stats_client._backend_url}",
    ]
    command = " ".join(command_parts)
    request = TaskCreateRequest(
        command=command,
        data_file=dict(simulations=simulations_json),
        git_hash=git_hash,
    )
    return stats_client.create_eval_task(request)
