from __future__ import annotations

from metta.sim.remote import SimulationList
import metta.tools as tools

# Used by eval_task_worker.py
def eval(
    task_data_path: str,
    result_file_path: str,
    policy_version_id: str | None = None,
    policy_uri: str | None = None,
) -> tools.EvalWithResultTool:
    with open(task_data_path, "rb") as f:
        task_data = SimulationList.model_validate_json(f.read())

    if not bool(policy_version_id) ^ bool(policy_uri):
        raise ValueError("Exactly one of policy_version_id or policy_uri must be provided")
    policy_uri = f"metta://policy/{policy_version_id}" if policy_version_id else policy_uri
    if not policy_uri:
        raise ValueError("policy_uri is required")

    return tools.EvalWithResultTool(
        simulations=task_data.simulations,
        policy_uris=[policy_uri],
        push_metrics_to_wandb=True,
        result_file_path=result_file_path,
    )
