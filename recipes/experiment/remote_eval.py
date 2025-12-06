from metta.common.util.constants import PROD_STATS_SERVER_URI
from metta.sim.remote import SimulationList
from metta.tools.eval import EvalWithResultTool


# Used by eval_task_worker.py
def eval(
    policy_version_id: str,
    task_data_path: str,
    result_file_path: str,
) -> EvalWithResultTool:
    with open(task_data_path, "rb") as f:
        task_data = SimulationList.model_validate_json(f.read())

    return EvalWithResultTool(
        simulations=task_data.simulations,
        policy_uris=[f"metta://policy/{policy_version_id}"],
        stats_server_uri=PROD_STATS_SERVER_URI,
        push_metrics_to_wandb=True,
        result_file_path=result_file_path,
    )
