import json

from metta.common.util.constants import PROD_STATS_SERVER_URI
from metta.sim.runner import SimulationRunConfig
from metta.tools.eval import EvalWithResultTool


# Used by eval_task_worker.py
def eval(
    policy_version_id: str,
    task_data_path: str,
    result_file_path: str,
) -> EvalWithResultTool:
    with open(task_data_path, "rb") as f:
        simulations_json = f.read()
    sim_json_array = json.loads(simulations_json)["simulations"]
    simulations = [SimulationRunConfig.model_validate(sim) for sim in sim_json_array]

    return EvalWithResultTool(
        simulations=simulations,
        policy_uris=[f"metta://policy/{policy_version_id}"],
        stats_server_uri=PROD_STATS_SERVER_URI,
        push_metrics_to_wandb=True,
        result_file_path=result_file_path,
    )
