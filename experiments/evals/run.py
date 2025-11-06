import base64
import json

import metta.sim.simulation_config
import metta.tools.eval


# Used by eval_task_worker.py
def eval(
    policy_uri: str, simulations_json_base64_path: str, job_result_file_path: str
) -> metta.tools.eval.EvaluateRemoteJobTool:
    # Decode from base64 to avoid OmegaConf auto-parsing issues
    with open(simulations_json_base64_path, "rb") as f:
        simulations_json_base64 = f.read()
    simulations_json = base64.b64decode(simulations_json_base64).decode()
    simulations = [
        metta.sim.simulation_config.SimulationConfig.model_validate(sim)
        for sim in json.loads(simulations_json)
    ]

    return metta.tools.eval.EvaluateRemoteJobTool(
        simulations=simulations,
        policy_uri=policy_uri,
        job_result_file_path=job_result_file_path,
    )
