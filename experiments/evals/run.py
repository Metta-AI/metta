import base64
import json

from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateRemoteJobTool


# Used by eval_task_worker.py
def eval(
    policy_uri: str, simulations_json_base64_path: str, output_file_path: str
) -> EvaluateRemoteJobTool:
    # Decode from base64 to avoid OmegaConf auto-parsing issues
    with open(simulations_json_base64_path, "rb") as f:
        simulations_json_base64 = f.read()
    simulations_json = base64.b64decode(simulations_json_base64).decode()
    simulations = [
        SimulationConfig.model_validate(sim) for sim in json.loads(simulations_json)
    ]
    return EvaluateRemoteJobTool(
        simulations=simulations,
        policy_uri=policy_uri,
        output_file_path=output_file_path,
    )
