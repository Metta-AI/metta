import base64
import json

from metta.sim.simulation_config import SimulationConfig
from metta.tools.sim import SimTool


# Used by eval_task_worker.py
def eval(policy_uri: str, simulations_json_base64_path: str) -> SimTool:
    # Decode from base64 to avoid OmegaConf auto-parsing issues
    with open(simulations_json_base64_path, "rb") as f:
        simulations_json_base64 = f.read()
    simulations_json = base64.b64decode(simulations_json_base64).decode()
    simulations = [
        SimulationConfig.model_validate(sim) for sim in json.loads(simulations_json)
    ]
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
