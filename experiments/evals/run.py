import base64
import json

from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvalTool


# Used by eval_task_worker.py
def eval(policy_uri: str, simulations_json_base64: str) -> EvalTool:
    # Decode from base64 to avoid OmegaConf auto-parsing issues
    simulations_json = base64.b64decode(simulations_json_base64).decode()
    simulations = [
        SimulationConfig.model_validate(sim) for sim in json.loads(simulations_json)
    ]
    return EvalTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
