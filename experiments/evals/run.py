import json

from metta.sim.simulation_config import SimulationConfig
from metta.tools.sim import SimTool


# Used by eval_task_worker.py
def eval(policy_uri: str, simulations_json: str) -> SimTool:
    simulations = [
        SimulationConfig.model_validate_json(sim)
        for sim in json.loads(simulations_json)
    ]
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
