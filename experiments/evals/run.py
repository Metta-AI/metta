from metta.tools.sim import SimTool

from experiments.evals.registry import get_eval_suite


# Used by eval_task_worker.py
def eval(policy_uri: str, sim_suite: str) -> SimTool:
    simulations = get_eval_suite(sim_suite)

    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
