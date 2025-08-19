from experiments.evals.registry import get_eval_suite
from metta.common.util.constants import SOFTMAX_S3_BASE
from metta.rl.system_config import SystemConfig
from metta.tools.sim import SimTool


# Used by eval_task_worker.py
def eval(policy_uri: str, sim_suite: str) -> SimTool:
    simulations = get_eval_suite(sim_suite)

    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
        replay_dir=f"{SOFTMAX_S3_BASE}/replays",
        system=SystemConfig(
            device="cpu",
            vectorization="serial",
        ),
    )
