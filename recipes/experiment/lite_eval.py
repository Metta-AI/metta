from metta.sim.runner import SimulationRunConfig
from metta.tools.lite_eval import LiteEvalTool
from mettagrid.policy.policy import PolicySpec
from recipes.experiment import arena


def run_lite_eval() -> LiteEvalTool:
    policy_specs = [
        PolicySpec(
            class_path="mettagrid.policy.random_agent.RandomMultiAgentPolicy",
            data_path=None,
        ),
        PolicySpec(
            class_path="noop",  # try shorthand
            data_path=None,
        ),
    ]
    return LiteEvalTool(
        simulations=[
            SimulationRunConfig(
                env=arena.mettagrid(),
                num_episodes=1,
                policy_specs=policy_specs,
                proportions=[1.0, 2.0],
            )
        ],
    )
