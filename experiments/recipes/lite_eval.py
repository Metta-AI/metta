from experiments.recipes.arena import simulations
from metta.tools.lite_eval import LiteEvalTool
from mettagrid.policy.policy import PolicySpec


def run_lite_eval() -> LiteEvalTool:
    return LiteEvalTool(
        simulations=simulations(),
        policies=[
            PolicySpec(
                class_path="mettagrid.policy.random.RandomMultiAgentPolicy",
                data_path=None,
            ),
            PolicySpec(
                class_path="noop",  # try shorthand
                data_path=None,
            ),
        ],
        proportions=[1.0, 2.0],
    )
