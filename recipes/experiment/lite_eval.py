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
    basic_env = arena.mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return LiteEvalTool(
        policy_specs=policy_specs,
        simulations=[
            SimulationRunConfig(
                env=basic_env,
                num_episodes=1,
                proportions=[1.0, 2.0],
                episode_tags={"name": "basic", "category": "arena"},
            ),
            SimulationRunConfig(
                env=arena.mettagrid(num_agents=6),
                num_episodes=2,
                episode_tags={"name": "combat", "category": "arena"},
            ),
        ],
    )
