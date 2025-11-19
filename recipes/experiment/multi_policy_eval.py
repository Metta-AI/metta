from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.runner import SimulationRunConfig
from metta.tools.multi_policy_eval import MultiPolicyEvalTool
from mettagrid.policy.policy import PolicySpec
from recipes.experiment import arena


def run(policy_uri: str | None = None) -> MultiPolicyEvalTool:
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
    if policy_uri:
        policy_specs.append(CheckpointManager.policy_spec_from_uri(policy_uri, device="cpu"))
    basic_env = arena.mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return MultiPolicyEvalTool(
        policy_specs=policy_specs,
        simulations=[
            SimulationRunConfig(
                env=basic_env,
                num_episodes=1,
                proportions=[i + 1 for i in range(len(policy_specs))],
                episode_tags={"name": "basic", "category": "arena"},
            ),
            SimulationRunConfig(
                env=combat_env,
                num_episodes=2,
                episode_tags={"name": "combat", "category": "arena"},
            ),
        ],
    )
