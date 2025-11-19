import logging
from typing import Sequence

from metta.sim.runner import SimulationRunConfig
from metta.tools.multi_versioned_policy_eval import MultiPolicyVersionEvalTool
from metta.tools.utils.auto_config import auto_stats_server_uri
from recipes.experiment import arena

logger = logging.getLogger(__name__)


V0_LEADERBOARD_NAME_TAG_KEY = "v0-leaderboard-name"


def simulations() -> Sequence[SimulationRunConfig]:
    basic_env = arena.mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100
    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1
    return [
        SimulationRunConfig(
            env=basic_env,
            num_episodes=1,
            episode_tags={
                "v0-leaderboard-name": "arena-basic",
            },
        ),
        SimulationRunConfig(
            env=combat_env,
            num_episodes=1,
            episode_tags={
                "v0-leaderboard-name": "arena-combat",
            },
        ),
    ]


# ./tools/run.py recipes.experiment.v0_leaderboard_eval.run policy_version_id=127fb5d8-b530-4b27-9a60-9cf4e6be6365
def run(
    policy_version_id: str,
    result_file_path: str,
    eval_task_id: str,
    stats_server_uri: str | None = None,
) -> MultiPolicyVersionEvalTool:
    if (api_url := stats_server_uri or auto_stats_server_uri()) is None:
        raise ValueError("stats_server_uri is required")

    return MultiPolicyVersionEvalTool(
        result_file_path=result_file_path,
        simulations=simulations(),
        policy_version_ids=[
            policy_version_id,
            "4f00146e-7a14-4b5d-b15e-6068f1b82de6",  # nishad-test-baseline-scripted-3
            "3e9fca78-f179-47d8-bb56-63108a3ff7d3",  # nishad-test-baseline-scripted-2
        ],
        primary_policy_version_id=policy_version_id,
        verbose=True,
        stats_server_uri=api_url,
    )
