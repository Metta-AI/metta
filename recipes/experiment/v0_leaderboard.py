import logging
from typing import Sequence

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from metta.app_backend.leaderboard_constants import LEADERBOARD_SIM_NAME_EPISODE_KEY
from metta.sim.runner import SimulationRunConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.multi_versioned_policy_eval import MultiPolicyVersionEvalTool
from metta.tools.play import PlayTool
from metta.tools.utils.auto_config import auto_stats_server_uri

logger = logging.getLogger(__name__)


# Baseline UUIDs
THINKY_UUID = "4f00146e-7a14-4b5d-b15e-6068f1b82de6"
LADYBUG_UUID = "3e9fca78-f179-47d8-bb56-63108a3ff7d3"


def simulations(num_episodes: int = 1) -> Sequence[SimulationRunConfig]:
    # Setup Environment: Machina 1 Open World
    num_cogs = 4
    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = num_cogs
    env_config = mission.make_env()

    # We have 3 policies in the list: [Candidate, Thinky, Ladybug]
    # Indices: 0=Candidate, 1=Thinky, 2=Ladybug

    # 1. Self-Play: Candidate vs Candidate
    # Proportions: All Candidate
    sp_config = SimulationRunConfig(
        env=env_config,
        num_episodes=num_episodes,
        proportions=[1.0, 0.0, 0.0],
        episode_tags={
            LEADERBOARD_SIM_NAME_EPISODE_KEY: "machina1-self-play",
            "type": "self_play",
        },
    )

    # 2. With Thinky: Candidate vs Thinky
    # Proportions: 50% Candidate, 50% Thinky
    thinky_config = SimulationRunConfig(
        env=env_config,
        num_episodes=num_episodes,
        proportions=[1.0, 1.0, 0.0],
        episode_tags={
            LEADERBOARD_SIM_NAME_EPISODE_KEY: "machina1-with-thinky",
            "type": "with_thinky",
        },
    )

    # 3. With Ladybug: Candidate vs Ladybug
    # Proportions: 50% Candidate, 50% Ladybug
    ladybug_config = SimulationRunConfig(
        env=env_config,
        num_episodes=num_episodes,
        proportions=[1.0, 0.0, 1.0],
        episode_tags={
            LEADERBOARD_SIM_NAME_EPISODE_KEY: "machina1-with-ladybug",
            "type": "with_ladybug",
        },
    )

    return [sp_config, thinky_config, ladybug_config]


# ./tools/run.py recipes.experiment.v0_leaderboard.evaluate policy_version_id=f32ca3a3-b6f0-479f-8105-27ce02b873cb
def evaluate(
    policy_version_id: str,
    result_file_path: str | None = None,
    stats_server_uri: str | None = None,
    seed: int = 50,
    eval_task_id: str | None = None,  # TODO: remove, this is not used
) -> MultiPolicyVersionEvalTool:
    """
    Run the V0 Leaderboard Evaluation.

    Compares the candidate policy (policy_version_id) against known baselines
    (Thinky and Ladybug) in the Machina 1 Open World environment.
    """
    if (api_url := stats_server_uri or auto_stats_server_uri()) is None:
        raise ValueError("stats_server_uri is required")

    policy_version_ids = [policy_version_id, THINKY_UUID, LADYBUG_UUID]

    tool = MultiPolicyVersionEvalTool(
        result_file_path=result_file_path or f"leaderboard_eval_{policy_version_id}.json",
        simulations=simulations(),
        policy_version_ids=policy_version_ids,
        primary_policy_version_id=policy_version_id,
        verbose=True,
        stats_server_uri=api_url,
    )

    # Set the seed
    tool.system.seed = seed

    return tool


# ./tools/run.py recipes.experiment.v0_leaderboard.play policy_version_id=f32ca3a3-b6f0-479f-8105-27ce02b873cb
def play(policy_version_id: str | None = None, s3_path: str | None = None) -> PlayTool:
    return PlayTool(
        sim=SimulationConfig(env=simulations()[0].env, suite="v0_leaderboard", name="play"),
        policy_version_id=policy_version_id,
        s3_path=s3_path,
    )
