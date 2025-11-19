import logging
from typing import Sequence

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from metta.sim.runner import SimulationRunConfig
from metta.tools.multi_versioned_policy_eval import MultiPolicyVersionEvalTool
from metta.tools.utils.auto_config import auto_stats_server_uri

logger = logging.getLogger(__name__)

V0_LEADERBOARD_NAME_TAG_KEY = "v0-leaderboard-name"

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
            V0_LEADERBOARD_NAME_TAG_KEY: "machina1-self-play",
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
            V0_LEADERBOARD_NAME_TAG_KEY: "machina1-with-thinky",
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
            V0_LEADERBOARD_NAME_TAG_KEY: "machina1-with-ladybug",
            "type": "with_ladybug",
        },
    )

    return [sp_config, thinky_config, ladybug_config]


def run(
    policy_version_id: str,
    result_file_path: str,
    eval_task_id: str,
    stats_server_uri: str | None = None,
    seed: int = 50,
) -> MultiPolicyVersionEvalTool:
    """
    Run the V0 Leaderboard Evaluation.

    Compares the candidate policy (policy_version_id) against known baselines
    (Thinky and Ladybug) in the Machina 1 Open World environment.
    """
    if (api_url := stats_server_uri or auto_stats_server_uri()) is None:
        raise ValueError("stats_server_uri is required")

    # Construct the list of policy IDs to be fetched
    policy_version_ids = [
        policy_version_id,  # The Candidate (Index 0)
        THINKY_UUID,  # Thinky (Index 1)
        LADYBUG_UUID,  # Ladybug (Index 2)
    ]

    tool = MultiPolicyVersionEvalTool(
        result_file_path=result_file_path,
        simulations=simulations(),
        policy_version_ids=policy_version_ids,
        primary_policy_version_id=policy_version_id,
        verbose=True,
        stats_server_uri=api_url,
    )

    # Set the seed
    tool.system.seed = seed

    return tool
