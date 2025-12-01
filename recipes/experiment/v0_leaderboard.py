from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Sequence

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.leaderboard_constants import (
    LEADERBOARD_CANDIDATE_COUNT_KEY,
    LEADERBOARD_LADYBUG_COUNT_KEY,
    LEADERBOARD_SCENARIO_KEY,
    LEADERBOARD_SCENARIO_KIND_KEY,
    LEADERBOARD_SIM_NAME_EPISODE_KEY,
    LEADERBOARD_THINKY_COUNT_KEY,
)
from metta.app_backend.routes.stats_routes import EpisodeQueryRequest
from metta.sim.runner import SimulationRunConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.multi_versioned_policy_eval import MultiPolicyVersionEvalTool
from metta.tools.play import PlayTool
from metta.tools.utils.auto_config import auto_stats_server_uri

logger = logging.getLogger(__name__)

NUM_COGS = 4

# Baseline policy UUIDs for VOR calculation
THINKY_UUID = "674fc022-5f1f-41e5-ab9e-551fa329b723"
LADYBUG_UUID = "5a491d05-7fb7-41a0-a250-fe476999edcd"


def _episode_tags(
    *,
    scenario_name: str,
    scenario_kind: str,
    candidate_count: int,
    thinky_count: int,
    ladybug_count: int,
) -> dict[str, str]:
    return {
        LEADERBOARD_SIM_NAME_EPISODE_KEY: scenario_name,
        LEADERBOARD_SCENARIO_KEY: scenario_name,
        LEADERBOARD_SCENARIO_KIND_KEY: scenario_kind,
        LEADERBOARD_CANDIDATE_COUNT_KEY: str(candidate_count),
        LEADERBOARD_THINKY_COUNT_KEY: str(thinky_count),
        LEADERBOARD_LADYBUG_COUNT_KEY: str(ladybug_count),
        "type": scenario_kind,
        "category": "machina1_open_world",
    }


@dataclass(frozen=True)
class LeaderboardScenario:
    candidate_count: int
    thinky_count: int
    ladybug_count: int
    scenario_kind: str

    @property
    def sim_name(self) -> str:
        return f"machina1-c{self.candidate_count}-t{self.thinky_count}-l{self.ladybug_count}"

    @property
    def proportions(self) -> list[float]:
        return [float(self.candidate_count), float(self.thinky_count), float(self.ladybug_count)]

    def episode_tags(self) -> dict[str, str]:
        return _episode_tags(
            scenario_name=self.sim_name,
            scenario_kind=self.scenario_kind,
            candidate_count=self.candidate_count,
            thinky_count=self.thinky_count,
            ladybug_count=self.ladybug_count,
        )


def _scenario_kind(candidate_count: int, thinky_count: int, ladybug_count: int) -> str:
    if candidate_count == NUM_COGS:
        return "candidate_self_play"
    if candidate_count == 0:
        if thinky_count == NUM_COGS:
            return "thinky_self_play"
        if ladybug_count == NUM_COGS:
            return "ladybug_self_play"
        return "thinky_ladybug_dual"
    return "candidate_mix"


def _generate_candidate_scenarios(num_cogs: int) -> list[LeaderboardScenario]:
    """Generate all scenarios involving the candidate policy (candidate_count > 0).

    Returns:
        All scenarios where candidate_count > 0, covering all combinations
        of candidate, thinky, and ladybug agents.
    """
    scenarios: list[LeaderboardScenario] = []
    for candidate_count in range(num_cogs, 0, -1):  # From num_cogs down to 1 (exclusive)
        max_thinky = num_cogs - candidate_count
        for thinky_count in range(max_thinky + 1):
            ladybug_count = num_cogs - candidate_count - thinky_count
            scenarios.append(
                LeaderboardScenario(
                    candidate_count=candidate_count,
                    thinky_count=thinky_count,
                    ladybug_count=ladybug_count,
                    scenario_kind=_scenario_kind(candidate_count, thinky_count, ladybug_count),
                )
            )
    return scenarios


def _generate_replacement_scenarios(num_cogs: int) -> list[LeaderboardScenario]:
    """Generate baseline replacement scenarios (candidate_count = 0).

    Returns:
        Scenarios for thinky_self_play, ladybug_self_play, and thinky_ladybug_dual.
    """
    return [
        LeaderboardScenario(0, num_cogs, 0, "thinky_self_play"),
        LeaderboardScenario(0, num_cogs // 2, num_cogs // 2, "thinky_ladybug_dual"),
        LeaderboardScenario(0, 0, num_cogs, "ladybug_self_play"),
    ]


def _generate_scenarios(num_cogs: int, include_replacement: bool = True) -> list[LeaderboardScenario]:
    """Generate all scenarios for leaderboard evaluation.

    Args:
        num_cogs: Number of agents
        include_replacement: If True, include replacement scenarios (candidate_count=0)

    Returns:
        List of all scenarios, optionally including replacement scenarios.
    """
    scenarios = _generate_candidate_scenarios(num_cogs)
    if include_replacement:
        scenarios.extend(_generate_replacement_scenarios(num_cogs))
    return scenarios


def simulations(
    num_episodes: int = 1, map_seed: int | None = None, minimal: bool = False
) -> Sequence[SimulationRunConfig]:
    """Generate simulation configs for leaderboard evaluation.

    Args:
        num_episodes: Number of episodes per simulation
        map_seed: Optional seed for map generation
        minimal: If True, only generate candidate scenarios (candidate_count > 0).
                 Replacement scenarios will be fetched from existing evaluations.
                 If False, generate all scenarios including replacement (legacy behavior).
    """
    # Setup Environment: Machina 1 Open World
    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = NUM_COGS
    map_builder = getattr(mission.site, "map_builder", None)
    if map_seed is not None and map_builder is not None and hasattr(map_builder, "seed"):
        map_builder.seed = map_seed
    env_config = mission.make_env()

    scenarios = _generate_scenarios(NUM_COGS, include_replacement=not minimal)
    return [
        SimulationRunConfig(
            env=env_config,
            num_episodes=num_episodes,
            proportions=scenario.proportions,
            episode_tags=scenario.episode_tags(),
        )
        for scenario in scenarios
    ]


def _verify_baseline_scenarios_exist(stats_client: StatsClient) -> None:
    """Verify that replacement scenarios exist in the database.

    Logs warnings if baseline scenarios are missing. The backend VOR calculation
    will automatically aggregate all replacement scenarios (candidate_count=0),
    including thinky_self_play, ladybug_self_play, and thinky_ladybug_dual.
    """
    try:
        # Check if any replacement scenarios exist
        episodes_response = stats_client.query_episodes(
            EpisodeQueryRequest(
                tag_filters={LEADERBOARD_CANDIDATE_COUNT_KEY: ["0"]},
                limit=1,
            )
        )

        if not episodes_response.episodes:
            logger.warning(
                "No replacement scenarios (candidate_count=0) found in database. "
                "VOR calculation requires baseline evaluations "
                "(thinky_self_play, ladybug_self_play, thinky_ladybug_dual)."
            )
            return

        # Optionally log specific baseline scores for visibility
        baseline_checks = [
            (THINKY_UUID, "thinky_self_play", "Thinky self-play"),
            (LADYBUG_UUID, "ladybug_self_play", "Ladybug self-play"),
            (THINKY_UUID, "thinky_ladybug_dual", "Thinky-Ladybug dual"),
        ]

        for policy_uuid, scenario_kind, label in baseline_checks:
            try:
                episodes = stats_client.query_episodes(
                    EpisodeQueryRequest(
                        primary_policy_version_ids=[uuid.UUID(policy_uuid)],
                        tag_filters={LEADERBOARD_SCENARIO_KIND_KEY: [scenario_kind]},
                        limit=None,
                    )
                ).episodes

                if episodes:
                    policy_uuid_obj = uuid.UUID(policy_uuid)
                    scores = [ep.avg_rewards[policy_uuid_obj] for ep in episodes if policy_uuid_obj in ep.avg_rewards]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        logger.info(f"{label} baseline score: {avg_score:.4f}")
            except Exception as e:
                logger.debug(f"Could not fetch {label} baseline score: {e}")

    except Exception as e:
        logger.error(f"Error verifying baseline scenarios: {e}")


# ./tools/run.py recipes.experiment.v0_leaderboard.evaluate policy_version_id=f32ca3a3-b6f0-479f-8105-27ce02b873cb
def evaluate(
    policy_version_id: str,
    result_file_path: str | None = None,
    stats_server_uri: str | None = None,
    seed: int = 50,
    use_baseline_scores: bool = True,
) -> MultiPolicyVersionEvalTool:
    """
    Run the V0 Leaderboard Evaluation.

    Compares the candidate policy (policy_version_id) against known baselines
    (Thinky and Ladybug) in the Machina 1 Open World environment.

    Args:
        policy_version_id: UUID of candidate policy to evaluate
        result_file_path: Optional path to save results JSON
        stats_server_uri: Optional stats server URI (auto-detected if not provided)
        seed: Random seed for map generation
        use_baseline_scores: If True, only run candidate scenarios (candidate_count > 0)
                            and fetch baseline scores from existing Thinky/Ladybug evaluations.
                            If False, run all scenarios with all policies (legacy behavior).
    """
    if (api_url := stats_server_uri or auto_stats_server_uri()) is None:
        raise ValueError("stats_server_uri is required")

    policy_version_ids = [policy_version_id, THINKY_UUID, LADYBUG_UUID]
    sim_configs = simulations(map_seed=seed, minimal=use_baseline_scores)

    if use_baseline_scores:
        stats_client = StatsClient.create(api_url)
        _verify_baseline_scenarios_exist(stats_client)

    tool = MultiPolicyVersionEvalTool(
        result_file_path=result_file_path or f"leaderboard_eval_{policy_version_id}.json",
        simulations=sim_configs,
        policy_version_ids=policy_version_ids,
        primary_policy_version_id=policy_version_id,
        verbose=True,
        stats_server_uri=api_url,
    )

    tool.system.seed = seed
    return tool


# ./tools/run.py recipes.experiment.v0_leaderboard.play policy_version_id=f32ca3a3-b6f0-479f-8105-27ce02b873cb
def play(policy_version_id: str | None = None, s3_path: str | None = None) -> PlayTool:
    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = NUM_COGS
    env_config = mission.make_env()
    return PlayTool(
        sim=SimulationConfig(env=env_config, suite="v0_leaderboard", name="play"),
        policy_version_id=policy_version_id,
        s3_path=s3_path,
    )
