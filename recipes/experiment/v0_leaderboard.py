from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Sequence

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.leaderboard_constants import (
    LADYBUG_UUID,
    LEADERBOARD_CANDIDATE_COUNT_KEY,
    LEADERBOARD_LADYBUG_COUNT_KEY,
    LEADERBOARD_SCENARIO_KEY,
    LEADERBOARD_SCENARIO_KIND_KEY,
    LEADERBOARD_SIM_NAME_EPISODE_KEY,
    LEADERBOARD_THINKY_COUNT_KEY,
    THINKY_UUID,
)
from metta.app_backend.routes.stats_routes import EpisodeQueryRequest
from metta.sim.runner import SimulationRunConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluatePolicyVersionTool
from metta.tools.multi_versioned_policy_eval import MultiPolicyVersionEvalTool
from metta.tools.play import PlayTool
from metta.tools.utils.auto_config import auto_stats_server_uri

logger = logging.getLogger(__name__)

NUM_COGS = 4


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


def _scenario_kind(cand: int, thinky: int, lady: int) -> str:
    if cand == NUM_COGS:
        return "candidate_self_play"
    if cand == 0:
        if thinky == NUM_COGS:
            return "thinky_self_play"
        if lady == NUM_COGS:
            return "ladybug_self_play"
        return "thinky_ladybug_dual"
    return "candidate_mix"


def _generate_scenarios(
    num_cogs: int, *, include_candidate: bool = True, include_replacement: bool = True
) -> list[LeaderboardScenario]:
    """Generate scenario combinations for leaderboard evaluation."""
    scenarios: list[LeaderboardScenario] = []
    for cand in range(num_cogs, -1, -1):
        if cand > 0 and not include_candidate:
            continue
        if cand == 0 and not include_replacement:
            continue
        for thinky in range(num_cogs - cand, -1, -1):
            lady = num_cogs - cand - thinky
            scenarios.append(
                LeaderboardScenario(
                    candidate_count=cand,
                    thinky_count=thinky,
                    ladybug_count=lady,
                    scenario_kind=_scenario_kind(cand, thinky, lady),
                )
            )
    return scenarios


def _make_env(seed: int | None = None):
    """Create the Machina 1 Open World environment."""
    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = NUM_COGS
    if seed and (mb := getattr(mission.site, "map_builder", None)) and hasattr(mb, "seed"):
        mb.seed = seed
    return mission.make_env()


def simulations(
    num_episodes: int = 1, map_seed: int | None = None, minimal: bool = False
) -> Sequence[SimulationRunConfig]:
    """Generate simulation configs for leaderboard evaluation."""
    env_config = _make_env(map_seed)
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


# ./tools/run.py recipes.experiment.v0_leaderboard.evaluate policy_version_id=f32ca3a3-b6f0-479f-8105-27ce02b873cb
# Or using metta:// URI:
# ./tools/run.py recipes.experiment.v0_leaderboard.evaluate policy_uri=metta://policy/f32ca3a3-b6f0-479f-8105-27ce02b873cb
def evaluate(
    policy_version_id: str | None = None,
    policy_uri: str | None = None,
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
        policy_version_id: Observatory policy version UUID (legacy parameter)
        policy_uri: Policy URI (e.g., metta://policy/<uuid>, s3://..., file://...)
        result_file_path: Optional path to save results JSON
        stats_server_uri: Optional stats server URI (auto-detected if not provided)
        seed: Random seed for map generation
        use_baseline_scores: If True, only run candidate scenarios (candidate_count > 0)
                            and fetch baseline scores from existing Thinky/Ladybug evaluations.
                            If False, run all scenarios with all policies (legacy behavior).
    """
    # Support both legacy policy_version_id and new policy_uri parameter
    if policy_uri and policy_uri.startswith("metta://policy/"):
        policy_version_id = policy_uri.split("/")[-1]
    elif not policy_version_id:
        raise ValueError("Either policy_version_id or policy_uri is required")

    if (api_url := stats_server_uri or auto_stats_server_uri()) is None:
        raise ValueError("stats_server_uri is required")

    policy_version_ids = [policy_version_id, THINKY_UUID, LADYBUG_UUID]
    sim_configs = simulations(map_seed=seed, minimal=use_baseline_scores)

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


# ./tools/run.py recipes.experiment.v0_leaderboard.evaluate_baselines
def evaluate_baselines(
    stats_server_uri: str | None = None,
    seed: int = 50,
    force: bool = False,
) -> MultiPolicyVersionEvalTool | None:
    """Run replacement baseline scenarios if needed. Returns None if already exist."""
    if (api_url := stats_server_uri or auto_stats_server_uri()) is None:
        raise ValueError("stats_server_uri is required")

    scenarios = _generate_scenarios(NUM_COGS, include_candidate=False)

    if not force:
        stats_client = StatsClient.create(api_url)
        missing = _find_missing_baseline_scenarios(stats_client, scenarios)
        if not missing:
            logger.info("All baseline episodes already exist, skipping")
            return None
        logger.info(f"Missing {len(missing)} baseline scenarios: {[s.sim_name for s in missing]}")
        scenarios = missing

    env_config = _make_env(seed)
    sim_configs = [
        SimulationRunConfig(
            env=env_config,
            num_episodes=1,
            proportions=[float(s.thinky_count), float(s.ladybug_count)],
            episode_tags=s.episode_tags(),
        )
        for s in scenarios
    ]

    tool = MultiPolicyVersionEvalTool(
        result_file_path=f"leaderboard_baselines_{seed}.json",
        simulations=sim_configs,
        policy_version_ids=[THINKY_UUID, LADYBUG_UUID],
        primary_policy_version_id=THINKY_UUID,
        verbose=True,
        stats_server_uri=api_url,
    )
    tool.system.seed = seed
    return tool


def _find_missing_baseline_scenarios(
    stats_client: StatsClient,
    scenarios: list[LeaderboardScenario],
) -> list[LeaderboardScenario]:
    """Check which baseline scenarios are missing from the database."""
    # Query episodes with candidate_count=0 for thinky/ladybug
    response = stats_client.query_episodes(
        EpisodeQueryRequest(
            primary_policy_version_ids=[uuid.UUID(THINKY_UUID), uuid.UUID(LADYBUG_UUID)],
            tag_filters={LEADERBOARD_CANDIDATE_COUNT_KEY: ["0"]},
            limit=1000,
        )
    )

    # Extract existing scenario names from episode tags
    existing_scenario_names: set[str] = set()
    for episode in response.episodes:
        if episode.tags:
            scenario_name = episode.tags.get(LEADERBOARD_SCENARIO_KEY)
            if scenario_name:
                existing_scenario_names.add(scenario_name)

    # Return scenarios that don't have existing episodes
    return [s for s in scenarios if s.sim_name not in existing_scenario_names]


# ./tools/run.py recipes.experiment.v0_leaderboard.play policy_version_id=f32ca3a3-b6f0-479f-8105-27ce02b873cb
def play(policy_version_id: str | None = None, s3_path: str | None = None) -> PlayTool:
    return PlayTool(
        sim=SimulationConfig(env=_make_env(), suite="v0_leaderboard", name="play"),
        policy_version_id=policy_version_id,
        s3_path=s3_path,
    )


# This is similar to what evaluate_remote() ultimately calls within remote_eval_worker
# ./tools/run.py recipes.experiment.v0_leaderboard.test_remote_eval
#   policy_version_id=99810f21-d73d-4e72-8082-70516f2b6b2a
def test_remote_eval(policy_version_id: str) -> EvaluatePolicyVersionTool:
    sims = simulations()
    for sim in sims:
        sim.proportions = [1.0]
    return EvaluatePolicyVersionTool(
        simulations=sims,
        policy_version_id=policy_version_id,
        stats_server_uri=auto_stats_server_uri(),
    )
