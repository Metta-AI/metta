from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
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
from metta.sim.runner import SimulationRunConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool, EvalWithResultTool
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


# ./tools/run.py recipes.experiment.v0_leaderboard.evaluate policy_uri=metta://policy/f32ca3a3-b6f0-479f-8105-27ce02b873cb
def evaluate(
    policy_uri: str | None = None,
    policy_version_id: str | None = None,
    seed: int = 50,
    result_file_path: str | None = None,
) -> EvalWithResultTool:
    if policy_version_id and not policy_uri:
        policy_uri = f"metta://policy/{policy_version_id}"
    if not policy_uri:
        raise ValueError("policy_uri is required")
    policy_uris = [
        policy_uri,
        f"metta://policy/{THINKY_UUID}",
        f"metta://policy/{LADYBUG_UUID}",
    ]

    tool = EvalWithResultTool(
        simulations=simulations(map_seed=seed, minimal=True),
        policy_uris=policy_uris,
        verbose=True,
        result_file_path=result_file_path or f"leaderboard_eval_{policy_version_id}.json",
    )
    tool.system.seed = seed
    return tool


# ./tools/run.py recipes.experiment.v0_leaderboard.play policy_version_id=f32ca3a3-b6f0-479f-8105-27ce02b873cb
def play(policy_version_id: str | None = None, s3_path: str | None = None) -> PlayTool:
    return PlayTool(
        sim=SimulationConfig(env=_make_env(), suite="v0_leaderboard", name="play"),
        policy_version_id=policy_version_id,
        s3_path=s3_path,
    )


def baseline_simulations(num_episodes: int = 1, map_seed: int | None = None) -> Sequence[SimulationRunConfig]:
    """Generate simulation configs for baseline (c0) evaluation only.

    These scenarios have candidate_count=0 and are used to establish the replacement baseline.
    """
    env_config = _make_env(map_seed)
    scenarios = _generate_scenarios(NUM_COGS, include_candidate=False, include_replacement=True)
    return [
        SimulationRunConfig(
            env=env_config,
            num_episodes=num_episodes,
            proportions=scenario.proportions,
            episode_tags=scenario.episode_tags(),
        )
        for scenario in scenarios
    ]


# ./tools/run.py recipes.experiment.v0_leaderboard.evaluate_baseline seed=50
def evaluate_baseline(
    seed: int = 50,
    num_episodes: int = 3,
    result_file_path: str | None = None,
) -> EvalWithResultTool:
    """Evaluate the baseline policies (thinky + ladybug) in c0 scenarios.

    This establishes the replacement baseline for VOR calculations.
    Should be run periodically or when no c0 baseline data exists.
    """
    policy_uris = [
        f"metta://policy/{THINKY_UUID}",
        f"metta://policy/{LADYBUG_UUID}",
    ]

    tool = EvalWithResultTool(
        simulations=baseline_simulations(num_episodes=num_episodes, map_seed=seed),
        policy_uris=policy_uris,
        verbose=True,
        result_file_path=result_file_path or f"baseline_eval_{seed}.json",
    )
    tool.system.seed = seed
    return tool


# This is similar to what evaluate_remote() ultimately calls within remote_eval_worker
# ./tools/run.py recipes.experiment.v0_leaderboard.test_remote_eval
#   policy_version_id=99810f21-d73d-4e72-8082-70516f2b6b2a
def test_remote_eval(policy_version_id: str) -> EvaluateTool:
    sims = simulations()
    for sim in sims:
        sim.proportions = [1.0]
    return EvaluateTool(
        simulations=sims,
        policy_uris=[f"metta://policy/{policy_version_id}"],
        stats_server_uri=auto_stats_server_uri(),
        max_workers=3,
    )
