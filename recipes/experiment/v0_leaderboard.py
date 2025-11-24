from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from metta.app_backend.leaderboard_constants import LEADERBOARD_SIM_NAME_EPISODE_KEY
from metta.common.tool.tool import ToolResult
from metta.sim.runner import SimulationRunConfig, SimulationRunResult
from metta.sim.simulation_config import SimulationConfig
from metta.tools.multi_versioned_policy_eval import MultiPolicyVersionEvalTool
from metta.tools.play import PlayTool
from metta.tools.utils.auto_config import auto_stats_server_uri
from mettagrid.policy.policy import PolicySpec

logger = logging.getLogger(__name__)

NUM_COGS = 4

LEADERBOARD_SCENARIO_KEY = "leaderboard-scenario-key"
LEADERBOARD_SCENARIO_KIND_KEY = "leaderboard-scenario-kind"
LEADERBOARD_CANDIDATE_COUNT_KEY = "leaderboard-candidate-count"
LEADERBOARD_THINKY_COUNT_KEY = "leaderboard-thinky-count"
LEADERBOARD_LADYBUG_COUNT_KEY = "leaderboard-ladybug-count"

# Whatever `thinky` resolves to in the policy shorthand registry.
# Likely cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy.
THINKY_UUID = "674fc022-5f1f-41e5-ab9e-551fa329b723"

# Whatever `ladybug` resolves to in the policy shorthand registry.
# Likely cogames.policy.scripted_agent.unclipping_agent.UnclippingPolicy
LADYBUG_UUID = "5a491d05-7fb7-41a0-a250-fe476999edcd"


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = math.inf
        self._max = -math.inf

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self._mean
        self._mean += delta / self.count
        delta2 = value - self._mean
        self._m2 += delta * delta2
        if self.count == 1:
            self._min = value
            self._max = value
        else:
            self._min = min(self._min, value)
            self._max = max(self._max, value)

    @property
    def mean(self) -> float | None:
        return None if self.count == 0 else self._mean

    @property
    def variance(self) -> float | None:
        if self.count == 0:
            return None
        if self.count == 1:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def std_dev(self) -> float | None:
        variance = self.variance
        return None if variance is None else math.sqrt(variance)

    @property
    def min(self) -> float | None:
        return None if self.count == 0 else self._min

    @property
    def max(self) -> float | None:
        return None if self.count == 0 else self._max


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
        return {
            LEADERBOARD_SIM_NAME_EPISODE_KEY: self.sim_name,
            LEADERBOARD_SCENARIO_KEY: self.sim_name,
            LEADERBOARD_SCENARIO_KIND_KEY: self.scenario_kind,
            LEADERBOARD_CANDIDATE_COUNT_KEY: str(self.candidate_count),
            LEADERBOARD_THINKY_COUNT_KEY: str(self.thinky_count),
            LEADERBOARD_LADYBUG_COUNT_KEY: str(self.ladybug_count),
            "type": self.scenario_kind,
            "category": "machina1_open_world",
        }


class ScenarioSummary(BaseModel):
    scenario_name: str
    scenario_kind: str
    candidate_count: int
    thinky_count: int
    ladybug_count: int
    candidate_mean: float | None
    candidate_std: float | None
    candidate_samples: int
    replacement_mean: float | None = None
    replacement_std: float | None = None
    replacement_samples: int = 0


class CandidateCountSummary(BaseModel):
    candidate_count: int
    mean: float | None
    variance: float | None
    std_dev: float | None
    min_value: float | None
    max_value: float | None
    samples: int


class GraphPoint(BaseModel):
    candidate_count: int
    mean: float | None
    lower: float | None
    upper: float | None
    std_dev: float | None


class ValueOverReplacementSummary(BaseModel):
    policy_version_id: str
    scenario_summaries: list[ScenarioSummary]
    candidate_counts: list[CandidateCountSummary]
    replacement_summary: CandidateCountSummary | None
    value_over_replacement: dict[str, float | None]
    value_over_replacement_std: dict[str, float | None] = {}
    graph_points: list[GraphPoint]


@dataclass
class ScenarioAccumulator:
    candidate_count: int
    thinky_count: int
    ladybug_count: int
    scenario_kind: str
    candidate_stats: RunningStats = field(default_factory=RunningStats)
    replacement_stats: RunningStats = field(default_factory=RunningStats)


def _make_env(num_cogs: int, map_seed: int | None):
    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = num_cogs
    map_builder = getattr(mission.site, "map_builder", None)
    if map_seed is not None and map_builder is not None:
        map_builder.seed = map_seed
    return mission.make_env()


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


def _generate_scenarios(num_cogs: int) -> list[LeaderboardScenario]:
    scenarios: list[LeaderboardScenario] = []
    for candidate_count in range(num_cogs, -1, -1):
        if candidate_count == 0:
            scenarios.extend(
                [
                    LeaderboardScenario(0, num_cogs, 0, "thinky_self_play"),
                    LeaderboardScenario(0, num_cogs // 2, num_cogs // 2, "thinky_ladybug_dual"),
                    LeaderboardScenario(0, 0, num_cogs, "ladybug_self_play"),
                ]
            )
            continue
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


def simulations(num_episodes: int = 1, map_seed: int | None = None) -> Sequence[SimulationRunConfig]:
    env_config = _make_env(num_cogs=NUM_COGS, map_seed=map_seed)
    configs: list[SimulationRunConfig] = []
    for scenario in _generate_scenarios(NUM_COGS):
        configs.append(
            SimulationRunConfig(
                env=env_config,
                num_episodes=num_episodes,
                proportions=scenario.proportions,
                episode_tags=scenario.episode_tags(),
            )
        )
    return configs


class LeaderboardEvalTool(MultiPolicyVersionEvalTool):
    value_over_replacement_path: str

    def after_rollout(
        self, *, rollout_results: list[SimulationRunResult], policy_specs: Sequence[PolicySpec]
    ) -> ToolResult | None:
        del policy_specs  # Leaderboard summary only depends on recorded results.
        summary = build_value_over_replacement_summary(
            policy_version_id=self.primary_policy_version_id,
            rollout_results=rollout_results,
        )
        output_path = Path(self.value_over_replacement_path)
        output_path.write_text(summary.model_dump_json(indent=2))
        logger.info("Value over replacement summary written to %s", output_path)
        return ToolResult(result="success", output_uri=str(output_path))


def _iter_policy_rewards(sim_result: SimulationRunResult, policy_index: int) -> list[float]:
    rewards: list[float] = []
    for episode in sim_result.results.episodes:
        assignments = episode.assignments.tolist()
        per_agent = episode.rewards.tolist()
        rewards.extend(float(reward) for idx, reward in enumerate(per_agent) if assignments[idx] == policy_index)
    return rewards


def _iter_all_rewards(sim_result: SimulationRunResult) -> list[float]:
    rewards: list[float] = []
    for episode in sim_result.results.episodes:
        rewards.extend(float(reward) for reward in episode.rewards.tolist())
    return rewards


def _variance_of_mean(summary: CandidateCountSummary) -> float | None:
    if summary.samples <= 0 or summary.variance is None:
        return None
    return summary.variance / summary.samples


def build_value_over_replacement_summary(
    *, policy_version_id: str, rollout_results: Sequence[SimulationRunResult]
) -> ValueOverReplacementSummary:
    scenario_stats: dict[str, ScenarioAccumulator] = {}
    candidate_count_stats: dict[int, RunningStats] = defaultdict(RunningStats)

    for result in rollout_results:
        tags = result.run.episode_tags
        scenario_name = tags.get(LEADERBOARD_SCENARIO_KEY)
        if scenario_name is None:
            raise ValueError("leaderboard scenario key missing from episode tags")
        candidate_count = int(tags.get(LEADERBOARD_CANDIDATE_COUNT_KEY, "0"))
        thinky_count = int(tags.get(LEADERBOARD_THINKY_COUNT_KEY, "0"))
        ladybug_count = int(tags.get(LEADERBOARD_LADYBUG_COUNT_KEY, "0"))
        scenario_kind = tags.get(LEADERBOARD_SCENARIO_KIND_KEY, "unknown")

        entry = scenario_stats.setdefault(
            scenario_name,
            ScenarioAccumulator(
                candidate_count=candidate_count,
                thinky_count=thinky_count,
                ladybug_count=ladybug_count,
                scenario_kind=scenario_kind,
            ),
        )

        if candidate_count > 0:
            for reward in _iter_policy_rewards(result, policy_index=0):
                entry.candidate_stats.update(reward)
                candidate_count_stats[candidate_count].update(reward)
        else:
            for reward in _iter_all_rewards(result):
                entry.replacement_stats.update(reward)
                candidate_count_stats[0].update(reward)

    scenario_summaries: list[ScenarioSummary] = []
    for scenario_name, payload in sorted(scenario_stats.items()):
        candidate_stats = payload.candidate_stats
        replacement_stats = payload.replacement_stats
        candidate_count = payload.candidate_count
        scenario_summaries.append(
            ScenarioSummary(
                scenario_name=scenario_name,
                scenario_kind=payload.scenario_kind,
                candidate_count=candidate_count,
                thinky_count=payload.thinky_count,
                ladybug_count=payload.ladybug_count,
                candidate_mean=candidate_stats.mean,
                candidate_std=candidate_stats.std_dev,
                candidate_samples=candidate_stats.count,
                replacement_mean=replacement_stats.mean if candidate_count == 0 else None,
                replacement_std=replacement_stats.std_dev if candidate_count == 0 else None,
                replacement_samples=replacement_stats.count if candidate_count == 0 else 0,
            )
        )

    candidate_count_summaries: list[CandidateCountSummary] = []
    for candidate_count in sorted(candidate_count_stats.keys()):
        stats = candidate_count_stats[candidate_count]
        candidate_count_summaries.append(
            CandidateCountSummary(
                candidate_count=candidate_count,
                mean=stats.mean,
                variance=stats.variance,
                std_dev=stats.std_dev,
                min_value=stats.min,
                max_value=stats.max,
                samples=stats.count,
            )
        )

    replacement_summary = next(
        (summary for summary in candidate_count_summaries if summary.candidate_count == 0),
        None,
    )
    replacement_mean = replacement_summary.mean if replacement_summary else None
    replacement_var_mean = _variance_of_mean(replacement_summary) if replacement_summary else None

    value_over_replacement: dict[str, float | None] = {}
    value_over_replacement_std: dict[str, float | None] = {}
    graph_points: list[GraphPoint] = []
    for summary in candidate_count_summaries:
        candidate_count = summary.candidate_count
        candidate_mean = summary.mean
        candidate_var_mean = _variance_of_mean(summary)

        if candidate_count == 0:
            vor_mean = 0.0
            vor_std = 0.0
        else:
            if replacement_mean is None or candidate_mean is None:
                vor_mean = None
            else:
                vor_mean = candidate_mean - replacement_mean

            if (
                candidate_var_mean is None
                or replacement_var_mean is None
                or summary.samples <= 0
                or (replacement_summary and replacement_summary.samples <= 0)
            ):
                vor_std = None
            else:
                vor_std = math.sqrt(candidate_var_mean + replacement_var_mean)

            key = str(candidate_count)
            value_over_replacement[key] = vor_mean
            value_over_replacement_std[key] = vor_std

        if vor_mean is None or vor_std is None:
            graph_points.append(
                GraphPoint(
                    candidate_count=candidate_count,
                    mean=vor_mean,
                    lower=None,
                    upper=None,
                    std_dev=vor_std,
                )
            )
        else:
            graph_points.append(
                GraphPoint(
                    candidate_count=candidate_count,
                    mean=vor_mean,
                    lower=vor_mean - vor_std,
                    upper=vor_mean + vor_std,
                    std_dev=vor_std,
                )
            )

    return ValueOverReplacementSummary(
        policy_version_id=policy_version_id,
        scenario_summaries=scenario_summaries,
        candidate_counts=candidate_count_summaries,
        replacement_summary=replacement_summary,
        value_over_replacement=value_over_replacement,
        value_over_replacement_std=value_over_replacement_std,
        graph_points=graph_points,
    )


# ./tools/run.py recipes.experiment.v0_leaderboard.evaluate policy_version_id=f32ca3a3-b6f0-479f-8105-27ce02b873cb
def evaluate(
    policy_version_id: str,
    result_file_path: str | None = None,
    stats_server_uri: str | None = None,
    seed: int = 50,
    value_over_replacement_path: str | None = None,
) -> MultiPolicyVersionEvalTool:
    """
    Run the V0 Leaderboard Evaluation across all relevant candidate/baseline mixes.

    The run logs every combination of candidate, Thinky, and Ladybug agents, producing
    a value-over-replacement summary that compares the candidate policy against stored
    baseline outcomes.
    """
    if (api_url := stats_server_uri or auto_stats_server_uri()) is None:
        raise ValueError("stats_server_uri is required")

    policy_version_ids = [policy_version_id, THINKY_UUID, LADYBUG_UUID]
    vor_path = value_over_replacement_path or f"value_over_replacement_{policy_version_id}.json"

    tool = LeaderboardEvalTool(
        result_file_path=result_file_path or f"leaderboard_eval_{policy_version_id}.json",
        simulations=simulations(map_seed=seed),
        policy_version_ids=policy_version_ids,
        primary_policy_version_id=policy_version_id,
        verbose=True,
        stats_server_uri=api_url,
        value_over_replacement_path=vor_path,
    )

    tool.system.seed = seed
    return tool


# ./tools/run.py recipes.experiment.v0_leaderboard.play policy_version_id=f32ca3a3-b6f0-479f-8105-27ce02b873cb
def play(policy_version_id: str | None = None, s3_path: str | None = None) -> PlayTool:
    env_config = _make_env(num_cogs=NUM_COGS, map_seed=None)
    return PlayTool(
        sim=SimulationConfig(env=env_config, suite="v0_leaderboard", name="play"),
        policy_version_id=policy_version_id,
        s3_path=s3_path,
    )
