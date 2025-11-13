import logging
from collections import defaultdict
from typing import Sequence

from pydantic import Field, model_validator

from metta.common.tool import Tool
from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.sim.runner import SimulationRunConfig, SimulationRunResult, run_simulations
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri
from mettagrid.policy.policy import PolicySpec
from mettagrid.simulator.multi_episode.summary import (
    MultiEpisodeRolloutSummary,
    build_multi_episode_rollout_summaries,
)

logger = logging.getLogger(__name__)


class LiteEvalTool(Tool):
    policy_specs: Sequence[PolicySpec] = Field(description="Policy specs to evaluate")
    simulations: Sequence[SimulationRunConfig] = Field(description="Simulations to evaluate")
    replay_dir: str = Field(default_factory=auto_replay_dir)
    stats_server_uri: str | None = Field(default_factory=auto_stats_server_uri)

    @model_validator(mode="after")
    def validate(self) -> "LiteEvalTool":
        for simulation in self.simulations:
            if simulation.proportions is not None and len(simulation.proportions) != len(self.policy_specs):
                raise ValueError("Number of proportions must match number of policies.")
        return self

    def invoke(self, args: dict[str, str]) -> int | None:
        simulation_results = run_simulations(
            policy_specs=self.policy_specs,
            simulations=self.simulations,
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            enable_replays=True,
        )
        logger.info(f"Simulation results: {simulation_results}")

        summaries = build_multi_episode_rollout_summaries(
            rollout_results=[result.results for result in simulation_results],
            policy_specs=list(self.policy_specs),
        )
        logger.info(f"Summaries: {summaries}")

        eval_results = self._build_eval_results(simulation_results, summaries)
        if eval_results is not None:
            logger.info("EvalResults schema: %s", eval_results)

        return 0

    def _build_eval_results(
        self,
        simulation_results: Sequence[SimulationRunResult],
        summaries: Sequence[MultiEpisodeRolloutSummary],
    ) -> EvalResults | None:
        if not self.policy_specs:
            logger.info("No policy specs configured; skipping EvalResults conversion.")
            return None

        target_policy_idx = 0
        target_policy_name = self.policy_specs[target_policy_idx].name
        if len(self.policy_specs) > 1:
            logger.info(
                "Multiple policy specs provided (%d); EvalResults conversion will use only the first policy (%s).",
                len(self.policy_specs),
                target_policy_name,
            )

        if len(simulation_results) != len(summaries):
            logger.warning(
                "Simulation results (%d) and summaries (%d) are misaligned; skipping EvalResults conversion.",
                len(simulation_results),
                len(summaries),
            )
            return None

        simulation_scores: dict[tuple[str, str], float] = {}
        category_scores_accum: defaultdict[str, list[float]] = defaultdict(list)
        replay_urls: dict[str, list[str]] = {}

        for i, (result, summary) in enumerate(zip(simulation_results, summaries, strict=True)):
            category = result.run.episode_tags.get("category", "lite_eval_category")
            sim_name = result.run.episode_tags.get("name", f"lite_eval_name_{i}")
            policy_rewards: list[float] = []
            for per_policy_rewards in summary.per_episode_per_policy_avg_rewards.values():
                if not per_policy_rewards or len(per_policy_rewards) <= target_policy_idx:
                    continue
                policy_reward = per_policy_rewards[target_policy_idx]
                if policy_reward is not None:
                    policy_rewards.append(float(policy_reward))

            mission_score = sum(policy_rewards) / len(policy_rewards) if policy_rewards else 0.0
            simulation_scores[(category, sim_name)] = mission_score
            category_scores_accum[category].append(mission_score)

            if result.replay_urls:
                replay_urls[f"{category}.{sim_name}"] = list(result.replay_urls.values())

        if not simulation_scores:
            logger.info("No simulation scores computed; skipping EvalResults conversion.")
            return None

        category_scores = {
            category: sum(values) / len(values) for category, values in category_scores_accum.items() if values
        }

        return EvalResults(
            scores=EvalRewardSummary(
                category_scores=category_scores,
                simulation_scores=simulation_scores,
            ),
            replay_urls=replay_urls,
        )
