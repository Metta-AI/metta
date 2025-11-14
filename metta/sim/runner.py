from collections import defaultdict
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from mettagrid import MettaGridConfig
from mettagrid.policy.policy import MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult, multi_episode_rollout
from mettagrid.simulator.multi_episode.summary import build_multi_episode_rollout_summaries
from mettagrid.simulator.replay_log_writer import ReplayLogWriter


class SimulationRunConfig(BaseModel):
    env: MettaGridConfig  # noqa: F821
    num_episodes: int = Field(default=1, description="Number of episodes to run", ge=1)
    proportions: Sequence[float] | None = None

    max_action_time_ms: int | None = Field(
        default=10000, description="Maximum time (in ms) a policy is given to take an action"
    )
    episode_tags: dict[str, str] = Field(default_factory=dict, description="Tags to add to each episode")


class SimulationRunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run: SimulationRunConfig
    results: MultiEpisodeRolloutResult
    replay_urls: dict[str, str]


MultiAgentPolicyInitializer = Callable[[PolicyEnvInterface], MultiAgentPolicy]


def run_simulations(
    policy_initializers: Sequence[MultiAgentPolicyInitializer],
    simulations: Sequence[SimulationRunConfig],
    replay_dir: str | None,
    seed: int,
    enable_replays: bool = True,
) -> list[SimulationRunResult]:
    simulation_rollouts: list[SimulationRunResult] = []

    for simulation in simulations:
        proportions = simulation.proportions
        replay_writer: ReplayLogWriter | None = None
        if enable_replays and replay_dir:
            replay_writer = ReplayLogWriter(str(replay_dir))

        env_interface = PolicyEnvInterface.from_mg_cfg(simulation.env)
        multi_agent_policies: list[MultiAgentPolicy] = [pi(env_interface) for pi in policy_initializers]

        rollout_result = multi_episode_rollout(
            env_cfg=simulation.env,
            policies=multi_agent_policies,
            episodes=simulation.num_episodes,
            seed=seed,
            proportions=proportions,
            # TODO: support this if and only if we also reflect that it happened in results
            # max_time_s=simulation.max_time_s,
            max_action_time_ms=simulation.max_action_time_ms,
            event_handlers=[replay_writer] if replay_writer else None,
        )

        replay_urls = replay_writer.get_written_replay_urls() if replay_writer else {}

        simulation_rollouts.append(
            SimulationRunResult(
                run=simulation,
                results=rollout_result,
                replay_urls=replay_urls,
            )
        )

    return simulation_rollouts


# This gets the sim results into a format we know how to submit to wandb
# We should move away from this towards something with a schema that doesn't give e.g. `category` and `sim_name` meaning
def build_eval_results(
    rollout_results: list[SimulationRunResult], target_policy_idx: int, num_policies: int
) -> EvalResults:
    summaries = build_multi_episode_rollout_summaries(
        rollout_results=[result.results for result in rollout_results], num_policies=num_policies
    )
    simulation_scores: dict[tuple[str, str], float] = {}
    category_scores_accum: defaultdict[str, list[float]] = defaultdict(list)
    replay_urls: dict[str, list[str]] = {}

    for i, (result, summary) in enumerate(zip(rollout_results, summaries, strict=True)):
        category = result.run.episode_tags.get("category", "unknown")
        sim_name = result.run.episode_tags.get("name", f"unknown_{i}")
        policy_rewards: list[float] = []
        for per_policy_rewards in summary.per_episode_per_policy_avg_rewards.values():
            if not per_policy_rewards or len(per_policy_rewards) <= target_policy_idx:
                continue
            policy_reward = per_policy_rewards[target_policy_idx]
            if policy_reward is not None:
                policy_rewards.append(float(policy_reward))

        avg_reward = sum(policy_rewards) / len(policy_rewards) if policy_rewards else 0.0
        simulation_scores[(category, sim_name)] = avg_reward
        category_scores_accum[category].append(avg_reward)

        if result.replay_urls:
            replay_urls[f"{category}.{sim_name}"] = list(result.replay_urls.values())

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
