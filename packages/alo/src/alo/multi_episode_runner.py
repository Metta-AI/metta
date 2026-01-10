from __future__ import annotations

import math
import uuid
from typing import Callable, Optional, Sequence

import numpy as np

from alo.pure_single_episode_runner import PureSingleEpisodeSpecJob, run_pure_single_episode_from_specs
from mettagrid import MettaGridConfig
from mettagrid.policy.policy import PolicySpec
from mettagrid.renderer.renderer import RenderMode
from mettagrid.simulator.multi_episode.rollout import EpisodeRolloutResult, MultiEpisodeRolloutResult

ProgressCallback = Callable[[int], None]


def _compute_policy_agent_counts(num_agents: int, proportions: list[float]) -> list[int]:
    total = sum(proportions)
    if total <= 0:
        raise ValueError("Total policy proportion must be positive.")
    fractions = [proportion / total for proportion in proportions]

    ideals = [num_agents * f for f in fractions]
    counts = [math.floor(x) for x in ideals]
    remaining = num_agents - sum(counts)

    remainders = [(i, ideals[i] - counts[i]) for i in range(len(fractions))]
    remainders.sort(key=lambda x: x[1], reverse=True)
    for i in range(remaining):
        counts[remainders[i][0]] += 1
    return counts


def multi_episode_rollout(
    env_cfg: MettaGridConfig,
    policy_specs: list[PolicySpec],
    episodes: int,
    seed: int = 0,
    proportions: Optional[Sequence[float]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    save_replay: Optional[str] = None,
    max_action_time_ms: int | None = None,
    render_mode: Optional[RenderMode] = None,
    device_override: str | None = None,
) -> MultiEpisodeRolloutResult:
    if proportions is not None and len(proportions) != len(policy_specs):
        raise ValueError("Number of proportions must match number of policies.")

    policy_counts = _compute_policy_agent_counts(
        env_cfg.game.num_agents, list(proportions) if proportions is not None else [1.0] * len(policy_specs)
    )
    assignments = np.repeat(np.arange(len(policy_specs)), policy_counts)

    episode_results: list[EpisodeRolloutResult] = []
    rng = np.random.default_rng(seed)
    device = device_override or "cpu"
    max_action_time = max_action_time_ms if max_action_time_ms is not None else 10000

    for episode_idx in range(episodes):
        rng.shuffle(assignments)
        replay_path = None
        if save_replay is not None:
            replay_path = f"{save_replay}/{uuid.uuid4()}.json.z"

        job = PureSingleEpisodeSpecJob(
            policy_specs=policy_specs,
            assignments=assignments.tolist(),
            env=env_cfg,
            replay_uri=replay_path,
            seed=seed + episode_idx,
            max_action_time_ms=max_action_time,
        )

        results, replay = run_pure_single_episode_from_specs(job, device)

        if replay_path is not None:
            if replay is None:
                raise ValueError("No replay was generated")
            if replay_path.endswith(".z"):
                replay.set_compression("zlib")
            elif replay_path.endswith(".gz"):
                replay.set_compression("gzip")
            replay.write_replay(replay_path)

        if render_mode is not None and render_mode != "none":
            raise ValueError("Rendering is not supported in alo multi-episode runner")

        result = EpisodeRolloutResult(
            assignments=assignments.copy(),
            rewards=np.array(results.rewards, dtype=float),
            action_timeouts=np.array(results.action_timeouts, dtype=float),
            stats=results.stats,
            replay_path=replay_path,
            steps=results.steps,
            max_steps=env_cfg.game.max_steps,
        )

        episode_results.append(result)

        if progress_callback is not None:
            progress_callback(episode_idx)

    return MultiEpisodeRolloutResult(episodes=episode_results)
