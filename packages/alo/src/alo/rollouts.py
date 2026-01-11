from __future__ import annotations

import uuid
from pathlib import Path
from typing import Sequence

import numpy as np

from alo.pure_single_episode_runner import PureSingleEpisodeSpecJob, run_pure_single_episode_from_specs
from alo.replay import write_replay
from mettagrid import MettaGridConfig
from mettagrid.policy.policy import PolicySpec
from mettagrid.simulator.multi_episode.rollout import EpisodeRolloutResult, MultiEpisodeRolloutResult


def run_single_episode_rollout(
    *,
    policy_specs: Sequence[PolicySpec],
    assignments: np.ndarray,
    env_cfg: MettaGridConfig,
    seed: int,
    max_action_time_ms: int,
    replay_path: str | None = None,
    device: str | None = None,
) -> EpisodeRolloutResult:
    job = PureSingleEpisodeSpecJob(
        policy_specs=list(policy_specs),
        assignments=assignments.tolist(),
        env=env_cfg,
        replay_uri=replay_path,
        seed=seed,
        max_action_time_ms=max_action_time_ms,
    )
    results, replay = run_pure_single_episode_from_specs(job, device=device)
    if replay_path is not None and replay is not None:
        write_replay(replay, replay_path)

    return EpisodeRolloutResult(
        assignments=assignments.copy(),
        rewards=np.array(results.rewards, dtype=float),
        action_timeouts=np.array(results.action_timeouts, dtype=float),
        stats=results.stats,
        replay_path=replay_path,
        steps=results.steps,
        max_steps=env_cfg.game.max_steps,
    )


def run_multi_episode_rollout(
    *,
    policy_specs: Sequence[PolicySpec],
    assignments: np.ndarray,
    env_cfg: MettaGridConfig,
    episodes: int,
    seed: int,
    max_action_time_ms: int,
    replay_dir: str | Path | None = None,
    create_replay_dir: bool = False,
    device: str | None = None,
) -> tuple[MultiEpisodeRolloutResult, list[str]]:
    if replay_dir is not None and create_replay_dir:
        Path(replay_dir).mkdir(parents=True, exist_ok=True)

    assignments = np.array(assignments, dtype=int, copy=True)
    rng = np.random.default_rng(seed)
    episode_results: list[EpisodeRolloutResult] = []
    replay_paths: list[str] = []

    for episode_idx in range(episodes):
        rng.shuffle(assignments)
        replay_path = None
        if replay_dir is not None:
            replay_path = str(Path(replay_dir) / f"{uuid.uuid4()}.json.z")

        episode_results.append(
            run_single_episode_rollout(
                policy_specs=policy_specs,
                assignments=assignments,
                env_cfg=env_cfg,
                seed=seed + episode_idx,
                max_action_time_ms=max_action_time_ms,
                replay_path=replay_path,
                device=device,
            )
        )
        if replay_path is not None:
            replay_paths.append(replay_path)

    return MultiEpisodeRolloutResult(episodes=episode_results), replay_paths
