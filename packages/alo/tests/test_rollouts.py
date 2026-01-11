from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from alo import rollouts
from mettagrid import MettaGridConfig
from mettagrid.policy.policy import PolicySpec
from mettagrid.simulator.multi_episode.rollout import EpisodeRolloutResult


@dataclass
class DummyResult:
    rewards: list[float]
    action_timeouts: list[int]
    stats: dict[str, object]
    steps: int


def test_run_single_episode_rollout_writes_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_run(job: object, device: str | None = None) -> tuple[DummyResult, object]:
        calls["job"] = job
        calls["device"] = device
        return DummyResult(
            rewards=[1.0, 2.0],
            action_timeouts=[0, 1],
            stats={"game": {}, "agent": []},
            steps=7,
        ), object()

    def fake_write(replay: object, path: str) -> None:
        calls["replay_path"] = path

    monkeypatch.setattr(rollouts, "run_pure_single_episode_from_specs", fake_run)
    monkeypatch.setattr(rollouts, "write_replay", fake_write)

    env_cfg = MettaGridConfig.EmptyRoom(num_agents=2, width=2, height=2)
    assignments = np.array([0, 1], dtype=int)
    result = rollouts.run_single_episode_rollout(
        policy_specs=[
            PolicySpec(class_path="mettagrid.policy.random_agent.RandomMultiAgentPolicy"),
            PolicySpec(class_path="mettagrid.policy.random_agent.RandomMultiAgentPolicy"),
        ],
        assignments=assignments,
        env_cfg=env_cfg,
        seed=123,
        max_action_time_ms=100,
        replay_path="replay.json.z",
        device="cpu",
    )

    assert calls["replay_path"] == "replay.json.z"
    assert result.replay_path == "replay.json.z"
    assert result.rewards.tolist() == [1.0, 2.0]
    assert result.steps == 7


def test_run_multi_episode_rollout_collects_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_single_episode_rollout(
        *,
        policy_specs: object,
        assignments: np.ndarray,
        env_cfg: MettaGridConfig,
        seed: int,
        max_action_time_ms: int,
        replay_path: str | None = None,
        device: str | None = None,
    ) -> EpisodeRolloutResult:
        return EpisodeRolloutResult(
            assignments=assignments.copy(),
            rewards=np.array([1.0], dtype=float),
            action_timeouts=np.array([0.0], dtype=float),
            stats={"game": {}, "agent": []},
            replay_path=replay_path,
            steps=1,
            max_steps=env_cfg.game.max_steps,
        )

    monkeypatch.setattr(rollouts, "run_single_episode_rollout", fake_single_episode_rollout)

    env_cfg = MettaGridConfig.EmptyRoom(num_agents=1, width=2, height=2)
    assignments = np.array([0], dtype=int)
    rollout, replay_paths = rollouts.run_multi_episode_rollout(
        policy_specs=[PolicySpec(class_path="mettagrid.policy.random_agent.RandomMultiAgentPolicy")],
        assignments=assignments,
        env_cfg=env_cfg,
        episodes=2,
        seed=7,
        max_action_time_ms=100,
        replay_dir="replays",
        device="cpu",
    )

    assert len(rollout.episodes) == 2
    assert len(replay_paths) == 2
    assert all(path.startswith("replays/") for path in replay_paths)
