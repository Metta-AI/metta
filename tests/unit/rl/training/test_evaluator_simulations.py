from types import SimpleNamespace
from unittest.mock import Mock

import torch

from metta.rl.training.evaluator import Evaluator, EvaluatorConfig
from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import MettaGridConfig


class _FakeTask:
    def __init__(self, env_cfg: MettaGridConfig):
        self._env_cfg = env_cfg

    def get_env_cfg(self) -> MettaGridConfig:
        return self._env_cfg


def _make_evaluator(cfg: EvaluatorConfig) -> Evaluator:
    system_cfg = SimpleNamespace(vectorization="serial")
    return Evaluator(config=cfg, device=torch.device("cpu"), system_cfg=system_cfg, stats_client=None)


def test_build_simulations_defaults_to_curriculum_tasks() -> None:
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
    curriculum = Mock()
    curriculum.get_task.side_effect = [_FakeTask(env_cfg), _FakeTask(env_cfg)]

    cfg = EvaluatorConfig(
        epoch_interval=1,
        evaluate_local=False,
        evaluate_remote=False,
        replay_dir="replays",
        num_training_tasks=2,
    )

    evaluator = _make_evaluator(cfg)
    sims = evaluator._build_simulations(curriculum)

    assert len(sims) == 2
    assert all(sim.suite == "training" for sim in sims)
    assert [sim.name for sim in sims] == ["train_task_0", "train_task_1"]
    # Ensure env configs are deep copies
    for sim in sims:
        assert sim.env is not env_cfg
        assert sim.env.model_dump() == env_cfg.model_dump()


def test_build_simulations_respects_training_replay_overrides() -> None:
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=2)
    custom_sim = SimulationConfig(suite="custom", name="custom_env", env=env_cfg)
    curriculum = Mock()

    cfg = EvaluatorConfig(
        epoch_interval=1,
        evaluate_local=False,
        evaluate_remote=False,
        replay_dir="replays",
        num_training_tasks=2,
        training_replay_envs=[custom_sim],
    )

    evaluator = _make_evaluator(cfg)
    sims = evaluator._build_simulations(curriculum)

    curriculum.get_task.assert_not_called()
    assert len(sims) == 1
    sim = sims[0]
    assert sim is not custom_sim
    assert sim.model_dump() == custom_sim.model_dump()
