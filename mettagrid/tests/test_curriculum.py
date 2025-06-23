import random

import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum import LowRewardCurriculum, RandomCurriculum, SingleTaskCurriculum
from metta.mettagrid.curriculum.progressive import ProgressiveCurriculum
from metta.mettagrid.curriculum.sampling import SamplingCurriculum


@pytest.fixture
def env_cfg():
    return OmegaConf.create({"sampling": 0, "game": {"num_agents": 1, "map": {"width": 10, "height": 10}}})


def fake_curriculum_from_config_path(path, env_overrides=None):
    return SingleTaskCurriculum(
        path,
        task_cfg=OmegaConf.merge(
            OmegaConf.create({"game": {"num_agents": 5, "map": {"width": 10, "height": 10}}}), env_overrides
        ),
    )


def test_single_task_curriculum(env_cfg):
    curr = SingleTaskCurriculum("task", env_cfg)
    task = curr.get_task()
    assert task.id() == "task"
    assert task.env_cfg() == env_cfg
    assert not task.is_complete()
    task.complete(0.5)
    assert task.is_complete()
    with pytest.raises(AssertionError):
        task.complete(0.1)


def test_random_curriculum_selects_task(monkeypatch, env_cfg):
    monkeypatch.setattr(random, "choices", lambda population, weights: ["b"])
    monkeypatch.setattr("mettagrid.curriculum.multi_task.curriculum_from_config_path", fake_curriculum_from_config_path)

    curr = RandomCurriculum({"a": 1.0, "b": 1.0}, OmegaConf.create({}))
    task = curr.get_task()
    assert task.id() == "b"
    assert task.name() == "b:b"


def test_low_reward_curriculum_updates(monkeypatch, env_cfg):
    monkeypatch.setattr("mettagrid.curriculum.multi_task.curriculum_from_config_path", fake_curriculum_from_config_path)
    curr = LowRewardCurriculum({"a": 1.0, "b": 1.0}, OmegaConf.create({}))

    curr.complete_task("a", 0.1)
    weight_after_a = curr._task_weights["a"]
    assert weight_after_a > curr._task_weights["b"]

    prev_b = curr._task_weights["b"]
    curr.complete_task("b", 1.0)
    assert curr._task_weights["b"] > prev_b


def test_sampling_curriculum(monkeypatch, env_cfg):
    monkeypatch.setattr("mettagrid.curriculum.sampling.config_from_path", lambda path, env_overrides=None: env_cfg)

    curr = SamplingCurriculum("dummy")
    t1 = curr.get_task()
    t2 = curr.get_task()

    assert t1.id() == "sample(0)"
    assert t1.env_cfg().game.map.width == 10
    assert t1.id() == t2.id()
    assert t1 is not t2


def test_progressive_curriculum(monkeypatch, env_cfg):
    monkeypatch.setattr("mettagrid.curriculum.sampling.config_from_path", lambda path, env_overrides=None: env_cfg)

    curr = ProgressiveCurriculum("dummy")
    t1 = curr.get_task()
    assert t1.env_cfg().game.map.width == 10

    curr.complete_task(t1.id(), 0.6)
    t2 = curr.get_task()
    assert t2.env_cfg().game.map.width == 20
