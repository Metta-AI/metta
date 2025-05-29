import random

import pytest
from omegaconf import OmegaConf

from mettagrid import curriculum


@pytest.fixture
def env_cfg():
    return OmegaConf.create({"sampling": 0, "game": {"num_agents": 1, "map": {"width": 10, "height": 10}}})


def test_single_task_curriculum(env_cfg):
    curr = curriculum.SingleTaskCurriculum("task", env_cfg)
    task = curr.get_task()
    assert task.id() == "task"
    assert task.env_cfg() == env_cfg
    assert not task.is_complete()
    task.complete(0.5)
    assert task.is_complete()
    with pytest.raises(AssertionError):
        task.complete(0.1)


def test_random_curriculum_selects_task(monkeypatch, env_cfg):
    def fake_from_config(path, env_overrides=None):
        return curriculum.SingleTaskCurriculum(path, env_cfg)

    monkeypatch.setattr(curriculum.Curriculum, "from_config_path", staticmethod(fake_from_config))
    monkeypatch.setattr(random, "choices", lambda population, weights: ["b"])

    curr = curriculum.RandomCurriculum({"a": 1.0, "b": 1.0}, OmegaConf.create({}))
    task = curr.get_task()
    assert task.id() == "b"
    assert task.name() == "b:b"


def test_low_reward_curriculum_updates(monkeypatch, env_cfg):
    def fake_from_config(path, env_overrides=None):
        return curriculum.SingleTaskCurriculum(path, env_cfg)

    monkeypatch.setattr(curriculum.Curriculum, "from_config_path", staticmethod(fake_from_config))
    curr = curriculum.LowRewardCurriculum({"a": 1.0, "b": 1.0}, OmegaConf.create({}))

    curr.complete_task("a", 0.1)
    weight_after_a = curr._task_weights["a"]
    assert weight_after_a > curr._task_weights["b"]

    prev_b = curr._task_weights["b"]
    curr.complete_task("b", 1.0)
    assert curr._task_weights["b"] > prev_b


def test_sampling_curriculum(monkeypatch, env_cfg):
    monkeypatch.setattr(curriculum, "config_from_path", lambda path, env_overrides=None: env_cfg)
    curr = curriculum.SamplingCurriculum("dummy")
    t1 = curr.get_task()
    t2 = curr.get_task()

    assert t1.id() == "sample(0)"
    assert t1.env_cfg().game.map.width == 10
    assert t1.id() == t2.id()
    assert t1 is not t2


def test_progressive_curriculum(monkeypatch, env_cfg):
    monkeypatch.setattr(curriculum, "config_from_path", lambda path, env_overrides=None: env_cfg)
    curr = curriculum.ProgressiveCurriculum("dummy")
    t1 = curr.get_task()
    assert t1.env_cfg().game.map.width == 10

    curr.complete_task(t1.id(), 0.6)
    t2 = curr.get_task()
    assert t2.env_cfg().game.map.width == 20
