import random

import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.bucketed import BucketedCurriculum, _expand_buckets
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.curriculum.low_reward import LowRewardCurriculum
from metta.mettagrid.curriculum.progressive import ProgressiveCurriculum
from metta.mettagrid.curriculum.random import RandomCurriculum
from metta.mettagrid.curriculum.sampling import SampledTaskCurriculum, SamplingCurriculum


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
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.random.curriculum_from_config_path", fake_curriculum_from_config_path
    )

    curr = RandomCurriculum({"a": 1.0, "b": 1.0}, OmegaConf.create({}))
    task = curr.get_task()
    assert task.id() == "b"
    assert task.name() == "b:b"


def test_low_reward_curriculum_updates(monkeypatch, env_cfg):
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.random.curriculum_from_config_path", fake_curriculum_from_config_path
    )
    curr = LowRewardCurriculum({"a": 1.0, "b": 1.0}, OmegaConf.create({}))

    curr.complete_task("a", 0.1)
    weight_after_a = curr._task_weights["a"]
    assert weight_after_a > curr._task_weights["b"]

    prev_b = curr._task_weights["b"]
    curr.complete_task("b", 1.0)
    assert curr._task_weights["b"] > prev_b


def test_sampling_curriculum(monkeypatch, env_cfg):
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.sampling.config_from_path", lambda path, env_overrides=None: env_cfg
    )

    curr = SamplingCurriculum("dummy")
    t1 = curr.get_task()
    t2 = curr.get_task()

    assert t1.id() == "sample(0)"
    assert t1.env_cfg().game.map.width == 10
    assert t1.id() == t2.id()
    assert t1 is not t2


def test_progressive_curriculum(monkeypatch, env_cfg):
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.sampling.config_from_path", lambda path, env_overrides=None: env_cfg
    )

    curr = ProgressiveCurriculum("dummy")
    t1 = curr.get_task()
    assert t1.env_cfg().game.map.width == 10

    curr.complete_task(t1.id(), 0.6)
    t2 = curr.get_task()
    assert t2.env_cfg().game.map.width == 20


def test_bucketed_curriculum(monkeypatch, env_cfg):
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.bucketed.config_from_path", lambda path, env_overrides=None: env_cfg
    )
    buckets = {
        "game.map.width": {"values": [5, 10]},
        "game.map.height": {"values": [5, 10]},
    }
    curr = BucketedCurriculum("dummy", buckets=buckets)

    # There should be 4 tasks (2x2 grid)
    assert len(curr._id_to_curriculum) == 4
    # Sample a task
    task = curr.get_task()
    assert hasattr(task, "id")
    assert any(str(w) in task.id() for w in [5, 10])


def test_expand_buckets_values_and_range():
    buckets = {
        "param1": {"values": [1, 2, 3]},
        "param2": {"range": (0, 10), "bins": 2},
    }
    expanded = _expand_buckets(buckets)
    # param1 should be a direct list
    assert expanded["param1"] == [1, 2, 3]
    # param2 should be a list of 2 bins, each a dict with 'range' and 'want_int'
    assert len(expanded["param2"]) == 2
    assert expanded["param2"][0]["range"] == (0, 5)
    assert expanded["param2"][1]["range"] == (5, 10)
    assert all(isinstance(b, dict) and "range" in b for b in expanded["param2"])


def test_sampled_task_curriculum():
    # Setup: one value bucket, one range bucket (int), one range bucket (float)
    task_id = "test_task"
    task_cfg_template = OmegaConf.create({"param1": None, "param2": None, "param3": None})
    bucket_parameters = ["param1", "param2", "param3"]
    bucket_values = [42, {"range": (0, 10), "want_int": True}, {"range": (0.0, 1.0)}]
    curr = SampledTaskCurriculum(task_id, task_cfg_template, bucket_parameters, bucket_values)
    task = curr.get_task()
    assert task.id() == task_id
    cfg = task.env_cfg()
    assert set(cfg.keys()) == {"param1", "param2", "param3"}
    assert cfg["param1"] == 42
    assert 0 <= cfg["param2"] < 10 and isinstance(cfg["param2"], int)
    assert 0.0 <= cfg["param3"] < 1.0 and isinstance(cfg["param3"], float)
