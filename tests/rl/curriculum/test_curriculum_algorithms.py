#!/usr/bin/env python3
"""Tests for specific curriculum algorithm implementations.

This file contains basic tests for each curriculum algorithm type.
Scenario-based tests are in separate files for better organization.
"""


import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.bucketed import BucketedCurriculum, _expand_buckets
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.curriculum.multi_task import MultiTaskCurriculum
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedCurriculum
from metta.mettagrid.curriculum.progressive import ProgressiveCurriculum
from metta.mettagrid.curriculum.random import RandomCurriculum
from metta.mettagrid.curriculum.sampling import SampledTaskCurriculum, SamplingCurriculum

from .conftest import fake_curriculum_from_config_path


class TestCurriculumAlgorithms:
    """Test basic functionality of curriculum algorithms."""

    def test_single_task_curriculum(self, env_cfg):
        """Test SingleTaskCurriculum basic functionality."""
        curr = SingleTaskCurriculum("task", env_cfg)
        task = curr.get_task()
        assert task.id() == "task"
        assert task.env_cfg() == env_cfg
        assert not task.is_complete()
        task.complete(0.5)
        assert task.is_complete()
        with pytest.raises(AssertionError):
            task.complete(0.1)

    def test_random_curriculum(self, monkeypatch, env_cfg):
        """Test RandomCurriculum task selection."""
        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path",
            fake_curriculum_from_config_path
        )

        curr = RandomCurriculum({"a": 1.0, "b": 1.0}, OmegaConf.create({}))
        
        # Test that both tasks can be selected
        task_ids = set()
        for _ in range(20):
            task = curr.get_task()
            task_ids.add(task.id())
        
        # With equal weights, both tasks should be selected at least once in 20 tries
        assert len(task_ids) == 2, f"Should sample both tasks with equal weights, got {task_ids}"

    def test_prioritize_regressed_curriculum_basic(self, monkeypatch, env_cfg):
        """Test PrioritizeRegressedCurriculum basic weight updates."""
        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path",
            fake_curriculum_from_config_path
        )
        curr = PrioritizeRegressedCurriculum(
            {"a": 1.0, "b": 1.0},
            OmegaConf.create({})
        )

        # Complete task "a" with low reward 0.1
        curr.complete_task("a", 0.1)
        weight_after_a = curr._task_weights["a"]
        # Task "a" has max/avg = 0.1/0.1 = 1.0
        # Task "b" has max/avg = 0/0 (undefined, uses epsilon)
        # So task "a" should have higher weight
        assert weight_after_a > curr._task_weights["b"], (
            "Task with actual performance should have higher weight than untried task"
        )

        # Complete task "b" with high reward 1.0
        prev_b = curr._task_weights["b"]
        curr.complete_task("b", 1.0)
        # Task "b" now has max/avg = 1.0/1.0 = 1.0, similar to task "a"
        # But weight should have increased from epsilon
        assert curr._task_weights["b"] > prev_b, (
            "Weight should increase when task gets its first score"
        )

    def test_sampling_curriculum(self, monkeypatch, env_cfg):
        """Test SamplingCurriculum task generation."""
        monkeypatch.setattr(
            "metta.mettagrid.curriculum.sampling.config_from_path",
            lambda path, env_overrides=None: env_cfg
        )

        curr = SamplingCurriculum("dummy")
        t1 = curr.get_task()
        t2 = curr.get_task()

        assert t1.id() == "sample"
        assert t1.env_cfg().game.map.width == 10
        assert t1.id() == t2.id()
        assert t1 is not t2

    def test_progressive_curriculum_basic(self, monkeypatch, env_cfg):
        """Test ProgressiveCurriculum basic progression."""
        monkeypatch.setattr(
            "metta.mettagrid.curriculum.sampling.config_from_path",
            lambda path, env_overrides=None: env_cfg
        )

        curr = ProgressiveCurriculum("dummy")
        t1 = curr.get_task()
        assert t1.env_cfg().game.map.width == 10

        # Complete with high score to trigger progression
        curr.complete_task(t1.id(), 0.6)
        t2 = curr.get_task()
        assert t2.env_cfg().game.map.width == 20

    def test_bucketed_curriculum(self, monkeypatch, env_cfg):
        """Test BucketedCurriculum task generation."""
        monkeypatch.setattr(
            "metta.mettagrid.curriculum.bucketed.config_from_path",
            lambda path, env_overrides=None: env_cfg
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

    def test_expand_buckets_helper(self):
        """Test _expand_buckets helper function."""
        buckets = {
            "param1": {"values": [1, 2, 3]},
            "param2": {"range": (0, 10), "bins": 2},
        }
        expanded = _expand_buckets(buckets)
        # param1 should be a direct list
        assert expanded["param1"] == [1, 2, 3]
        # param2 should be a list of 2 bins
        assert len(expanded["param2"]) == 2
        assert expanded["param2"][0]["range"] == (0, 5)
        assert expanded["param2"][1]["range"] == (5, 10)
        assert all(isinstance(b, dict) and "range" in b for b in expanded["param2"])

    def test_sampled_task_curriculum(self):
        """Test SampledTaskCurriculum parameter sampling."""
        task_id = "test_task"
        task_cfg_template = OmegaConf.create({
            "param1": None,
            "param2": None,
            "param3": None
        })
        sampling_parameters = {
            "param1": 42,
            "param2": {"range": (0, 10), "want_int": True},
            "param3": {"range": (0.0, 1.0)},
        }
        curr = SampledTaskCurriculum(task_id, task_cfg_template, sampling_parameters)
        task = curr.get_task()
        assert task.id() == task_id
        cfg = task.env_cfg()
        assert set(cfg.keys()) == {"param1", "param2", "param3"}
        assert cfg["param1"] == 42
        assert 0 <= cfg["param2"] < 10 and isinstance(cfg["param2"], int)
        assert 0.0 <= cfg["param3"] < 1.0 and isinstance(cfg["param3"], float)

    def test_multi_task_curriculum_completion_rates(self, env_cfg):
        """Test MultiTaskCurriculum completion rate tracking."""
        # Dummy curriculum that returns a task with env_cfg
        class DummyCurriculum:
            def get_task(self):
                class DummyTask:
                    def env_cfg(self):
                        return env_cfg
                return DummyTask()

            def complete_task(self, id, score):
                pass

        curricula = {
            "a": DummyCurriculum(),
            "b": DummyCurriculum(),
            "c": DummyCurriculum()
        }
        curr = MultiTaskCurriculum(curricula)
        
        # Simulate completions: a, a, b
        curr.complete_task("a", 1.0)
        curr.complete_task("a", 1.0)
        curr.complete_task("b", 1.0)
        
        stats = curr.stats()
        # There are 3 completions, so a:2/3, b:1/3, c:0/3
        assert abs(stats["task_completions/a"] - 2/3) < 1e-6
        assert abs(stats["task_completions/b"] - 1/3) < 1e-6
        assert abs(stats["task_completions/c"] - 0.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])