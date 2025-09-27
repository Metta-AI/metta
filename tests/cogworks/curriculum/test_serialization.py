#!/usr/bin/env python3
"""Test for curriculum configuration serialization and deserialization.

This test ensures that configs can be serialized to JSON and deserialized
back to identical configs.
"""

import unittest

import softmax.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from softmax.cogworks.curriculum import BucketedTaskGenerator, CurriculumConfig, SingleTaskGenerator, TaskGeneratorSet
from softmax.cogworks.curriculum.task_generator import Span


class TestCurriculumConfigSerialization(unittest.TestCase):
    """Test curriculum configuration serialization/deserialization round-trip."""

    def test_single_task_generator(self):
        """Test `SingleTaskGenerator.Config` round-trip."""
        arena = eb.make_arena(num_agents=2)
        single_config = SingleTaskGenerator.Config(env=arena)
        original = CurriculumConfig(task_generator=single_config, num_active_tasks=10)

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        self.assertEqual(original.model_dump_json(), restored.model_dump_json())

    def test_bucketed_task_generator(self):
        """Test `BucketedTaskGenerator.Config` round-trip."""
        arena = eb.make_arena(num_agents=4)
        arena_tasks = cc.bucketed(arena)

        # Add various bucket types
        arena_tasks.add_bucket("game.level_map.width", [10, 20, 30])
        arena_tasks.add_bucket("game.level_map.height", [10, 20, 30])
        arena_tasks.add_bucket("game.agent.rewards.inventory.ore_red", [0, Span(0, 1.0)])

        original = CurriculumConfig(task_generator=arena_tasks)
        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        self.assertEqual(original.model_dump_json(), restored.model_dump_json())

    def test_task_generator_set(self):
        """Test `TaskGeneratorSet.Config` round-trip."""
        arena1 = eb.make_arena(num_agents=2)
        arena2 = eb.make_arena(num_agents=4)

        single1 = SingleTaskGenerator.Config(env=arena1)
        single2 = SingleTaskGenerator.Config(env=arena2)

        set_config = TaskGeneratorSet.Config(task_generators=[single1, single2], weights=[0.5, 0.5])
        original = CurriculumConfig(task_generator=set_config, num_active_tasks=20)

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        self.assertEqual(original.model_dump_json(), restored.model_dump_json())

    def test_deeply_nested_bucketed(self):
        """Test nested `BucketedTaskGenerator.Config` round-trip."""
        arena = eb.make_arena(num_agents=2)

        # Create inner bucketed config
        inner_tasks = cc.bucketed(arena)
        inner_tasks.add_bucket("game.level_map.width", [5, 10])

        # Create outer bucketed config with inner as child
        outer_config = BucketedTaskGenerator.Config(
            child_generator_config=inner_tasks, buckets={"game.level_map.height": [15, 20]}
        )

        original = CurriculumConfig(task_generator=outer_config)

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        self.assertEqual(original.model_dump_json(), restored.model_dump_json())

    def test_value_ranges(self):
        """Test that ValueRange objects survive round-trip."""
        arena = eb.make_arena(num_agents=1)
        arena_tasks = cc.bucketed(arena)

        # Add bucket with ValueRange
        arena_tasks.add_bucket("test.param", [0, Span(0.5, 1.5), 2])

        original = CurriculumConfig(task_generator=arena_tasks)

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        self.assertEqual(original.model_dump_json(), restored.model_dump_json())


if __name__ == "__main__":
    unittest.main()
