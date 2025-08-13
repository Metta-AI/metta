#!/usr/bin/env python3
"""Test for curriculum configuration serialization and deserialization.

This test ensures that polymorphic TaskGeneratorConfig types are properly
serialized and deserialized with all their specific fields intact.
"""

import json
import unittest

from pydantic import ValidationError

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.builder as eb
from metta.cogworks.curriculum import (
    BucketedTaskGeneratorConfig,
    CurriculumConfig,
    SingleTaskGeneratorConfig,
    TaskGeneratorSetConfig,
)
from metta.cogworks.curriculum.task_generator import ValueRange as vr


class TestCurriculumConfigSerialization(unittest.TestCase):
    """Test curriculum configuration serialization with polymorphic task generators."""

    def test_single_task_generator_serialization(self):
        """Test that SingleTaskGeneratorConfig serializes properly."""
        arena = eb.arena(num_agents=2)
        single_config = SingleTaskGeneratorConfig(env_config=arena)
        curriculum_cfg = CurriculumConfig(task_generator_config=single_config, num_active_tasks=10)

        # Serialize
        json_str = curriculum_cfg.model_dump_json(indent=2)
        data = json.loads(json_str)

        # Check all fields are present
        self.assertIn("task_generator_config", data)
        self.assertIn("type", data["task_generator_config"])
        self.assertEqual(data["task_generator_config"]["type"], "single")
        self.assertIn("env_config", data["task_generator_config"])
        self.assertEqual(data["num_active_tasks"], 10)

        # Test deserialization
        deserialized = CurriculumConfig.model_validate_json(json_str)
        self.assertEqual(deserialized.task_generator_config.type, "single")
        self.assertIsNotNone(deserialized.task_generator_config.env_config)
        self.assertEqual(deserialized.num_active_tasks, 10)

    def test_bucketed_task_generator_serialization(self):
        """Test that BucketedTaskGeneratorConfig with nested polymorphism serializes properly."""
        arena = eb.arena(num_agents=4)
        arena_tasks = cc.tasks(arena)

        # Add various bucket types
        arena_tasks.add_bucket("game.level_map.width", [10, 20, 30])
        arena_tasks.add_bucket("game.level_map.height", [10, 20, 30])
        arena_tasks.add_bucket("game.agent.rewards.inventory.ore_red", [0, vr.vr(0, 1.0)])

        curriculum_cfg = cc.curriculum(arena_tasks, num_tasks=5)

        # Serialize
        json_str = curriculum_cfg.model_dump_json(indent=2)
        data = json.loads(json_str)

        # Check structure
        self.assertIn("task_generator_config", data)
        task_gen = data["task_generator_config"]

        # Check BucketedTaskGeneratorConfig fields
        self.assertEqual(task_gen["type"], "bucketed")
        self.assertIn("buckets", task_gen)
        self.assertIn("child_generator_config", task_gen)

        # Check nested child_generator_config
        child_config = task_gen["child_generator_config"]
        self.assertEqual(child_config["type"], "single")
        self.assertIn("env_config", child_config)

        # Check buckets content
        buckets = task_gen["buckets"]
        self.assertIn("game.level_map.width", buckets)
        self.assertEqual(buckets["game.level_map.width"], [10, 20, 30])
        self.assertIn("game.agent.rewards.inventory.ore_red", buckets)

        # Test deserialization
        deserialized = CurriculumConfig.model_validate_json(json_str)
        self.assertEqual(deserialized.task_generator_config.type, "bucketed")
        self.assertEqual(len(deserialized.task_generator_config.buckets), 3)
        self.assertEqual(deserialized.num_active_tasks, 5)

    def test_task_generator_set_serialization(self):
        """Test that TaskGeneratorSetConfig with list of polymorphic configs serializes properly."""
        arena1 = eb.arena(num_agents=2)
        arena2 = eb.arena(num_agents=4)

        single1 = SingleTaskGeneratorConfig(env_config=arena1)
        single2 = SingleTaskGeneratorConfig(env_config=arena2)

        set_config = TaskGeneratorSetConfig(task_generator_configs=[single1, single2], weights=[0.5, 0.5])

        curriculum_cfg = CurriculumConfig(task_generator_config=set_config, num_active_tasks=20)

        # Serialize
        json_str = curriculum_cfg.model_dump_json(indent=2)
        data = json.loads(json_str)

        # Check structure
        task_gen = data["task_generator_config"]
        self.assertEqual(task_gen["type"], "set")
        self.assertIn("task_generator_configs", task_gen)
        self.assertIn("weights", task_gen)
        self.assertEqual(len(task_gen["task_generator_configs"]), 2)
        self.assertEqual(task_gen["weights"], [0.5, 0.5])

        # Check each config in the set
        for cfg in task_gen["task_generator_configs"]:
            self.assertEqual(cfg["type"], "single")
            self.assertIn("env_config", cfg)

        # Test deserialization
        deserialized = CurriculumConfig.model_validate_json(json_str)
        self.assertEqual(deserialized.task_generator_config.type, "set")
        self.assertEqual(len(deserialized.task_generator_config.task_generator_configs), 2)
        self.assertEqual(deserialized.task_generator_config.weights, [0.5, 0.5])

    def test_deeply_nested_bucketed_serialization(self):
        """Test serialization of BucketedTaskGeneratorConfig with a bucketed child."""
        arena = eb.arena(num_agents=2)

        # Create inner bucketed config
        inner_tasks = cc.tasks(arena)
        inner_tasks.add_bucket("game.level_map.width", [5, 10])

        # Create outer bucketed config with inner as child
        outer_config = BucketedTaskGeneratorConfig(
            child_generator_config=inner_tasks, buckets={"game.level_map.height": [15, 20]}
        )

        curriculum_cfg = CurriculumConfig(task_generator_config=outer_config)

        # Serialize
        json_str = curriculum_cfg.model_dump_json(indent=2)
        data = json.loads(json_str)

        # Check nested structure
        outer = data["task_generator_config"]
        self.assertEqual(outer["type"], "bucketed")
        self.assertIn("game.level_map.height", outer["buckets"])

        inner = outer["child_generator_config"]
        self.assertEqual(inner["type"], "bucketed")
        self.assertIn("game.level_map.width", inner["buckets"])

        innermost = inner["child_generator_config"]
        self.assertEqual(innermost["type"], "single")
        self.assertIn("env_config", innermost)

        # Test deserialization
        deserialized = CurriculumConfig.model_validate_json(json_str)
        self.assertEqual(deserialized.task_generator_config.type, "bucketed")
        self.assertEqual(deserialized.task_generator_config.child_generator_config.type, "bucketed")

    def test_serialization_preserves_value_ranges(self):
        """Test that ValueRange objects serialize and deserialize correctly."""
        arena = eb.arena(num_agents=1)
        arena_tasks = cc.tasks(arena)

        # Add bucket with ValueRange
        arena_tasks.add_bucket("test.param", [0, vr.vr(0.5, 1.5), 2])

        curriculum_cfg = cc.curriculum(arena_tasks)

        # Serialize
        json_str = curriculum_cfg.model_dump_json(indent=2)
        data = json.loads(json_str)

        # Check ValueRange serialization
        bucket_values = data["task_generator_config"]["buckets"]["test.param"]
        self.assertEqual(len(bucket_values), 3)
        self.assertEqual(bucket_values[0], 0)
        self.assertIsInstance(bucket_values[1], dict)
        self.assertEqual(bucket_values[1]["range_min"], 0.5)
        self.assertEqual(bucket_values[1]["range_max"], 1.5)
        self.assertEqual(bucket_values[2], 2)

        # Test deserialization
        deserialized = CurriculumConfig.model_validate_json(json_str)
        bucket = deserialized.task_generator_config.buckets["test.param"]
        self.assertEqual(len(bucket), 3)
        self.assertIsInstance(bucket[1], vr)
        self.assertEqual(bucket[1].range_min, 0.5)
        self.assertEqual(bucket[1].range_max, 1.5)

    def test_backwards_compatibility(self):
        """Test that old serialized configs without type field fail gracefully."""
        # Create a config dict without type discriminators
        old_config = {
            "task_generator_config": {
                "overrides": {},
                # Missing "type" field - should cause validation error
                "env_config": {},
            },
            "max_task_id": 1000000,
            "num_active_tasks": 100,
            "new_task_rate": 0.01,
        }

        # This should raise a validation error due to missing discriminator
        with self.assertRaises(ValidationError):
            CurriculumConfig.model_validate(old_config)


if __name__ == "__main__":
    unittest.main()
