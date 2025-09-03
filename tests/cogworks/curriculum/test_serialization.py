"""Test for curriculum configuration serialization and deserialization.

This test ensures that configs can be serialized to JSON and deserialized
back to identical configs.
"""

import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum import (
    BucketedTaskGeneratorConfig,
    CurriculumConfig,
    SingleTaskGeneratorConfig,
)
from metta.cogworks.curriculum.task_generator import ValueRange


class TestCurriculumConfigSerialization:
    """Test curriculum configuration serialization/deserialization round-trip."""

    def test_single_task_generator(self, arena_env):
        """Test SingleTaskGeneratorConfig round-trip."""
        single_config = SingleTaskGeneratorConfig(env=arena_env)
        original = CurriculumConfig(task_generator=single_config, num_active_tasks=10)

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        assert original.model_dump_json() == restored.model_dump_json()

    def test_bucketed_task_generator(self, bucketed_task_generator_config):
        """Test BucketedTaskGeneratorConfig round-trip."""
        original = CurriculumConfig(task_generator=bucketed_task_generator_config)

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        assert original.model_dump_json() == restored.model_dump_json()

    def test_task_generator_set(self, task_generator_set_config):
        """Test TaskGeneratorSetConfig round-trip."""
        original = CurriculumConfig(task_generator=task_generator_set_config, num_active_tasks=20)

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        assert original.model_dump_json() == restored.model_dump_json()

    def test_deeply_nested_bucketed(self, arena_env):
        """Test nested BucketedTaskGeneratorConfig round-trip."""
        # Create inner bucketed config
        inner_tasks = cc.bucketed(arena_env)
        inner_tasks.add_bucket("game.level_map.width", [5, 10])

        # Create outer bucketed config with inner as child
        outer_config = BucketedTaskGeneratorConfig(
            child_generator_config=inner_tasks, buckets={"game.level_map.height": [15, 20]}
        )

        original = CurriculumConfig(task_generator=outer_config)

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        assert original.model_dump_json() == restored.model_dump_json()

    def test_value_ranges(self, arena_env):
        """Test that ValueRange objects survive round-trip."""
        arena_tasks = cc.bucketed(arena_env)

        # Add bucket with ValueRange
        arena_tasks.add_bucket("test.param", [0, ValueRange.vr(0.5, 1.5), 2])

        original = CurriculumConfig(task_generator=arena_tasks)

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        assert original.model_dump_json() == restored.model_dump_json()

    def test_complex_production_curriculum(self, production_curriculum_config):
        """Test that production-like curriculum configurations can be serialized."""
        original = production_curriculum_config

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        assert original.model_dump_json() == restored.model_dump_json()

    def test_navigation_curriculum(self, production_navigation_curriculum):
        """Test that navigation curriculum configurations can be serialized."""
        original = production_navigation_curriculum

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        assert original.model_dump_json() == restored.model_dump_json()

    def test_curriculum_with_algorithm(self, curriculum_with_algorithm):
        """Test that curriculum with algorithm can be serialized."""
        original = curriculum_with_algorithm

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CurriculumConfig.model_validate_json(json_str)

        # Check they serialize to the same JSON
        assert original.model_dump_json() == restored.model_dump_json()
