"""Test curriculum task generator invariant validation.

This module tests that TaskGenerators properly enforce invariants across
generated tasks, specifically that resources, actions, and num_agents
remain consistent across all tasks.
"""

import pytest

from metta.cogworks.curriculum.task_generator import BucketedTaskGenerator, SingleTaskGenerator, TaskGeneratorSet
from mettagrid.builder.envs import make_arena


class TestTaskGeneratorInvariants:
    """Test that TaskGenerator validates invariants across generated tasks."""

    def test_single_task_generator_maintains_invariants(self):
        """SingleTaskGenerator should maintain invariants trivially."""
        base_config = make_arena(num_agents=24)
        generator = SingleTaskGenerator.Config(env=base_config).create()

        # Generate multiple tasks - all should be identical
        task1 = generator.get_task(0)
        task2 = generator.get_task(1)
        task3 = generator.get_task(100)

        # All should have same invariants
        assert len(task1.game.resource_names) == len(task2.game.resource_names) == len(task3.game.resource_names)
        assert task1.game.num_agents == task2.game.num_agents == task3.game.num_agents

    def test_bucketed_generator_with_consistent_config(self):
        """BucketedTaskGenerator with consistent configs should pass validation."""
        base_config = make_arena(num_agents=24)
        generator_config = BucketedTaskGenerator.Config.from_mg(base_config)

        # Add buckets that don't change invariants (use correct nested paths)
        generator_config.add_bucket("game.map_builder.width", [25, 32, 40])
        generator_config.add_bucket("game.max_steps", [1000, 2000, 3000])

        generator = generator_config.create()

        # Generate multiple tasks - should all pass validation
        for task_id in range(10):
            task = generator.get_task(task_id)
            assert len(task.game.resource_names) == len(base_config.game.resource_names)
            assert task.game.num_agents == base_config.game.num_agents

    def test_task_generator_set_with_consistent_generators(self):
        """TaskGeneratorSet with consistent generators should pass validation."""
        config1 = make_arena(num_agents=24)
        config2 = make_arena(num_agents=24)  # Same agent count

        generator_config = (
            TaskGeneratorSet.Config()
            .add(SingleTaskGenerator.Config(env=config1), weight=1.0)
            .add(SingleTaskGenerator.Config(env=config2), weight=1.0)
        )

        generator = generator_config.create()

        # Generate multiple tasks - should all pass validation
        for task_id in range(20):
            task = generator.get_task(task_id)
            assert len(task.game.resource_names) == len(config1.game.resource_names)
            assert task.game.num_agents == config1.game.num_agents

    def test_inconsistent_resources_raises_error(self):
        """TaskGenerator with inconsistent resources should raise AssertionError."""
        config1 = make_arena(num_agents=24)
        config2 = make_arena(num_agents=24)

        # Modify config2 to have different resources
        config2.game.resource_names = config2.game.resource_names[:2]  # Remove some resources

        generator_config = (
            TaskGeneratorSet.Config()
            .add(SingleTaskGenerator.Config(env=config1), weight=1.0)
            .add(SingleTaskGenerator.Config(env=config2), weight=1.0)
        )

        generator = generator_config.create()

        # Try generating tasks - should raise AssertionError when hitting inconsistent config
        # Note: TaskGeneratorSet randomly chooses, so first task establishes reference
        with pytest.raises(AssertionError, match="inconsistent resource count"):
            # Try enough tasks to hit the config with different resources
            for task_id in range(0, 50):
                generator.get_task(task_id)

    def test_inconsistent_actions_raises_error(self):
        """TaskGenerator with inconsistent actions should raise AssertionError."""
        config1 = make_arena(num_agents=24)
        config2 = make_arena(num_agents=24)

        # Modify config2 to have different actions by disabling some
        config2.game.actions.move.enabled = False  # Disable an action
        config2.game.actions.put_items.enabled = False  # Disable another action

        generator_config = (
            TaskGeneratorSet.Config()
            .add(SingleTaskGenerator.Config(env=config1), weight=1.0)
            .add(SingleTaskGenerator.Config(env=config2), weight=1.0)
        )

        generator = generator_config.create()

        # First task establishes reference
        generator.get_task(0)

        # Subsequent tasks that violate invariants should raise AssertionError
        with pytest.raises(AssertionError, match="inconsistent action count"):
            # Try enough tasks to hit the config with different actions
            for task_id in range(1, 50):
                generator.get_task(task_id)

    def test_inconsistent_num_agents_raises_error(self):
        """TaskGenerator with inconsistent num_agents should raise AssertionError."""
        config1 = make_arena(num_agents=24)
        config2 = make_arena(num_agents=12)  # Different agent count!

        generator_config = (
            TaskGeneratorSet.Config()
            .add(SingleTaskGenerator.Config(env=config1), weight=1.0)
            .add(SingleTaskGenerator.Config(env=config2), weight=1.0)
        )

        generator = generator_config.create()

        # Try generating tasks - should raise AssertionError when hitting inconsistent config
        # Note: TaskGeneratorSet randomly chooses, so first task establishes reference
        with pytest.raises(AssertionError, match="inconsistent agent count"):
            # Try enough tasks to hit the config with different agents
            for task_id in range(0, 50):
                generator.get_task(task_id)

    def test_error_message_contains_task_id(self):
        """Error messages should include the task_id for debugging."""
        config1 = make_arena(num_agents=24)
        config2 = make_arena(num_agents=12)

        generator_config = (
            TaskGeneratorSet.Config()
            .add(SingleTaskGenerator.Config(env=config1), weight=1.0)
            .add(SingleTaskGenerator.Config(env=config2), weight=1.0)
        )

        generator = generator_config.create()

        # Establish reference with task 0
        generator.get_task(0)

        # Try to trigger error and check message quality
        try:
            for task_id in range(1, 50):
                generator.get_task(task_id)
        except AssertionError as e:
            error_msg = str(e)
            # Should mention what failed
            assert "task" in error_msg.lower()
            # Should mention expected vs actual
            assert "expected" in error_msg.lower()
            assert "got" in error_msg.lower()
            # Should mention which invariant
            assert "resource" in error_msg.lower() or "action" in error_msg.lower() or "agent" in error_msg.lower(), (
                f"Error message should specify which invariant failed: {error_msg}"
            )
        else:
            pytest.fail("Expected AssertionError to be raised")

    def test_overrides_do_not_violate_invariants(self):
        """Overrides should not be able to violate invariants."""
        base_config = make_arena(num_agents=24)

        # Try to override num_agents - should still be validated
        generator_config = SingleTaskGenerator.Config(env=base_config, overrides={"game.num_agents": 12})

        generator = generator_config.create()

        # First task with override
        task1 = generator.get_task(0)
        assert task1.game.num_agents == 12  # Override applied

        # Subsequent tasks should also have the override
        task2 = generator.get_task(1)
        assert task2.game.num_agents == 12  # Consistent with reference

    def test_multiple_inconsistent_invariants(self):
        """When multiple invariants differ, error should mention the first one encountered."""
        config1 = make_arena(num_agents=24)
        config2 = make_arena(num_agents=12)  # Different agents
        config2.game.resource_names = config2.game.resource_names[:2]  # Different resources too

        generator_config = (
            TaskGeneratorSet.Config()
            .add(SingleTaskGenerator.Config(env=config1), weight=1.0)
            .add(SingleTaskGenerator.Config(env=config2), weight=1.0)
        )

        generator = generator_config.create()

        # Establish reference
        generator.get_task(0)

        # Should fail on first check (resources are checked first in the code)
        with pytest.raises(AssertionError, match="resource"):
            for task_id in range(1, 50):
                generator.get_task(task_id)
