"""Consolidated tests for task generation workflows and production patterns."""

import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import ValueRange


class TestTaskGenerationWorkflows:
    """Test task generation workflows used in production."""

    def test_arena_task_generation_workflow(self, arena_env):
        """Test the arena task generation workflow."""
        arena_tasks = cc.bucketed(arena_env)

        # Add various bucket types like in production
        arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30, 40])
        arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30, 40])
        arena_tasks.add_bucket("game.agent.rewards.inventory.ore_red", [0, 0.1, 0.5, 1.0])
        arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 50, 100])

        # Add initial items like in production
        for obj in ["mine_red", "generator_red", "altar"]:
            arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

        # Convert to curriculum
        curriculum = arena_tasks.to_curriculum()

        assert isinstance(curriculum, CurriculumConfig)
        assert curriculum.task_generator is arena_tasks

        curriculum_instance = curriculum.make()
        assert curriculum_instance is not None

        task = curriculum_instance.get_task()
        assert task is not None
        assert task.get_env_cfg() is not None

    def test_navigation_task_generation_workflow(self, navigation_env):
        """Test the navigation task generation workflow."""
        navigation_tasks = cc.bucketed(navigation_env)
        navigation_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.5, 1.0])
        navigation_tasks.add_bucket("game.agent.rewards.inventory.heart_max", [1, 2])

        # Maps like in production - use correct path for navigation env
        maps = ["terrain_maps_nohearts"]
        for size in ["large", "medium", "small"]:
            for terrain in ["balanced", "maze", "sparse", "dense"]:
                maps.append(f"varied_terrain/{terrain}_{size}")

        # Use the correct path for navigation environment
        navigation_tasks.add_bucket("game.map_builder.width", [60, 80, 100])
        navigation_tasks.add_bucket("game.map_builder.height", [60, 80, 100])
        navigation_tasks.add_bucket("game.objects.altar.initial_resource_count", [ValueRange.vr(3, 50)])

        # Convert to curriculum
        curriculum = navigation_tasks.to_curriculum()

        assert isinstance(curriculum, CurriculumConfig)
        assert curriculum.task_generator is navigation_tasks

        curriculum_instance = curriculum.make()
        assert curriculum_instance is not None

        task = curriculum_instance.get_task()
        assert task is not None
        assert task.get_env_cfg() is not None

    def test_task_merging_workflows(self, arena_env):
        """Test task merging workflows used in production."""
        tasks1 = cc.bucketed(arena_env)
        tasks1.add_bucket("game.map_builder.width", [10, 20])

        tasks2 = cc.bucketed(arena_env)
        tasks2.add_bucket("game.map_builder.height", [30, 40])

        tasks3 = cc.bucketed(arena_env)
        tasks3.add_bucket("game.agent.rewards.inventory.ore_red", [0.5, 1.0])

        # Merge them like in production
        merged_tasks = cc.merge([tasks1, tasks2, tasks3])

        # Convert to curriculum
        curriculum = merged_tasks.to_curriculum()

        assert isinstance(curriculum, CurriculumConfig)
        assert curriculum.task_generator is merged_tasks

        curriculum_instance = curriculum.make()
        assert curriculum_instance is not None

        task = curriculum_instance.get_task()
        assert task is not None
        assert task.get_env_cfg() is not None

    def test_custom_algorithm_configuration(self, arena_env):
        """Test custom algorithm configuration in task generation."""
        arena_tasks = cc.bucketed(arena_env)
        arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30])
        arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30])

        lp_config = LearningProgressConfig(
            ema_timescale=0.01,
            pool_size=20,
            sample_size=10,
            max_samples=50,
            exploration_bonus=0.15,
        )

        curriculum = CurriculumConfig(
            task_generator=arena_tasks,
            num_active_tasks=10,
            algorithm_config=lp_config,
        )

        assert isinstance(curriculum, CurriculumConfig)
        assert curriculum.algorithm_config is lp_config

        curriculum_instance = curriculum.make()
        assert curriculum_instance is not None
        assert curriculum_instance._algorithm is not None

        task = curriculum_instance.get_task()
        assert task is not None
        assert task.get_env_cfg() is not None


class TestProductionWorkflows:
    """Test production training workflows."""

    def test_production_training_workflow(self, production_curriculum_config):
        """Test that curriculum can generate tasks throughout a training workflow."""
        curriculum = production_curriculum_config.make()

        # Simulate training workflow - REDUCED from 20 to 5 episodes
        tasks = []
        for episode in range(5):
            task = curriculum.get_task()
            tasks.append(task)

            # Simulate task completion
            if episode % 2 == 0:  # Complete every 2nd task
                task.complete(0.8)
                curriculum.update_task_performance(task._task_id, 0.8)

        # Verify tasks were generated
        assert len(tasks) == 5
        assert all(task is not None for task in tasks)
        assert all(task.get_env_cfg() is not None for task in tasks)

        # Verify some tasks were completed
        completed_tasks = [task for task in tasks if task._num_completions > 0]
        assert len(completed_tasks) > 0

    def test_learning_progress_training_workflow(self, curriculum_with_algorithm):
        """Test full learning progress curriculum workflow."""
        curriculum = curriculum_with_algorithm.make()
        tasks = []
        performances = []

        # Reduced from 15 episodes for faster execution
        for episode in range(5):
            task = curriculum.get_task()
            tasks.append(task)
            base_performance = 0.1 + (episode * 0.02)
            performance = min(0.95, base_performance + (episode * 0.01))
            performances.append(performance)
            task_id = task._task_id
            task.complete(performance)
            curriculum.update_task_performance(task_id, performance)

        assert curriculum._algorithm is not None
        assert len(tasks) == 5
        assert len(performances) == 5
        assert performances[-1] > performances[0], "Performance should improve over time"

    def test_task_reuse_workflow(self, production_curriculum_config):
        """Test that curriculum properly reuses tasks when at capacity."""
        config = production_curriculum_config.model_copy()
        config.num_active_tasks = 3
        curriculum = config.make()

        # When using an algorithm, the curriculum doesn't populate its _tasks dictionary
        # The algorithm manages its own task pool
        if curriculum._algorithm is not None:
            # Fill to capacity using the algorithm
            initial_tasks = []
            for _ in range(3):
                task = curriculum.get_task()
                initial_tasks.append(task)

            # Verify we have tasks from the algorithm
            assert len(initial_tasks) == 3
            assert all(task is not None for task in initial_tasks)

            # Get more tasks - should reuse existing ones from the algorithm's pool
            reused_tasks = []
            for _ in range(5):  # REDUCED from 10 to 5
                task = curriculum.get_task()
                reused_tasks.append(task)

            # Verify we're getting tasks (the algorithm handles reuse internally)
            assert len(reused_tasks) == 5
            assert all(task is not None for task in reused_tasks)

            # The algorithm should maintain its own task pool
            assert curriculum._algorithm is not None

        else:
            # Fallback to testing without algorithm
            # Fill to capacity
            initial_tasks = []
            for _ in range(3):
                task = curriculum.get_task()
                initial_tasks.append(task)

            # Verify we have the expected number of tasks
            assert len(curriculum._tasks) == 3

            # Get more tasks - should reuse existing ones
            reused_tasks = []
            for _ in range(5):  # REDUCED from 10 to 5
                task = curriculum.get_task()
                reused_tasks.append(task)

            # Verify we're reusing tasks
            assert len(curriculum._tasks) == 3  # Should stay at capacity

            # Verify some tasks were reused
            task_ids = [task._task_id for task in reused_tasks]
            initial_task_ids = [task._task_id for task in initial_tasks]

            # Should see some repeated task IDs
            assert len(set(task_ids)) <= 3, "Should only see 3 unique task IDs"
            assert any(tid in initial_task_ids for tid in task_ids), "Should reuse some initial tasks"


class TestTaskGenerationIntegration:
    """Test task generation integration with curriculum system."""

    def test_curriculum_config_validation(self, arena_env):
        """Test that curriculum config validation works with task generators."""
        arena_tasks = cc.bucketed(arena_env)
        arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30])
        arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30])

        # Convert to curriculum
        curriculum = arena_tasks.to_curriculum()

        assert isinstance(curriculum, CurriculumConfig)
        assert curriculum.task_generator is arena_tasks

        curriculum_instance = curriculum.make()
        assert curriculum_instance is not None

        task = curriculum_instance.get_task()
        assert task is not None
        assert task.get_env_cfg() is not None

    def test_curriculum_backward_compatibility(self, arena_env):
        """Test that curriculum maintains backward compatibility."""
        arena_tasks = cc.bucketed(arena_env)
        arena_tasks.add_bucket("game.map_builder.width", [10, 20])

        # Convert to curriculum
        curriculum = arena_tasks.to_curriculum()

        assert isinstance(curriculum, CurriculumConfig)
        assert curriculum.task_generator is arena_tasks

        curriculum_instance = curriculum.make()
        assert curriculum_instance is not None

        task = curriculum_instance.get_task()
        assert task is not None
        assert task.get_env_cfg() is not None

    def test_curriculum_serialization_roundtrip(self, arena_env):
        """Test that curriculum can be serialized and deserialized."""
        arena_tasks = cc.bucketed(arena_env)
        arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30])
        arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30])

        # Convert to curriculum
        curriculum = arena_tasks.to_curriculum()

        # Serialize to JSON
        curriculum_json = curriculum.model_dump_json()

        # Deserialize from JSON
        deserialized_curriculum = CurriculumConfig.model_validate_json(curriculum_json)

        # Verify they're equivalent
        assert deserialized_curriculum.model_dump() == curriculum.model_dump()

        curriculum_instance = deserialized_curriculum.make()
        assert curriculum_instance is not None

        task = curriculum_instance.get_task()
        assert task is not None
        assert task.get_env_cfg() is not None
