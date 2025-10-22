"""Tests for baseline normalization configuration in bidirectional learning progress."""

from metta.cogworks.curriculum.curriculum import Curriculum, CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import SingleTaskGenerator
from mettagrid.builder.envs import MettaGridConfig
from mettagrid.config import GameConfig


class TestBaselineNormalization:
    """Test baseline normalization configuration."""

    def test_default_no_baseline_normalization(self):
        """Test that baseline normalization is disabled by default."""
        config = LearningProgressConfig(num_active_tasks=50, use_bidirectional=True)

        assert config.use_baseline_normalization is False

    def test_baseline_normalization_can_be_enabled(self):
        """Test that baseline normalization can be explicitly enabled."""
        config = LearningProgressConfig(num_active_tasks=50, use_bidirectional=True, use_baseline_normalization=True)

        assert config.use_baseline_normalization is True

    def test_raw_scores_used_by_default(self):
        """Test that raw success rates are used when baseline normalization is disabled."""
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

        curriculum_config = CurriculumConfig(
            task_generator=task_generator_config,
            num_active_tasks=10,
            algorithm_config=LearningProgressConfig(
                num_active_tasks=10, use_bidirectional=True, use_baseline_normalization=False
            ),
        )

        curriculum = Curriculum(curriculum_config)
        algorithm = curriculum._algorithm

        # Generate and score some tasks
        for _ in range(5):
            task = curriculum.get_task()
            for score in [0.3, 0.4, 0.5]:
                task.complete(score)
                curriculum.update_task_performance(task._task_id, score)

        # Verify scorer has the config setting
        assert algorithm.scorer.config.use_baseline_normalization is False

        # Random baseline should not be initialized when disabled
        assert algorithm.scorer._random_baseline is None

    def test_baseline_normalization_when_enabled(self):
        """Test that baseline normalization works when explicitly enabled."""
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

        curriculum_config = CurriculumConfig(
            task_generator=task_generator_config,
            num_active_tasks=10,
            algorithm_config=LearningProgressConfig(
                num_active_tasks=10, use_bidirectional=True, use_baseline_normalization=True
            ),
        )

        curriculum = Curriculum(curriculum_config)
        algorithm = curriculum._algorithm

        # Generate and score some tasks
        for _ in range(5):
            task = curriculum.get_task()
            for score in [0.3, 0.4, 0.5]:
                task.complete(score)
                curriculum.update_task_performance(task._task_id, score)

        # Verify scorer has the config setting
        assert algorithm.scorer.config.use_baseline_normalization is True

        # Random baseline should be initialized when enabled
        # Need to trigger update first
        algorithm.scorer.get_stats()
        if len(algorithm.scorer._outcomes) > 0:
            assert algorithm.scorer._random_baseline is not None

    def test_both_modes_produce_valid_learning_progress(self):
        """Test that both normalization modes produce valid learning progress scores."""
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

        for use_baseline in [False, True]:
            curriculum_config = CurriculumConfig(
                task_generator=task_generator_config,
                num_active_tasks=10,
                algorithm_config=LearningProgressConfig(
                    num_active_tasks=10,
                    use_bidirectional=True,
                    use_baseline_normalization=use_baseline,
                ),
            )

            curriculum = Curriculum(curriculum_config)

            # Generate tasks and collect scores
            for _ in range(10):
                task = curriculum.get_task()
                # Simulate improving performance
                for score in [0.3, 0.5, 0.7, 0.9]:
                    task.complete(score)
                    curriculum.update_task_performance(task._task_id, score)

            # Should be able to get stats without error
            stats = curriculum.stats()
            assert "algorithm/num_tasks" in stats

            # Learning progress should be computed
            if len(curriculum._algorithm.scorer._outcomes) > 0:
                lp_scores = curriculum._algorithm.scorer.get_stats()
                assert "mean_learning_progress" in lp_scores
