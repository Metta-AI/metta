"""Test that all LPParams can be overridden via command-line style configuration."""

import pytest

from experiments.recipes.in_context_learning.assemblers.assembly_lines import (
    curriculum_args,
    make_task_generator_cfg,
)
from metta.cogworks.curriculum.curriculum import CurriculumConfig


class TestLPConfigOverrides:
    """Test that all learning progress parameters can be overridden."""

    @pytest.fixture
    def task_generator_cfg(self):
        """Create a task generator config for testing."""
        return make_task_generator_cfg(**curriculum_args["train"], map_dir=None)

    @pytest.mark.parametrize(
        "param_name,override_value,expected_value",
        [
            # Core bidirectional LP parameters
            ("use_bidirectional", False, False),
            ("ema_timescale", 0.5, 0.5),
            ("slow_timescale_factor", 0.5, 0.5),
            ("exploration_bonus", 0.3, 0.3),
            ("progress_smoothing", 0.05, 0.05),
            ("performance_bonus_weight", 2.0, 2.0),
            # Task management
            ("num_active_tasks", 500, 500),
            ("rand_task_rate", 0.05, 0.05),
            ("sample_threshold", 20, 20),
            ("memory", 50, 50),
            ("eviction_threshold_percentile", 0.6, 0.6),
            # Basic EMA mode
            ("basic_ema_initial_alpha", 0.5, 0.5),
            ("basic_ema_alpha_decay", 0.4, 0.4),
            ("exploration_blend_factor", 0.7, 0.7),
            # Task tracker EMA
            ("task_tracker_ema_alpha", 0.05, 0.05),
            # Memory and logging
            ("max_slice_axes", 5, 5),
            ("enable_detailed_slice_logging", True, True),
            ("use_shared_memory", False, False),
            ("session_id", "test_session_123", "test_session_123"),
        ],
    )
    def test_parameter_override_via_config(self, task_generator_cfg, param_name, override_value, expected_value):
        """Test that each parameter can be overridden via Pydantic config."""
        # Simulate command-line override by creating config with override
        curriculum_dict = {
            "task_generator": task_generator_cfg,
            "algorithm_config": {
                "type": "learning_progress",
                param_name: override_value,
            },
        }
        curriculum_config = CurriculumConfig(**curriculum_dict)

        # Verify the value is set in algorithm_config
        actual_value = getattr(curriculum_config.algorithm_config, param_name)
        assert actual_value == expected_value, (
            f"Parameter {param_name} was not correctly overridden in algorithm_config. "
            f"Expected {expected_value}, got {actual_value}"
        )

    def test_num_active_tasks_syncs_to_curriculum_config(self, task_generator_cfg):
        """Test that num_active_tasks syncs from algorithm_config to CurriculumConfig."""
        override_value = 750
        curriculum_dict = {
            "task_generator": task_generator_cfg,
            "algorithm_config": {
                "type": "learning_progress",
                "num_active_tasks": override_value,
            },
        }
        curriculum_config = CurriculumConfig(**curriculum_dict)

        # Verify both configs have the same value
        assert curriculum_config.algorithm_config.num_active_tasks == override_value
        assert curriculum_config.num_active_tasks == override_value, (
            "num_active_tasks should sync from algorithm_config to CurriculumConfig"
        )

    def test_max_memory_tasks_automatically_set_to_num_active_tasks(self, task_generator_cfg):
        """Test that max_memory_tasks is automatically set equal to num_active_tasks."""
        num_active_tasks_value = 2500
        curriculum_dict = {
            "task_generator": task_generator_cfg,
            "algorithm_config": {
                "type": "learning_progress",
                "num_active_tasks": num_active_tasks_value,
            },
        }
        curriculum_config = CurriculumConfig(**curriculum_dict)

        # Create the algorithm to verify max_memory_tasks equals num_active_tasks
        algorithm = curriculum_config.algorithm_config.create(curriculum_config.num_active_tasks)

        # Verify that max_memory_tasks was automatically set to num_active_tasks
        assert algorithm.task_tracker.max_memory_tasks == num_active_tasks_value
        assert algorithm.task_tracker._backend.max_tasks == num_active_tasks_value

    def test_task_tracker_ema_alpha_reaches_task_tracker(self, task_generator_cfg):
        """Test that task_tracker_ema_alpha reaches the TaskTracker."""
        override_value = 0.15
        curriculum_dict = {
            "task_generator": task_generator_cfg,
            "algorithm_config": {
                "type": "learning_progress",
                "task_tracker_ema_alpha": override_value,
                "num_active_tasks": 1000,
            },
        }
        curriculum_config = CurriculumConfig(**curriculum_dict)

        # Create the algorithm to verify it reaches TaskTracker
        algorithm = curriculum_config.algorithm_config.create(curriculum_config.num_active_tasks)

        assert algorithm.task_tracker.ema_alpha == override_value

    def test_use_bidirectional_reaches_scorer(self, task_generator_cfg):
        """Test that use_bidirectional determines the scorer type."""
        # Test bidirectional mode
        curriculum_dict = {
            "task_generator": task_generator_cfg,
            "algorithm_config": {
                "type": "learning_progress",
                "use_bidirectional": True,
                "num_active_tasks": 1000,
            },
        }
        curriculum_config = CurriculumConfig(**curriculum_dict)
        algorithm = curriculum_config.algorithm_config.create(curriculum_config.num_active_tasks)

        assert algorithm.hypers.use_bidirectional is True
        from metta.cogworks.curriculum.lp_scorers import BidirectionalLPScorer

        assert isinstance(algorithm.scorer, BidirectionalLPScorer)

        # Test basic mode
        curriculum_dict["algorithm_config"]["use_bidirectional"] = False
        curriculum_config = CurriculumConfig(**curriculum_dict)
        algorithm = curriculum_config.algorithm_config.create(curriculum_config.num_active_tasks)

        assert algorithm.hypers.use_bidirectional is False
        from metta.cogworks.curriculum.lp_scorers import BasicLPScorer

        assert isinstance(algorithm.scorer, BasicLPScorer)

    def test_multiple_parameters_override_simultaneously(self, task_generator_cfg):
        """Test that multiple parameters can be overridden at once."""
        overrides = {
            "type": "learning_progress",
            "use_bidirectional": False,
            "ema_timescale": 0.25,
            "exploration_bonus": 0.25,
            "num_active_tasks": 1500,
            "task_tracker_ema_alpha": 0.04,
            "enable_detailed_slice_logging": True,
        }

        curriculum_dict = {
            "task_generator": task_generator_cfg,
            "algorithm_config": overrides,
        }
        curriculum_config = CurriculumConfig(**curriculum_dict)

        # Verify all overrides are applied
        algo_config = curriculum_config.algorithm_config
        assert algo_config.use_bidirectional is False
        assert algo_config.ema_timescale == 0.25
        assert algo_config.exploration_bonus == 0.25
        assert algo_config.num_active_tasks == 1500
        assert algo_config.task_tracker_ema_alpha == 0.04
        assert algo_config.enable_detailed_slice_logging is True

        # Verify num_active_tasks syncs
        assert curriculum_config.num_active_tasks == 1500

        # Create algorithm and verify deep parameters
        algorithm = algo_config.create(curriculum_config.num_active_tasks)
        assert algorithm.task_tracker.max_memory_tasks == 1500
        assert algorithm.task_tracker.ema_alpha == 0.04

    def test_defaults_remain_when_not_overridden(self, task_generator_cfg):
        """Test that default values are used when parameters are not overridden."""
        curriculum_dict = {
            "task_generator": task_generator_cfg,
            "algorithm_config": {
                "type": "learning_progress",
                # Override only one parameter
                "ema_timescale": 0.3,
            },
        }
        curriculum_config = CurriculumConfig(**curriculum_dict)

        algo_config = curriculum_config.algorithm_config

        # Check the overridden parameter
        assert algo_config.ema_timescale == 0.3

        # Check that defaults are preserved for other parameters
        assert algo_config.use_bidirectional is True  # Default
        assert algo_config.exploration_bonus == 0.1  # Default
        assert algo_config.num_active_tasks == 1000  # Default (updated from 10000 in refactor)
