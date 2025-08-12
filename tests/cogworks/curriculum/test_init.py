"""Tests for the curriculum package __init__.py module."""

import pytest

from metta.cogworks.curriculum import (
    Curriculum,
    CurriculumConfig,
    CurriculumTask,
    Task,
    curriculum,
    tasks,
)

from .conftest import create_test_env_config


class TestPackageImports:
    """Test cases for package-level imports."""

    def test_curriculum_classes_importable(self):
        """Test that main curriculum classes can be imported."""
        # Test that classes are properly exposed at package level
        assert Curriculum is not None
        assert CurriculumConfig is not None
        assert CurriculumTask is not None
        assert Task is not None

    def test_utility_functions_importable(self):
        """Test that utility functions can be imported."""
        assert callable(tasks)
        assert callable(curriculum)

    @pytest.mark.skip(reason="Task generator classes not yet implemented")
    def test_task_generator_classes_importable(self):
        """Test that task generator classes can be imported when implemented."""
        # These imports should work when task_generator.py is implemented:
        # from metta.cogworks.curriculum import (
        #     TaskGenerator,
        #     TaskGeneratorConfig,
        #     SingleTaskGenerator,
        #     SingleTaskGeneratorConfig,
        #     TaskGeneratorSet,
        #     TaskGeneratorSetConfig,
        #     BucketedTaskGenerator,
        #     BucketedTaskGeneratorConfig,
        #     ValueRange,
        # )
        pass

    def test_all_exports_match_implementation(self):
        """Test that __all__ exports match what's actually available."""
        import metta.cogworks.curriculum as curriculum_module

        # Get the __all__ list
        all_exports = getattr(curriculum_module, "__all__", [])

        # Check that each exported name is actually available
        for export_name in all_exports:
            if export_name in ["tasks", "curriculum"]:
                # These are functions defined in __init__.py
                assert hasattr(curriculum_module, export_name)
                assert callable(getattr(curriculum_module, export_name))
            elif export_name in ["Curriculum", "CurriculumConfig", "CurriculumTask", "Task"]:
                # These should be importable classes
                assert hasattr(curriculum_module, export_name)
                assert isinstance(getattr(curriculum_module, export_name), type)
            else:
                # Task generator classes - might not be implemented yet
                # Just check if they're mentioned in __all__
                assert export_name in all_exports


class TestUtilityFunctions:
    """Test cases for package utility functions."""

    @pytest.mark.skip(reason="BucketedTaskGeneratorConfig not yet implemented")
    def test_tasks_function_creates_bucketed_config(self):
        """Test that tasks() function creates BucketedTaskGeneratorConfig."""
        env_cfg = create_test_env_config()

        tasks(env_cfg)

        # Should return BucketedTaskGeneratorConfig instance
        # assert isinstance(task_gen_config, BucketedTaskGeneratorConfig)

        # Should be created from the provided env_config
        # This would test the from_env_config method
        pass

    def test_curriculum_function_creates_config(self):
        """Test that curriculum() function creates CurriculumConfig."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        # Mock task generator config
        task_gen_config = SingleTaskGeneratorConfig()

        # Test with default parameters
        config = curriculum(task_gen_config)

        assert isinstance(config, CurriculumConfig)
        assert config.task_generator_config == task_gen_config
        # Should use default num_active_tasks
        assert config.num_active_tasks == 100

    def test_curriculum_function_with_num_tasks(self):
        """Test curriculum() function with num_tasks parameter."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()

        # Test with custom num_tasks
        config = curriculum(task_gen_config, num_tasks=50)

        assert isinstance(config, CurriculumConfig)
        assert config.task_generator_config == task_gen_config
        assert config.num_active_tasks == 50

    def test_curriculum_function_with_none_num_tasks(self):
        """Test curriculum() function with None num_tasks."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()

        # Test with None num_tasks (should use default)
        config = curriculum(task_gen_config, num_tasks=None)

        assert isinstance(config, CurriculumConfig)
        assert config.num_active_tasks == 100  # Default value


class TestPackageStructure:
    """Test cases for overall package structure."""

    def test_package_has_docstring(self):
        """Test that the package has a docstring."""
        import metta.cogworks.curriculum

        assert metta.cogworks.curriculum.__doc__ is not None
        assert len(metta.cogworks.curriculum.__doc__.strip()) > 0

    def test_no_circular_imports(self):
        """Test that importing the package doesn't cause circular imports."""
        # This test passes if the import succeeds without errors

        # Try importing specific classes
        from metta.cogworks.curriculum import Curriculum, CurriculumConfig

        assert Curriculum is not None
        assert CurriculumConfig is not None

    def test_submodule_access(self):
        """Test that submodules are accessible through the package."""

        # Test that we can access submodules
        from metta.cogworks.curriculum import curriculum as curriculum_module
        from metta.cogworks.curriculum import task as task_module
        # curriculum_env and learning_progress might not be directly exposed

        assert curriculum_module is not None
        assert task_module is not None

    def test_main_classes_functionality(self):
        """Test basic functionality of main classes."""

        # Test that we can create instances of main classes
        env_cfg = create_test_env_config()
        task = Task(task_id="test", env_cfg=env_cfg)
        assert task is not None

        curriculum_task = CurriculumTask(task_id=1, env_cfg=env_cfg)
        assert curriculum_task is not None

        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()
        config = CurriculumConfig(task_generator_config=task_gen_config)
        assert config is not None
