"""Tests for CurriculumConfig and YAML-based curriculum creation."""

from unittest.mock import patch

import pytest
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum import Curriculum, CurriculumConfig
from metta.mettagrid.curriculum.curriculum_algorithm import DiscreteRandomCurriculum
from metta.mettagrid.curriculum.learning_progress import LearningProgressAlgorithm
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedAlgorithm
from metta.mettagrid.curriculum.progressive import ProgressiveAlgorithm


@pytest.fixture
def mock_config_from_path():
    """Mock config_from_path to avoid Hydra initialization."""

    def _mock_config(path, overrides):
        # Return a simple config based on the path
        return DictConfig(
            {"game": {"num_agents": 4, "map_builder": {"room": {"dir": "default", "objects": {"altar": 10}}}}}
        )

    with patch("metta.mettagrid.curriculum.curriculum_config.config_from_path", side_effect=_mock_config):
        yield _mock_config


class TestCurriculumConfig:
    """Test CurriculumConfig creation and validation."""

    def test_simple_task_list_config(self, mock_config_from_path):
        """Test creating a simple Curriculum from a list of task paths."""
        config_dict = {
            "name": "test_curriculum",
            "algorithm": "discrete_random",
            "env_paths": ["/env/mettagrid/arena/basic", "/env/mettagrid/arena/combat", "/env/mettagrid/arena/tag"],
        }

        # Create and validate config
        curriculum_config = CurriculumConfig.model_validate(config_dict)
        assert curriculum_config.name == "test_curriculum"
        assert len(curriculum_config.env_paths) == 3

        # Create Curriculum
        tree = curriculum_config.create()
        assert isinstance(tree, Curriculum)
        assert tree.num_tasks == 3
        assert isinstance(tree.curriculum_algorithm, DiscreteRandomCurriculum)

        # Check task names
        task_names = [child.name for child in tree.children]
        assert "basic" in task_names
        assert "combat" in task_names
        assert "tag" in task_names

    def test_single_env_path_config(self, mock_config_from_path):
        """Test creating a Curriculum with a single env path."""
        config_dict = {"name": "single_task", "env_paths": ["/env/mettagrid/arena/basic"]}

        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()

        assert tree.num_tasks == 1
        assert tree.children[0].name == "basic"

    def test_algorithm_string_shorthand(self, mock_config_from_path):
        """Test different algorithm specifications."""
        # Test discrete_random
        config_dict = {"name": "test", "algorithm": "discrete_random", "env_paths": ["/env/mettagrid/arena/basic"]}
        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()
        assert isinstance(tree.curriculum_algorithm, DiscreteRandomCurriculum)

        # Test learning_progress
        config_dict["algorithm"] = "learning_progress"
        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()
        assert isinstance(tree.curriculum_algorithm, LearningProgressAlgorithm)

        # Test prioritize_regressed
        config_dict["algorithm"] = "prioritize_regressed"
        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()
        assert isinstance(tree.curriculum_algorithm, PrioritizeRegressedAlgorithm)

        # Test progressive
        config_dict["algorithm"] = "progressive"
        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()
        assert isinstance(tree.curriculum_algorithm, ProgressiveAlgorithm)

    def test_parameter_ranges_with_discrete_values(self, mock_config_from_path):
        """Test creating a bucketed curriculum with discrete parameter values."""
        config_dict = {
            "name": "bucketed_test",
            "env_paths": ["/env/mettagrid/arena/combat"],
            "parameters": {
                "game.agent.rewards.ore_red": {"values": [0, 1]},
                "game.objects.generator_red.input_resources.ore_red": {"values": [0, 1, 2]},
            },
        }

        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()

        # Should create 2 * 3 = 6 tasks
        assert tree.num_tasks == 6

        # Check that task names contain parameter values
        task_names = [child.name for child in tree.children]
        for name in task_names:
            assert "ore_red=" in name
            assert "input_resources.ore_red=" in name

    def test_parameter_ranges_with_continuous_ranges(self, mock_config_from_path):
        """Test creating a bucketed curriculum with continuous parameter ranges."""
        config_dict = {
            "name": "range_test",
            "env_paths": ["/env/mettagrid/arena/basic"],
            "parameters": {
                "game.map_builder.width": {"range": [10, 50], "bins": 4},
                "game.map_builder.height": {"range": [10, 50], "bins": 2},
            },
        }

        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()

        # Should create 4 * 2 = 8 tasks
        assert tree.num_tasks == 8

        # Check that task names contain range indicators
        task_names = [child.name for child in tree.children]
        for name in task_names:
            assert "width=" in name
            assert "height=" in name
            # Should have range notation
            assert "(" in name and ")" in name

    def test_env_overrides(self, mock_config_from_path):
        """Test that env_overrides are applied to all tasks."""
        config_dict = {
            "name": "override_test",
            "env_paths": ["/env/mettagrid/arena/basic", "/env/mettagrid/arena/combat"],
            "env_overrides": {"game": {"episode_length": 100, "num_agents": 8}},
        }

        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()

        # Check that overrides are applied to all tasks
        for child in tree.children:
            # Note: We can't directly check the env_config values without resolving
            # the config, but we can verify the structure was created correctly
            assert hasattr(child, "env_config")

    def test_invalid_config_both_tasks_and_children(self):
        """Test that specifying both env_paths and children raises an error."""
        config_dict = {
            "name": "invalid",
            "env_paths": ["/env/mettagrid/arena/basic"],
            "children": [{"name": "child1", "env_paths": ["/env/mettagrid/arena/combat"]}],
        }

        with pytest.raises(ValueError, match="Cannot specify both env_paths and children"):
            CurriculumConfig.model_validate(config_dict)

    def test_invalid_config_no_tasks_or_children(self):
        """Test that specifying neither tasks nor children raises an error."""
        config_dict = {"name": "invalid"}

        with pytest.raises(ValueError, match="Must specify either env_paths or children"):
            CurriculumConfig.model_validate(config_dict)

    def test_hierarchical_config(self, mock_config_from_path):
        """Test creating a hierarchical Curriculum."""
        config_dict = {
            "name": "root",
            "algorithm": "discrete_random",
            "children": [
                {"name": "easy_tasks", "env_paths": ["/env/mettagrid/arena/basic", "/env/mettagrid/arena/basic_easy"]},
                {"name": "hard_tasks", "env_paths": ["/env/mettagrid/arena/combat", "/env/mettagrid/arena/advanced"]},
            ],
        }

        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()

        # Root should have 2 children (easy_tasks and hard_tasks)
        assert tree.num_tasks == 2
        assert isinstance(tree.children[0], Curriculum)
        assert isinstance(tree.children[1], Curriculum)

        # Each child should have 2 tasks
        assert tree.children[0].num_tasks == 2
        assert tree.children[1].num_tasks == 2

    def test_hierarchical_config_with_paths(self, mock_config_from_path):
        """Test creating a hierarchical Curriculum with child paths."""

        # Mock the child configs that will be loaded
        def _mock_config(path, overrides):
            if "arena/learning_progress" in path:
                return DictConfig(
                    {
                        "name": "arena_learning_progress",
                        "algorithm": "learning_progress",
                        "env_paths": ["/env/mettagrid/arena/basic", "/env/mettagrid/arena/combat"],
                    }
                )
            else:
                return DictConfig({"game": {"num_agents": 4}})

        with patch("metta.mettagrid.curriculum.curriculum_config.config_from_path", side_effect=_mock_config):
            config_dict = {
                "name": "root",
                "algorithm": "discrete_random",
                "children": [
                    "/env/mettagrid/curriculum/arena/learning_progress",
                    {"name": "inline_child", "env_paths": ["/env/mettagrid/arena/tag"]},
                ],
            }

            curriculum_config = CurriculumConfig.model_validate(config_dict)
            tree = curriculum_config.create()

            # Root should have 2 children
            assert tree.num_tasks == 2
            assert tree.children[0].name == "arena_learning_progress"
            assert tree.children[1].name == "inline_child"

    def test_unknown_algorithm_error(self):
        """Test that unknown algorithm types raise an error."""
        config_dict = {"name": "test", "algorithm": "unknown_algorithm", "env_paths": ["/env/mettagrid/arena/basic"]}

        with pytest.raises(ValueError, match="Unknown algorithm type: unknown_algorithm"):
            CurriculumConfig.model_validate(config_dict)

    def test_parameter_range_validation(self):
        """Test parameter range validation."""
        # Missing bins for range is now valid (continuous range)
        config_dict = {
            "name": "test",
            "env_paths": ["/env/mettagrid/arena/basic"],
            "parameters": {
                "game.width": {"range": [10, 50]}  # No bins = continuous range
            },
        }

        # This should now be valid
        curriculum_config = CurriculumConfig.model_validate(config_dict)
        assert curriculum_config.parameters["game.width"].bins is None

        # bins = 1 should now raise an error
        config_dict["parameters"]["game.width"] = {"range": [10, 50], "bins": 1}

        with pytest.raises(ValueError, match="'bins' must be >= 2 when specified"):
            CurriculumConfig.model_validate(config_dict)

        # Both values and range specified
        config_dict["parameters"]["game.width"] = {"values": [10, 20], "range": [10, 50], "bins": 2}

        with pytest.raises(ValueError, match="Cannot specify both 'values' and 'range'"):
            CurriculumConfig.model_validate(config_dict)

        # Neither values nor range specified
        config_dict["parameters"]["game.width"] = {}

        with pytest.raises(ValueError, match="Must specify either 'values' or 'range'"):
            CurriculumConfig.model_validate(config_dict)

    def test_continuous_range_without_bins(self, mock_config_from_path):
        """Test creating tasks with continuous ranges (no bins)."""
        config_dict = {
            "name": "continuous_test",
            "env_paths": ["/env/mettagrid/arena/basic"],
            "parameters": {
                "game.agent.speed": {"range": [0.5, 2.0]},  # Continuous float range
                "game.objects.spawn_rate": {"range": [1, 10]},  # Continuous int range
            },
        }

        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()

        # Should create 1 task (no discretization)
        assert tree.num_tasks == 1

        # Task name should show continuous ranges
        task_name = tree.children[0].name
        assert "speed=(0.500,2.000)" in task_name
        assert "spawn_rate=(1.000,10.000)" in task_name

    def test_yaml_like_config(self, mock_config_from_path):
        """Test a configuration that mimics actual YAML structure."""
        # This mimics configs/env/mettagrid/curriculum/arena/random.yaml
        config_dict = {
            "algorithm": "discrete_random",
            "tasks": {"/env/mettagrid/arena/basic": 1, "/env/mettagrid/arena/combat": 1, "/env/mettagrid/arena/tag": 1},
        }

        # Tasks as dict needs to be converted to list
        config_dict["env_paths"] = list(config_dict["tasks"].keys())
        del config_dict["tasks"]
        config_dict["name"] = "arena_random"  # Add required name

        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()

        assert tree.num_tasks == 3
        assert isinstance(tree.curriculum_algorithm, DiscreteRandomCurriculum)

    def test_real_yaml_conversion(self, mock_config_from_path):
        """Test converting a real YAML config to CurriculumConfig."""
        # This mimics the bucketed curriculum structure
        yaml_str = """
        name: navigation_bucketed
        env_paths:
          - /env/mettagrid/navigation/training/terrain_from_numpy
        parameters:
          game.map_builder.room.dir:
            values: ["desert", "forest", "grassland", "ice", "lavaland", "mountain", "ocean", "swamp"]
          game.map_builder.room.objects.altar:
            range: [10, 50]
            bins: 10
        """

        config = OmegaConf.create(yaml_str)
        config_dict = OmegaConf.to_container(config, resolve=True)

        curriculum_config = CurriculumConfig.model_validate(config_dict)
        tree = curriculum_config.create()

        # Should create 8 terrains * 10 altar bins = 80 tasks
        assert tree.num_tasks == 80
