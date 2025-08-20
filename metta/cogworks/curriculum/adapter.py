"""Simple adapter for learning progress curriculum integration."""

from typing import Any, Dict

from learning_progress_minimal import (
    LearningProgressCurriculum,
    LearningProgressCurriculumConfig,
)


def create_learning_progress_curriculum(config_dict: Dict[str, Any]) -> LearningProgressCurriculum:
    """Create learning progress curriculum from dict config.

    Args:
        config_dict: Dictionary containing curriculum configuration

    Returns:
        LearningProgressCurriculum instance

    Example:
        config = {
            "ema_timescale": 0.001,
            "progress_smoothing": 0.05,
            "rand_task_rate": 0.25,
            "memory": 25,
        }
        curriculum = create_learning_progress_curriculum(config)
    """
    config = LearningProgressCurriculumConfig(**config_dict)
    return LearningProgressCurriculum(config)


def load_learning_progress_from_yaml(yaml_path: str) -> LearningProgressCurriculum:
    """Load learning progress curriculum from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        LearningProgressCurriculum instance
    """
    import yaml

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return create_learning_progress_curriculum(config_dict)
