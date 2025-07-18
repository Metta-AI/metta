from omegaconf import DictConfig, OmegaConf

from metta.common.util.config import config_from_path
from metta.mettagrid.curriculum.task_tree import TaskTree
from metta.mettagrid.curriculum.task_tree_config import TaskTreeConfig


def task_tree_from_config_path(config_path: str, env_overrides: DictConfig) -> TaskTree:
    """Load a curriculum (TaskTree) from a config path.

    Args:
        config_path: Path to the curriculum config file
        env_overrides: Environment configuration overrides to apply

    Returns:
        A TaskTree instance representing the curriculum
    """
    cfg = config_from_path(config_path, None)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Apply env_overrides to the config
    if env_overrides:
        overrides_dict = OmegaConf.to_container(env_overrides, resolve=True)
        if "env_overrides" in config_dict:
            config_dict["env_overrides"].update(overrides_dict)
        else:
            config_dict["env_overrides"] = overrides_dict

    # Create TaskTreeConfig and then TaskTree
    task_tree_config = TaskTreeConfig.model_validate(config_dict)
    return task_tree_config.create()
