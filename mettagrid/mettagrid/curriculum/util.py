from typing import Optional

import hydra
from omegaconf import DictConfig

from mettagrid.curriculum.curriculum import Curriculum, SingleTaskCurriculum
from mettagrid.curriculum.sampling import SamplingCurriculum
from mettagrid.util.hydra import config_from_path


def curriculum_from_config_path(config_path: str, env_overrides: Optional[DictConfig] = None) -> "Curriculum":
    if "_target_" in config_from_path(config_path, {}):
        return hydra.utils.instantiate(config_from_path(config_path, {"env_overrides": env_overrides}))
    else:
        # If this is an environment rather than a curriculum, we need to wrap it in a curriculum
        # but we have to sample it first.
        task = SamplingCurriculum(config_path, env_overrides).get_task()
        return SingleTaskCurriculum(task.id(), task.env_cfg())
