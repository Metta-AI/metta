from typing import Optional

import hydra
from omegaconf import DictConfig, ListConfig

from mettagrid.curriculum.curriculum import Curriculum, SingleTaskCurriculum
from mettagrid.curriculum.sampling import SamplingCurriculum
from mettagrid.util.hydra import config_from_path


def curriculum_from_config(
    config: str | DictConfig | ListConfig, env_overrides: Optional[DictConfig] = None
) -> "Curriculum":
    if isinstance(config, str):
        config = config_from_path(config, {"env_overrides": env_overrides})
    if "_target_" in config:
        return hydra.utils.instantiate(config)
    else:
        # If this is an environment rather than a curriculum, we need to wrap it in a curriculum
        # but we have to sample it first.
        task = SamplingCurriculum(config, env_overrides).get_task()
        return SingleTaskCurriculum(task.id(), task.env_cfg())
