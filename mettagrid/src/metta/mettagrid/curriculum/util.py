import hydra
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, SingleTaskCurriculum
from metta.mettagrid.curriculum.sampling import SamplingCurriculum
from metta.mettagrid.util.hydra import config_from_path


def curriculum_from_config_path(config_path: str, env_overrides: DictConfig) -> "Curriculum":
    if "_target_" in config_from_path(config_path, None):
        return hydra.utils.instantiate(
            config_from_path(config_path, OmegaConf.create({"env_overrides": env_overrides}))
        )
    else:
        # If this is an environment rather than a curriculum, we need to wrap it in a curriculum
        # but we have to sample it first.
        task = SamplingCurriculum(config_path, env_overrides).get_task()
        return SingleTaskCurriculum(task.id(), task.env_cfg())
