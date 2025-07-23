import hydra
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.curriculum.sampling import SamplingCurriculum
from metta.mettagrid.util.hydra import config_from_path


def curriculum_from_config_path(config_path: str, env_overrides: DictConfig) -> "Curriculum":
    if "_target_" in config_from_path(config_path, None):
        return hydra.utils.instantiate(
            # Don't recurse here. We want one level of instantiation so we get the curriculum object, but we don't
            # want to instantiate sub-curricula or map builders, since we may still want to override their
            # config.
            config_from_path(config_path, OmegaConf.create({"env_overrides": env_overrides})),
            _recursive_=False,
        )
    else:
        return SamplingCurriculum(config_path, env_overrides)
