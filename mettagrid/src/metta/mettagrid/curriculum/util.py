import hydra
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, SingleTaskCurriculum
from metta.mettagrid.util.hydra import config_from_path


def curriculum_from_config_path(config_path: str, env_overrides: DictConfig) -> Curriculum:
    if config_path.startswith("/synthetic/"):
        # Load plain YAML under the current config search path (e.g., configs/curriculum_analysis)
        cfg = config_from_path(config_path, env_overrides)
        task_id = config_path.split("/")[-1]
        return SingleTaskCurriculum(task_id, cfg)

    if "_target_" in config_from_path(config_path, None):
        return hydra.utils.instantiate(
            # (a) Don't recurse here. We want one level of instantiation so we get the curriculum object, but
            # we don't want to instantiate sub-curricula or map builders, since we may still want to override their
            # config.
            # (b) Notice that we're wrapping env_overrides in an extra layer. This means we're overriding the
            # overrides, so we carry them forward, rather than trying to apply them now.
            config_from_path(config_path, OmegaConf.create({"env_overrides": env_overrides})),
            _recursive_=False,
        )
    else:
        config = config_from_path(config_path, env_overrides)
        if not isinstance(config, DictConfig):
            raise ValueError(f"Invalid curriculum config at {config_path}: {config}")
        return SingleTaskCurriculum(config_path, config)
