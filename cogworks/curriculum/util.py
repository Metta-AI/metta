"""Utility functions for curriculum management."""

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.common.util.config import config_from_path
from .core import Curriculum, SingleTaskCurriculum


def curriculum_from_config_path(config_path: str, env_overrides: DictConfig) -> Curriculum:
    """Load curriculum from configuration file path.
    
    Args:
        config_path: Path to curriculum configuration
        env_overrides: Environment configuration overrides
        
    Returns:
        Curriculum instance loaded from config
    """
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