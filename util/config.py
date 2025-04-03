import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError
import logging
logger = logging.getLogger(__name__)

def config_from_path(config_path: str, overrides: DictConfig = None) -> DictConfig:
    env_cfg = hydra.compose(config_name=config_path)
    if config_path.startswith("/"):
        config_path = config_path[1:]
    path = config_path.split("/")
    for p in path[:-1]:
        env_cfg = env_cfg[p]
    if overrides is not None:
        try:
            env_cfg = OmegaConf.merge(env_cfg, overrides)
        except ConfigKeyError:
            logger.warning(f"Cannot parse overrides when using multienv_mettagrid")
    return env_cfg
