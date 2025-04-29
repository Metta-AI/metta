import numpy as np
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from metta.util.config import config_from_path
from mettagrid.mettagrid_env import MettaGridEnv


class MettaGridEnvSet(MettaGridEnv):
    """
    This is a wrapper around MettaGridEnv that allows for multiple environments to be used for training.
    """

    def __init__(
        self,
        env_cfg: DictConfig,
        render_mode: str,
        buf=None,
        **kwargs,
    ):
        self._envs = list(env_cfg.envs.keys())
        self._probabilities = list(env_cfg.envs.values())
        self._num_agents_global = env_cfg.num_agents
        self._env_cfg = self._get_new_env_cfg()
        self.check_action_space()

        super().__init__(env_cfg, render_mode=render_mode, buf=buf, **kwargs)
        self._cfg_template = None  # we don't use this with multiple envs, so we clear it to emphasize that fact

    def check_action_space(self):
        env_cfgs = [config_from_path(env) for env in self._envs]
        action_spaces = [env_cfg.game.actions for env_cfg in env_cfgs]
        if not all(action_space == action_spaces[0] for action_space in action_spaces):
            raise ValueError("All environments must have the same action space.")

    def _get_new_env_cfg(self):
        selected_env = np.random.choice(self._envs, p=self._probabilities)
        env_cfg = config_from_path(selected_env)
        if self._num_agents_global != env_cfg.game.num_agents:
            raise ValueError(
                "For MettaGridEnvSet, the number of agents must be the same for all environments. "
                f"Global: {self._num_agents_global}, Env: {env_cfg.game.num_agents}"
            )
        env_cfg = OmegaConf.create(env_cfg)
        OmegaConf.resolve(env_cfg)
        return env_cfg
