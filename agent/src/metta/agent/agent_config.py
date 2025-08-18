from typing import ClassVar, Optional, Union

import gymnasium as gym
import torch.nn as nn
from omegaconf import DictConfig
from pydantic import ConfigDict

from metta.common.util.config import Config
from metta.mettagrid import MettaGridEnv
from metta.rl.system_config import SystemConfig


class AgentConfig(Config):
    env: MettaGridEnv
    system_cfg: SystemConfig
    agent_cfg: Union[str, DictConfig]
    policy: Optional[nn.Module] = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
    )


class ComponentPolicyConfig(Config):
    obs_space: Optional[Union[gym.spaces.Space, gym.spaces.Dict]] = None
    obs_width: Optional[int] = None
    obs_height: Optional[int] = None
    action_space: Optional[gym.spaces.Space] = None
    feature_normalizations: Optional[dict[int, float]] = None
    clip_range: Optional[float] = 0.1  # What should be the default value?
    device: Optional[str] = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
    )
