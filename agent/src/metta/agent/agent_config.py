from typing import Any, ClassVar, Optional, Union

import gymnasium as gym
import torch.nn as nn
from omegaconf import DictConfig
from pydantic import ConfigDict
from tensordict import TensorDict

from metta.common.util.config import Config
from metta.rl.system_config import SystemConfig


class AgentConfig(Config):
    env: Any
    system_cfg: SystemConfig
    agent_cfg: str
    policy: Optional[nn.Module] = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
    )


class AgentOutput(Config):
    td: TensorDict
    state: Optional[Any] = None
    metadata: Optional[dict[str, Any]] = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
    )


class ComponentPolicyConfig(Config):
    obs_space: Optional[Union[gym.spaces.Space, gym.spaces.Dict]] = None
    obs_width: Optional[int] = None
    obs_height: Optional[int] = None
    action_space: Optional[gym.spaces.Space] = None
    feature_normalizations: Optional[dict[int, float]] = None
    device: Optional[str] = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
    )
