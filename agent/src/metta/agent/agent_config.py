from typing import ClassVar, Literal, Optional, Union

import gymnasium as gym
import torch.nn as nn
from pydantic import ConfigDict

from metta.common.util.config import Config


class AgentConfig(Config):
    agent: Literal[
        "fast",
        "latent_attn_med",
        "latent_attn_small",
        "latent_attn_tiny",
        "fast.py",
        "latent_attn_med.py",
        "latent_attn_small.py",
        "latent_attn_tiny.py",
        "vanilla.py",
    ]
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
