from typing import ClassVar, Optional, Union

import gymnasium as gym
from pydantic import ConfigDict

from metta.common.util.config import Config


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
