from pydantic import BaseModel, model_validator, ConfigDict, Field, field_serializer
from gymnasium.spaces import MultiDiscrete, Discrete, Box
from typing import Dict, Any, Union, Optional
import json
import numpy as np

class AgentEnvConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    obs_width: int
    obs_height: int
    feature_normalizations: Dict[int, float]
    single_action_space: Any = Field(validate_default=False)
    max_action_args: list[int]
    single_observation_space: Any = Field(validate_default=False)
    agent_config: Any = Field(validate_default=False)

    @field_serializer('single_action_space')
    def serialize_action_space(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, MultiDiscrete):
            return {
                "_type": "MultiDiscrete",
                "nvec": value.nvec.tolist(),
                "dtype": str(value.dtype) if hasattr(value, 'dtype') else None
            }
        return value

    @field_serializer('single_observation_space')
    def serialize_observation_space(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, Box):
            return {
                "_type": "Box",
                "low": value.low.tolist(),
                "high": value.high.tolist(),
                "shape": list(value.shape),
                "dtype": str(value.dtype)
            }
        return value

    @model_validator(mode='before')
    @classmethod
    def preprocess_and_deserialize(cls, values: Any) -> Any:
        """Handle preprocessing and deserialization"""
        if not isinstance(values, dict):
            return values

        values = values.copy()

        # Handle feature_normalizations
        if 'feature_normalizations' in values:
            feature_norm = values['feature_normalizations']
            if isinstance(feature_norm, (list, np.ndarray)):
                values['feature_normalizations'] = {
                    i: float(val) for i, val in enumerate(feature_norm)
                }
            elif isinstance(feature_norm, dict):
                values['feature_normalizations'] = {
                    int(k): float(v) for k, v in feature_norm.items()
                }

        # Handle gym spaces deserialization
        if 'single_action_space' in values:
            space = values['single_action_space']
            if isinstance(space, dict) and space.get("_type") == "MultiDiscrete":
                values['single_action_space'] = MultiDiscrete(space["nvec"])

        if 'single_observation_space' in values:
            space = values['single_observation_space']
            if isinstance(space, dict) and space.get("_type") == "Box":
                values['single_observation_space'] = Box(
                    low=np.array(space["low"], dtype=space["dtype"]),
                    high=np.array(space["high"], dtype=space["dtype"]),
                    shape=tuple(space["shape"])
                )

        return values

    @classmethod
    def create(cls, env, agent_config):
        return AgentEnvConfig(
            obs_width=env.obs_width,
            obs_height=env.obs_height,
            feature_normalizations=env.feature_normalizations,
            single_action_space=env.single_action_space,
            max_action_args=env.max_action_args,
            single_observation_space=env.single_observation_space,
            agent_config=agent_config,
        )
