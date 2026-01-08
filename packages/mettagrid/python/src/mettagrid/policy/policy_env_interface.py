"""Policy environment interface for providing environment information to policies."""

import json
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.config.mettagrid_config import ActionsConfig, MettaGridConfig
from mettagrid.mettagrid_c import dtype_observations


class PolicyEnvInterface(BaseModel):
    """Interface providing environment information needed by policies.

    This class encapsulates the environment configuration details that policies
    need to initialize their networks, such as observation dimensions, action spaces,
    and agent counts.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("observation_space")
    def serialize_observation_space(self, v: gym.spaces.Box) -> dict[str, Any]:
        # gym.spaces.Box doesn't have a json serializer, so this attempts to implement it
        return {
            "low": v.low.tolist(),
            "high": v.high.tolist(),
            "shape": list(v.shape),
            "dtype": str(v.dtype),
        }

    @field_validator("observation_space", mode="before")
    @classmethod
    def validate_observation_space(cls, v: Any) -> gym.spaces.Box:
        if isinstance(v, gym.spaces.Box):
            return v
        if isinstance(v, dict):
            # gym.spaces.Box doesn't have a json deserializer, so this attempts to implement it
            dtype = np.dtype(v["dtype"]) if isinstance(v["dtype"], str) else v["dtype"]
            return gym.spaces.Box(
                low=np.array(v["low"]),
                high=np.array(v["high"]),
                shape=tuple(v["shape"]),
                dtype=dtype,  # type: ignore[arg-type]
            )
        raise ValueError(f"Cannot convert {type(v)} to gym.spaces.Box")

    @field_serializer("action_space")
    def serialize_action_space(self, v: gym.spaces.Discrete) -> dict[str, Any]:
        # gym.spaces.Discrete doesn't have a json serializer, so this attempts to implement it
        return {"n": int(v.n), "start": int(v.start)}

    @field_validator("action_space", mode="before")
    @classmethod
    def validate_action_space(cls, v: Any) -> gym.spaces.Discrete:
        if isinstance(v, gym.spaces.Discrete):
            return v
        if isinstance(v, dict):
            # gym.spaces.Discrete doesn't have a json deserializer, so this attempts to implement it
            return gym.spaces.Discrete(n=v["n"], start=v.get("start", 0))
        raise ValueError(f"Cannot convert {type(v)} to gym.spaces.Discrete")

    obs_features: list[ObservationFeatureSpec]
    tags: list[str]
    actions: ActionsConfig
    num_agents: int
    observation_space: gym.spaces.Box
    action_space: gym.spaces.Discrete
    obs_width: int
    obs_height: int
    assembler_protocols: list  # Assembler protocols for recipe initialization
    tag_id_to_name: dict[int, str]  # Tag ID to name mapping for observation parsing

    @property
    def action_names(self) -> list[str]:
        """Expose action names for policies that expect a flat list."""
        return [action.name for action in self.actions.actions()]

    @staticmethod
    def from_mg_cfg(mg_cfg: MettaGridConfig) -> "PolicyEnvInterface":
        """Create PolicyEnvInterface from MettaGridConfig.

        Args:
            mg_cfg: The MettaGrid configuration

        Returns:
            A PolicyEnvInterface instance with environment information
        """
        # Extract assembler protocols if available
        assembler_protocols = []
        assembler_config = mg_cfg.game.objects.get("assembler")
        if assembler_config and hasattr(assembler_config, "protocols"):
            assembler_protocols = assembler_config.protocols

        # Get tag ID to name mapping from id_map
        id_map = mg_cfg.game.id_map()
        tag_names_list = id_map.tag_names()
        # Tag IDs are assigned based on alphabetical order (index in sorted list)
        tag_id_to_name = {i: name for i, name in enumerate(tag_names_list)}

        return PolicyEnvInterface(
            obs_features=id_map.features(),
            tags=tag_names_list,
            actions=mg_cfg.game.actions,
            num_agents=mg_cfg.game.num_agents,
            observation_space=gym.spaces.Box(
                0, 255, (mg_cfg.game.obs.num_tokens, mg_cfg.game.obs.token_dim), dtype=dtype_observations
            ),
            action_space=gym.spaces.Discrete(len(mg_cfg.game.actions.actions())),
            obs_width=mg_cfg.game.obs.width,
            obs_height=mg_cfg.game.obs.height,
            assembler_protocols=assembler_protocols,
            tag_id_to_name=tag_id_to_name,
        )

    def to_json(self) -> str:
        """Convert PolicyEnvInterface to JSON."""
        # TODO: Andre: replace this with `.model_dump(mode="json")`, now that it supports all fields
        payload = self.model_dump(mode="json", include={"num_agents", "obs_width", "obs_height", "tags"})
        payload["actions"] = self.action_names
        payload["obs_features"] = [feature.model_dump(mode="json") for feature in self.obs_features]
        payload["assembler_protocols"] = [
            {
                "input_resources": getattr(proto, "input_resources", {}) or {},
                "output_resources": getattr(proto, "output_resources", {}) or {},
            }
            for proto in self.assembler_protocols
        ]
        return json.dumps(payload)
