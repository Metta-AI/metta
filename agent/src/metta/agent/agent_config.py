"""Agent configuration helpers for selecting policy architectures."""

from typing import Callable, Dict, Literal

from metta.agent.policies.fast import FastConfig
from metta.agent.policies.fast_dynamics import FastDynamicsConfig
from metta.agent.policies.fast_td_transformer import FastTransformerConfig
from metta.agent.policies.vit import ViTSmallConfig
from mettagrid.config import Config

PolicyFactory = Callable[[], Config]


class AgentConfig(Config):
    """Configuration for selecting a policy architecture."""

    name: Literal[
        "fast",
        "fast_dynamics",
        "fast_transformer",
        "vit_small",
    ] = "fast"


ARCHITECTURE_REGISTRY: Dict[str, PolicyFactory] = {
    "fast": FastConfig,
    "fast_dynamics": FastDynamicsConfig,
    "fast_transformer": FastTransformerConfig,
    "vit_small": ViTSmallConfig,
}


def create_agent(config: AgentConfig, env_metadata) -> Config:
    """Instantiate the configured policy architecture for the given environment."""

    if config.name not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"Unknown agent: '{config.name}'. Available: {list(ARCHITECTURE_REGISTRY.keys())}")

    architecture_cfg = ARCHITECTURE_REGISTRY[config.name]()
    return architecture_cfg.make_policy(env_metadata)
