"""
Agent configuration following the dehydration branch Config pattern.
"""

from typing import Literal, Optional

# ComponentPolicy implementations (modular architecture)
from metta.agent.component_policies.fast import Fast as ComponentFast
from metta.agent.component_policies.latent_attn_med import LatentAttnMed as ComponentLatentAttnMed
from metta.agent.component_policies.latent_attn_small import LatentAttnSmall as ComponentLatentAttnSmall
from metta.agent.component_policies.latent_attn_tiny import LatentAttnTiny as ComponentLatentAttnTiny

# PyTorch implementations (vanilla models)
from metta.agent.pytorch.example import Example
from metta.agent.pytorch.fast import Fast as PyTorchFast
from metta.agent.pytorch.latent_attn_med import LatentAttnMed as PyTorchLatentAttnMed
from metta.agent.pytorch.latent_attn_small import LatentAttnSmall as PyTorchLatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import LatentAttnTiny as PyTorchLatentAttnTiny

# New TD policy
from metta.common.config import Config


class AgentConfig(Config):
    """Configuration for agent architecture selection."""

    name: Literal[
        "fast",
        "latent_attn_tiny",
        "latent_attn_small",
        "latent_attn_med",
        "pytorch/example",
        "pytorch/fast",
        "pytorch/latent_attn_tiny",
        "pytorch/latent_attn_small",
        "pytorch/latent_attn_med",
    ] = "fast"

    clip_range: float = 0
    analyze_weights_interval: int = 300
    policy_config: Optional[Config] = None


# Registry mapping agent names to classes. This will be removed in the future.
AGENT_REGISTRY = {
    "fast": ComponentFast,
    "latent_attn_tiny": ComponentLatentAttnTiny,
    "latent_attn_small": ComponentLatentAttnSmall,
    "latent_attn_med": ComponentLatentAttnMed,
    "pytorch/example": Example,
    "pytorch/fast": PyTorchFast,
    "pytorch/latent_attn_tiny": PyTorchLatentAttnTiny,
    "pytorch/latent_attn_small": PyTorchLatentAttnSmall,
    "pytorch/latent_attn_med": PyTorchLatentAttnMed,
}


def create_agent(
    config: AgentConfig,
    # obs_space=None,
    # obs_width=None,
    # obs_height=None,
    # feature_normalizations=None,
    env=None,
):
    """Create an agent instance from configuration."""
    # New, preferred pattern: config objects are responsible for instantiation. This will allow us to remove the agent
    # registry. Agent config objects must therefore have an `instantiate` method.

    # package obs information
    # obs_meta = {
    #     "obs_space": obs_space,
    #     "obs_width": obs_width,
    #     "obs_height": obs_height,
    #     "feature_normalizations": feature_normalizations,
    # }

    if config.policy_config is not None:
        if hasattr(config.policy_config, "instantiate"):
            policy = config.policy_config.instantiate(env=env)
            return policy
        else:
            raise AttributeError(
                f"Provided policy config {type(config.policy_config).__name__} does not have an 'instantiate' \
                method."
            )

    # Backwards compatibility for old agent types. This will be removed in the future.
    if config.name not in AGENT_REGISTRY:
        raise ValueError(
            f"Agent name '{config.name}' not specified and no policy config provided, or name not in registry. "
            f"Available names: {list(AGENT_REGISTRY.keys())}"
        )

    AgentClass = AGENT_REGISTRY[config.name]

    # PyTorch models use env, ComponentPolicies use structured parameters. This will be removed in the future.
    if config.name.startswith("pytorch/"):
        return AgentClass(
            env=env,
            clip_range=config.clip_range,
            analyze_weights_interval=config.analyze_weights_interval,
        )
    else:
        return AgentClass(
            # obs_space=obs_space,
            # obs_width=obs_width,
            # obs_height=obs_height,
            # feature_normalizations=feature_normalizations,
            config={
                "clip_range": config.clip_range,
                "analyze_weights_interval": config.analyze_weights_interval,
            },
        )
