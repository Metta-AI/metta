"""Factory functions for creating Metta agents."""

from typing import Dict, Type

import gymnasium as gym

from .attention import AttentionAgent
from .base_agent import BaseAgent
from .large_cnn import LargeCNNAgent
from .multi_head_attention import MultiHeadAttentionAgent
from .simple_cnn import SimpleCNNAgent

# Registry of available agents
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "simple_cnn": SimpleCNNAgent,
    "large_cnn": LargeCNNAgent,
    "attention": AttentionAgent,
    "multi_head_attention": MultiHeadAttentionAgent,
}


def create_agent(
    agent_name: str,
    obs_space: gym.Space,
    action_space: gym.Space,
    obs_width: int,
    obs_height: int,
    feature_normalizations: dict,
    device: str = "cuda",
    **kwargs,
) -> BaseAgent:
    """Create an agent by name.

    Args:
        agent_name: Name of the agent type
        obs_space: Observation space
        action_space: Action space
        obs_width: Width of observations
        obs_height: Height of observations
        feature_normalizations: Feature normalization config
        device: Device to place agent on
        **kwargs: Additional agent-specific arguments

    Returns:
        Instantiated agent

    Raises:
        ValueError: If agent_name is not recognized
    """
    if agent_name not in AGENT_REGISTRY:
        available = ", ".join(AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown agent: {agent_name}. Available agents: {available}")

    agent_class = AGENT_REGISTRY[agent_name]
    return agent_class(
        obs_space=obs_space,
        action_space=action_space,
        obs_width=obs_width,
        obs_height=obs_height,
        feature_normalizations=feature_normalizations,
        device=device,
        **kwargs,
    )


def list_agents() -> list[str]:
    """List all available agent types."""
    return sorted(AGENT_REGISTRY.keys())


__all__ = ["create_agent", "list_agents", "AGENT_REGISTRY"]
