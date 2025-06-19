"""Metta agents module - provides neural network agents for reinforcement learning."""

from .attention import AttentionAgent
from .base_agent import BaseAgent
from .factory import AGENT_REGISTRY, create_agent, list_agents
from .large_cnn import LargeCNNAgent
from .multi_head_attention import MultiHeadAttentionAgent
from .simple_cnn import SimpleCNNAgent


def register_agent(name: str, agent_class: type) -> None:
    """Register a custom agent class.

    Args:
        name: Name to register the agent under
        agent_class: Agent class (must inherit from BaseAgent)

    Example:
        >>> class MyAgent(BaseAgent):
        ...     pass
        >>> register_agent("my_agent", MyAgent)
        >>> agent = create_agent("my_agent", ...)
    """
    if not issubclass(agent_class, BaseAgent):
        raise TypeError(f"Agent class must inherit from BaseAgent, got {agent_class}")
    AGENT_REGISTRY[name] = agent_class


__all__ = [
    "BaseAgent",
    "SimpleCNNAgent",
    "LargeCNNAgent",
    "AttentionAgent",
    "MultiHeadAttentionAgent",
    "create_agent",
    "register_agent",
    "list_agents",
]
