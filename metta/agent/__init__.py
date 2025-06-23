"""Metta agent module - provides neural network agents and utilities for reinforcement learning."""

# Agent classes
from .attention import AttentionAgent
from .base_agent import BaseAgent
from .distributed import DistributedMettaAgent
from .factory import AGENT_REGISTRY, create_agent, list_agents
from .large_cnn import LargeCNNAgent
from .metta_agent import MettaAgent, make_policy
from .multi_head_attention import MultiHeadAttentionAgent

# Core modules
from .policy_state import PolicyState
from .policy_store import PolicyStore
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
    # Agent classes
    "BaseAgent",
    "SimpleCNNAgent",
    "LargeCNNAgent",
    "AttentionAgent",
    "MultiHeadAttentionAgent",
    "DistributedMettaAgent",
    "MettaAgent",  # Deprecated
    # Factory functions
    "create_agent",
    "register_agent",
    "list_agents",
    "make_policy",
    # Core modules
    "PolicyState",
    "PolicyStore",
]
