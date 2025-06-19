"""Metta agent implementations as torch.nn.Module classes."""

from .attention import AttentionAgent
from .base_agent import BaseAgent
from .factory import create_agent, list_agents, register_agent
from .large_cnn import LargeCNNAgent
from .multi_head_attention import MultiHeadAttentionAgent
from .simple_cnn import SimpleCNNAgent

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
