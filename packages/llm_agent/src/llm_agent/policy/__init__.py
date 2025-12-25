"""LLM policy classes for MettaGrid.

This module is auto-discovered by mettagrid.policy.loader for policy registration.
"""

from llm_agent.policy.anthropic_policy import AnthropicAgentPolicy, AnthropicMultiAgentPolicy
from llm_agent.policy.llm_agent_policy import LLMAgentPolicy
from llm_agent.policy.ollama_policy import OllamaAgentPolicy, OllamaMultiAgentPolicy
from llm_agent.policy.openai_policy import OpenAIAgentPolicy, OpenAIMultiAgentPolicy

# Default LLMMultiAgentPolicy points to OpenAI
LLMMultiAgentPolicy = OpenAIMultiAgentPolicy

__all__ = [
    # Base class
    "LLMAgentPolicy",
    # OpenAI
    "OpenAIAgentPolicy",
    "OpenAIMultiAgentPolicy",
    # Anthropic
    "AnthropicAgentPolicy",
    "AnthropicMultiAgentPolicy",
    # Ollama
    "OllamaAgentPolicy",
    "OllamaMultiAgentPolicy",
    # Default alias
    "LLMMultiAgentPolicy",
]
