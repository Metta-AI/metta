"""LLM policy classes for MettaGrid.

This module is auto-discovered by mettagrid.policy.loader for policy registration.
"""

from llm_agent.policy.anthropic_policy import (
    AnthropicAgentPolicy,
    AnthropicMultiAgentPolicy,
    LLMClaudeMultiAgentPolicy,
)
from llm_agent.policy.llm_agent_policy import LLMAgentPolicy
from llm_agent.policy.ollama_policy import (
    LLMOllamaMultiAgentPolicy,
    OllamaAgentPolicy,
    OllamaMultiAgentPolicy,
)
from llm_agent.policy.openai_policy import (
    LLMGPTMultiAgentPolicy,
    OpenAIAgentPolicy,
    OpenAIMultiAgentPolicy,
)

# Default LLMMultiAgentPolicy points to OpenAI
LLMMultiAgentPolicy = OpenAIMultiAgentPolicy

__all__ = [
    # Base class
    "LLMAgentPolicy",
    # OpenAI
    "OpenAIAgentPolicy",
    "OpenAIMultiAgentPolicy",
    "LLMGPTMultiAgentPolicy",
    # Anthropic
    "AnthropicAgentPolicy",
    "AnthropicMultiAgentPolicy",
    "LLMClaudeMultiAgentPolicy",
    # Ollama
    "OllamaAgentPolicy",
    "OllamaMultiAgentPolicy",
    "LLMOllamaMultiAgentPolicy",
    # Default alias
    "LLMMultiAgentPolicy",
]
