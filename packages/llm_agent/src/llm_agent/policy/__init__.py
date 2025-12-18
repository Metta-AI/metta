"""LLM policy classes for MettaGrid.

This module is auto-discovered by mettagrid.policy.loader for policy registration.
"""

from llm_agent.policy.llm_policy import (
    LLMAgentPolicy,
    LLMClaudeMultiAgentPolicy,
    LLMGPTMultiAgentPolicy,
    LLMMultiAgentPolicy,
    LLMOllamaMultiAgentPolicy,
)

__all__ = [
    "LLMAgentPolicy",
    "LLMMultiAgentPolicy",
    "LLMGPTMultiAgentPolicy",
    "LLMClaudeMultiAgentPolicy",
    "LLMOllamaMultiAgentPolicy",
]
