"""LLM policy classes for MettaGrid.

This module is auto-discovered by mettagrid.policy.loader for policy registration.
"""

from llm_agent.policy.llm_agent_policy import LLMAgentPolicy
from llm_agent.policy.llm_multi_agent_policy import LLMMultiAgentPolicy
from llm_agent.policy.llm_provider_policies import (
    LLMClaudeMultiAgentPolicy,
    LLMGPTMultiAgentPolicy,
    LLMOllamaMultiAgentPolicy,
)

__all__ = [
    "LLMAgentPolicy",
    "LLMMultiAgentPolicy",
    "LLMGPTMultiAgentPolicy",
    "LLMClaudeMultiAgentPolicy",
    "LLMOllamaMultiAgentPolicy",
]
