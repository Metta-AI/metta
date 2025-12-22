"""Provider-specific LLM policy classes for MettaGrid."""

from llm_agent.policy.llm_multi_agent_policy import LLMMultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class LLMGPTMultiAgentPolicy(LLMMultiAgentPolicy):
    """OpenAI GPT-based policy for MettaGrid."""

    short_names = ["llm-openai"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7,
        verbose: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        debug_summary_interval: int = 0,
        mg_cfg=None,
    ):
        super().__init__(
            policy_env_info,
            provider="openai",
            model=model,
            temperature=temperature,
            verbose=verbose,
            context_window_size=context_window_size,
            summary_interval=summary_interval,
            debug_summary_interval=debug_summary_interval,
            mg_cfg=mg_cfg,
        )


class LLMClaudeMultiAgentPolicy(LLMMultiAgentPolicy):
    """Anthropic Claude-based policy for MettaGrid."""

    short_names = ["llm-anthropic"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7,
        verbose: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        debug_summary_interval: int = 0,
        mg_cfg=None,
    ):
        super().__init__(
            policy_env_info,
            provider="anthropic",
            model=model,
            temperature=temperature,
            verbose=verbose,
            context_window_size=context_window_size,
            summary_interval=summary_interval,
            debug_summary_interval=debug_summary_interval,
            mg_cfg=mg_cfg,
        )


class LLMOllamaMultiAgentPolicy(LLMMultiAgentPolicy):
    """Ollama local LLM-based policy for MettaGrid."""

    short_names = ["llm-ollama"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7,
        verbose: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        debug_summary_interval: int = 0,
        mg_cfg=None,
    ):
        super().__init__(
            policy_env_info,
            provider="ollama",
            model=model,
            temperature=temperature,
            verbose=verbose,
            context_window_size=context_window_size,
            summary_interval=summary_interval,
            debug_summary_interval=debug_summary_interval,
            mg_cfg=mg_cfg,
        )
