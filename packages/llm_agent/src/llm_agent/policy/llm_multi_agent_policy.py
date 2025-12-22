"""LLM-based multi-agent policy base class for MettaGrid."""

import atexit
import os
import sys
from typing import Literal

from llm_agent.cost_tracker import CostTracker
from llm_agent.model_config import validate_model_context
from llm_agent.policy.llm_agent_policy import LLMAgentPolicy
from llm_agent.providers import (
    ensure_ollama_model,
    select_anthropic_model,
    select_openai_model,
)
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class LLMMultiAgentPolicy(MultiAgentPolicy):
    """LLM-based multi-agent policy for MettaGrid."""

    short_names = ["llm"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        provider: Literal["openai", "anthropic", "ollama"] = "openai",
        model: str | None = None,
        temperature: float = 0.7,
        verbose: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        debug_summary_interval: int = 0,
        mg_cfg=None,
    ):
        """Initialize LLM multi-agent policy.

        Args:
            policy_env_info: Policy environment interface
            provider: LLM provider ("openai", "anthropic", or "ollama")
            model: Model name (defaults based on provider)
            temperature: Sampling temperature for LLM
            verbose: If True, print human-readable observation debug info (default: True)
            context_window_size: Number of steps before resending basic info (default: 20)
            summary_interval: Number of steps between history summaries (default: 5)
            debug_summary_interval: Steps between LLM debug summaries written to file (0=disabled, e.g., 100)
            mg_cfg: Optional MettaGridConfig for extracting game-specific info (chest vibes, etc.)
        """
        super().__init__(policy_env_info)
        self.provider: Literal["openai", "anthropic", "ollama"] = provider
        self.temperature = temperature
        self.cost_tracker = CostTracker()  # Singleton - shared across all policy instances
        # Handle string "true"/"false" from CLI kwargs
        if isinstance(verbose, str):
            self.verbose = verbose.lower() not in ("false", "0", "no", "")
        else:
            self.verbose = bool(verbose)
        self.context_window_size = context_window_size
        self.summary_interval = int(summary_interval) if isinstance(summary_interval, str) else summary_interval
        self.debug_summary_interval = (
            int(debug_summary_interval) if isinstance(debug_summary_interval, str) else debug_summary_interval
        )
        self.mg_cfg = mg_cfg

        # Check API key before model selection for paid providers
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            print(
                "\n\033[1;31mError:\033[0m OPENAI_API_KEY environment variable is not set.\n\n"
                "To use OpenAI GPT models, you need to:\n"
                "  1. Get an API key from https://platform.openai.com/api-keys\n"
                "  2. Export it in your terminal:\n"
                "     export OPENAI_API_KEY='your-api-key-here'\n\n"
                "Alternatively, use local Ollama (free):\n"
                "  cogames play -m <mission> -p class=llm-ollama\n"
            )
            sys.exit(1)
        elif provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            print(
                "\n\033[1;31mError:\033[0m ANTHROPIC_API_KEY environment variable is not set.\n\n"
                "To use Anthropic Claude models, you need to:\n"
                "  1. Get an API key from https://console.anthropic.com/settings/keys\n"
                "  2. Export it in your terminal:\n"
                "     export ANTHROPIC_API_KEY='your-api-key-here'\n\n"
                "Alternatively, use local Ollama (free):\n"
                "  cogames play -m <mission> -p class=llm-ollama\n"
            )
            sys.exit(1)

        # Select model once for all agents if not specified
        if model is None:
            if provider == "openai":
                self.model = select_openai_model()
            elif provider == "anthropic":
                self.model = select_anthropic_model()
            elif provider == "ollama":
                self.model = ensure_ollama_model(None)
            else:
                self.model = None
        else:
            self.model = model

        # Validate model context window is sufficient for the config
        if self.model:
            validate_model_context(
                model=self.model,
                context_window_size=self.context_window_size,
                summary_interval=self.summary_interval,
            )

        # Register atexit handler to print costs when program ends (for paid APIs only)
        if provider in ("openai", "anthropic") and not hasattr(LLMMultiAgentPolicy, "_atexit_registered"):
            atexit.register(self.cost_tracker.print_summary)
            LLMMultiAgentPolicy._atexit_registered = True

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Create an LLM agent policy for a specific agent.

        Args:
            agent_id: Agent ID

        Returns:
            LLMAgentPolicy instance
        """
        return LLMAgentPolicy(
            self.policy_env_info,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            verbose=self.verbose,
            context_window_size=self.context_window_size,
            summary_interval=self.summary_interval,
            debug_summary_interval=self.debug_summary_interval,
            mg_cfg=self.mg_cfg,
            agent_id=agent_id,
        )
