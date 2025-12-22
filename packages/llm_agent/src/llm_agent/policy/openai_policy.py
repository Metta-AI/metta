"""OpenAI-based policies for MettaGrid."""

import atexit
import os
import sys

from llm_agent.cost_tracker import CostTracker
from llm_agent.model_config import validate_model_context
from llm_agent.policy.llm_agent_policy import LLMAgentPolicy
from llm_agent.providers import select_openai_model
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class OpenAIAgentPolicy(LLMAgentPolicy):
    """OpenAI GPT-based agent policy."""

    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _call(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        """Call OpenAI API and return (response, input_tokens, output_tokens)."""
        is_gpt5_or_o1 = self.model.startswith("gpt-5") or self.model.startswith("o1")
        params = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": 150 if is_gpt5_or_o1 else None,
            "max_tokens": None if is_gpt5_or_o1 else 150,
            "temperature": None if is_gpt5_or_o1 else self.temperature,
        }
        params = {k: v for k, v in params.items() if v is not None}

        response = self.client.chat.completions.create(**params)
        raw = response.choices[0].message.content or "noop"
        usage = response.usage
        return raw, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


class OpenAIMultiAgentPolicy(MultiAgentPolicy):
    """OpenAI GPT-based multi-agent policy for MettaGrid."""

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
        super().__init__(policy_env_info)
        self.temperature = temperature
        self.cost_tracker = CostTracker()
        self.verbose = self._parse_verbose(verbose)
        self.context_window_size = context_window_size
        self.summary_interval = int(summary_interval) if isinstance(summary_interval, str) else summary_interval
        self.debug_summary_interval = (
            int(debug_summary_interval) if isinstance(debug_summary_interval, str) else debug_summary_interval
        )
        self.mg_cfg = mg_cfg

        if not os.getenv("OPENAI_API_KEY"):
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

        self.model = model or select_openai_model()
        self._validate_and_register()

    @staticmethod
    def _parse_verbose(verbose: bool | str) -> bool:
        if isinstance(verbose, str):
            return verbose.lower() not in ("false", "0", "no", "")
        return bool(verbose)

    def _validate_and_register(self) -> None:
        if self.model:
            validate_model_context(
                model=self.model,
                context_window_size=self.context_window_size,
                summary_interval=self.summary_interval,
            )
        if not hasattr(OpenAIMultiAgentPolicy, "_atexit_registered"):
            atexit.register(self.cost_tracker.print_summary)
            OpenAIMultiAgentPolicy._atexit_registered = True

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return OpenAIAgentPolicy(
            self.policy_env_info,
            model=self.model,
            temperature=self.temperature,
            verbose=self.verbose,
            context_window_size=self.context_window_size,
            summary_interval=self.summary_interval,
            debug_summary_interval=self.debug_summary_interval,
            mg_cfg=self.mg_cfg,
            agent_id=agent_id,
        )


# Backwards compatibility alias
LLMGPTMultiAgentPolicy = OpenAIMultiAgentPolicy
