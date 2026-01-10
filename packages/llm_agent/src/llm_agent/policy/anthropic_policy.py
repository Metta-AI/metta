"""Anthropic-based policies for MettaGrid."""

import atexit
import os
import sys

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock

from llm_agent.cost_tracker import CostTracker
from llm_agent.model_config import validate_model_context
from llm_agent.policy.llm_agent_policy import LLMAgentPolicy
from llm_agent.providers import select_anthropic_model
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class AnthropicAgentPolicy(LLMAgentPolicy):
    """Anthropic Claude-based agent policy."""

    def _init_client(self) -> None:
        """Initialize the Anthropic client."""
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _call(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        anthropic_messages: list[MessageParam] = [
            MessageParam(role="user", content=m["content"])
            if m["role"] == "user"
            else MessageParam(role="assistant", content=m["content"])
            for m in messages
        ]
        response = self.client.messages.create(
            model=self.model,
            messages=anthropic_messages,
            temperature=self.temperature,
            max_tokens=150,
        )

        raw = "noop"
        for block in response.content:
            if isinstance(block, TextBlock):
                raw = block.text
                break

        usage = response.usage
        return raw, usage.input_tokens, usage.output_tokens


class AnthropicMultiAgentPolicy(MultiAgentPolicy):
    """Anthropic Claude-based multi-agent policy for MettaGrid."""

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

        if not os.getenv("ANTHROPIC_API_KEY"):
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

        self.model = model or select_anthropic_model()
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
        if not hasattr(AnthropicMultiAgentPolicy, "_atexit_registered"):
            atexit.register(self.cost_tracker.print_summary)
            AnthropicMultiAgentPolicy._atexit_registered = True

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return AnthropicAgentPolicy(
            self.policy_env_info,
            model=self.model,
            temperature=self.temperature,
            verbose=self.verbose,
            context_window_size=self.context_window_size,
            summary_interval=self.summary_interval,
            debug_summary_interval=self.debug_summary_interval,
            agent_id=agent_id,
        )
