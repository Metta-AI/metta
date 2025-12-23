"""Ollama-based policies for MettaGrid."""

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

from llm_agent.cost_tracker import CostTracker
from llm_agent.model_config import validate_model_context
from llm_agent.policy.llm_agent_policy import LLMAgentPolicy
from llm_agent.providers import ensure_ollama_model
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class OllamaAgentPolicy(LLMAgentPolicy):
    """Ollama local LLM-based agent policy."""

    def _init_client(self) -> None:
        """Initialize the Ollama client (uses OpenAI-compatible API)."""
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def _call(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        openai_messages: list[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content=m["content"])
            if m["role"] == "user"
            else ChatCompletionAssistantMessageParam(role="assistant", content=m["content"])
            for m in messages
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=150,
        )

        message = response.choices[0].message
        raw = message.content or ""

        if not raw and hasattr(message, "reasoning") and message.reasoning:
            raw = message.reasoning

        if not raw:
            print(f"[ERROR] Ollama empty response for Agent {self.agent_id}")
            raw = "noop"

        usage = response.usage
        return raw, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


class OllamaMultiAgentPolicy(MultiAgentPolicy):
    """Ollama local LLM-based multi-agent policy for MettaGrid."""

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

        self.model = ensure_ollama_model(model)
        self._validate()

    @staticmethod
    def _parse_verbose(verbose: bool | str) -> bool:
        if isinstance(verbose, str):
            return verbose.lower() not in ("false", "0", "no", "")
        return bool(verbose)

    def _validate(self) -> None:
        if self.model:
            validate_model_context(
                model=self.model,
                context_window_size=self.context_window_size,
                summary_interval=self.summary_interval,
            )

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return OllamaAgentPolicy(
            self.policy_env_info,
            model=self.model,
            temperature=self.temperature,
            verbose=self.verbose,
            context_window_size=self.context_window_size,
            summary_interval=self.summary_interval,
            debug_summary_interval=self.debug_summary_interval,
            agent_id=agent_id,
        )
