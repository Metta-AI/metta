"""LLM-based policy for MettaGrid using GPT or Claude."""

import atexit
import json
import logging
import os
import random
import subprocess
import sys
from typing import Literal

from mettagrid.policy.observation_debugger import ObservationDebugger
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

logger = logging.getLogger(__name__)


def calculate_llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an LLM API call based on model and token usage.

    Prices are per 1M tokens as of January 2025.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    # OpenAI pricing (per 1M tokens)
    openai_prices = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-5.1": {"input": 5.00, "output": 15.00},
    }

    # Anthropic pricing (per 1M tokens)
    anthropic_prices = {
        "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "claude-opus-4-5": {"input": 15.00, "output": 75.00},
    }

    # Combine all pricing
    all_prices = {**openai_prices, **anthropic_prices}

    # Get pricing for the model
    if model not in all_prices:
        logger.warning(f"Unknown model '{model}' for cost calculation. Using default pricing.")
        # Default to GPT-4o-mini pricing if model is unknown
        prices = openai_prices["gpt-4o-mini"]
    else:
        prices = all_prices[model]

    # Calculate cost (prices are per 1M tokens, so divide by 1,000,000)
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]

    return input_cost + output_cost


# Model context window sizes (in tokens)
MODEL_CONTEXT_WINDOWS = {
    # OpenAI models
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-5.1": 128_000,
    # Anthropic models
    "claude-haiku-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    # Common Ollama models
    "llama3.2": 8_000,
    "llama3.2:1b": 8_000,
    "llama3.2:3b": 8_000,
    "llama3.1": 128_000,
    "llama3.1:8b": 128_000,
    "llama3.1:70b": 128_000,
    "qwen2.5": 32_000,
    "qwen2.5:7b": 32_000,
    "qwen2.5:14b": 32_000,
    "qwen2.5:32b": 32_000,
    "qwen2.5:72b": 128_000,
    "mistral": 32_000,
    "mixtral": 32_000,
    "gemma2": 8_000,
    "gemma2:2b": 8_000,
    "gemma2:9b": 8_000,
    "phi3": 128_000,
    "deepseek-r1": 64_000,
}

# Estimated tokens per step based on config
# Base prompt ~1500 tokens + ~100 tokens per conversation turn
TOKENS_PER_STEP_BASE = 1500
TOKENS_PER_CONVERSATION_TURN = 100


def get_model_context_window(model: str) -> int | None:
    """Get the context window size for a model.

    Args:
        model: Model name

    Returns:
        Context window size in tokens, or None if unknown
    """
    # Direct lookup
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]

    # Try partial match (e.g., "llama3.2:latest" -> "llama3.2")
    base_model = model.split(":")[0]
    if base_model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[base_model]

    return None


def estimate_required_context(context_window_size: int, summary_interval: int) -> int:
    """Estimate the required context window for the given config.

    Args:
        context_window_size: Number of steps before conversation reset
        summary_interval: Number of steps between summaries

    Returns:
        Estimated required context in tokens
    """
    # Base prompt + (context_window_size * tokens per turn) + buffer for summaries
    num_summaries = context_window_size // summary_interval
    summary_tokens = num_summaries * 150  # ~150 tokens per summary

    required = TOKENS_PER_STEP_BASE + (context_window_size * TOKENS_PER_CONVERSATION_TURN * 2) + summary_tokens

    # Add 50% safety margin
    return int(required * 1.5)


def validate_model_context(
    model: str,
    context_window_size: int | str,
    summary_interval: int | str,
) -> None:
    """Validate that the model has sufficient context for the config.

    Args:
        model: Model name
        context_window_size: Number of steps before conversation reset
        summary_interval: Number of steps between summaries

    Raises:
        SystemExit: If model context is insufficient
    """
    # Handle string inputs from CLI
    context_window_size = int(context_window_size) if isinstance(context_window_size, str) else context_window_size
    summary_interval = int(summary_interval) if isinstance(summary_interval, str) else summary_interval

    model_context = get_model_context_window(model)
    required_context = estimate_required_context(context_window_size, summary_interval)

    if model_context is None:
        # Unknown model - just warn
        print(
            f"\n\033[1;33mWarning:\033[0m Unknown model '{model}' - cannot verify context window.\n"
            f"Estimated requirement: ~{required_context:,} tokens for context_window_size={context_window_size}\n"
        )
        return

    if model_context < required_context:
        print(
            f"\n\033[1;31mError:\033[0m Model '{model}' has insufficient context window.\n\n"
            f"  Model context:    {model_context:,} tokens\n"
            f"  Required context: ~{required_context:,} tokens\n"
            f"  (context_window_size={context_window_size}, summary_interval={summary_interval})\n\n"
            f"Options:\n"
            f"  1. Use a smaller context window:\n"
            f"     kw.context_window_size=5,kw.summary_interval=5\n\n"
            f"  2. Use a model with larger context:\n"
        )
        # Suggest compatible models
        compatible = [
            (name, ctx) for name, ctx in MODEL_CONTEXT_WINDOWS.items()
            if ctx >= required_context
        ]
        compatible.sort(key=lambda x: x[1])
        for name, ctx in compatible[:5]:
            print(f"     - {name} ({ctx:,} tokens)")
        print()
        sys.exit(1)

    # Context is sufficient - show info
    usage_pct = (required_context / model_context) * 100
    if usage_pct > 50:
        print(
            f"\n\033[1;33mNote:\033[0m Model '{model}' context usage: ~{usage_pct:.0f}%\n"
            f"  ({required_context:,} / {model_context:,} tokens)\n"
        )


def _print_cost_summary_on_exit() -> None:
    """Print cost summary when program exits or is interrupted."""
    # Access the class through globals to avoid circular import
    try:
        llm_policy_class = globals().get("LLMAgentPolicy")
        if llm_policy_class is None:
            return

        summary = llm_policy_class.get_cost_summary()
        if summary["total_calls"] > 0:
            print("\n" + "=" * 60)
            print("LLM API USAGE SUMMARY")
            print("=" * 60)
            print(f"Total API calls: {summary['total_calls']}")
            print(f"Total tokens: {summary['total_tokens']:,}")
            print(f"  - Input tokens: {summary['total_input_tokens']:,}")
            print(f"  - Output tokens: {summary['total_output_tokens']:,}")
            print(f"Total cost: ${summary['total_cost']:.4f}")
            print("=" * 60 + "\n")
    except Exception:
        # Silently fail if there's any issue printing the summary
        pass

def check_ollama_available() -> bool:
    """Check if Ollama server is running.

    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        # Try to list models as a health check
        client.models.list()
        return True
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """List available Ollama models.

    Returns:
        List of model names
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        # Parse output: skip header line, extract model names
        lines = result.stdout.strip().split("\n")[1:]  # Skip header
        models = [line.split()[0] for line in lines if line.strip()]
        return models
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return []


def get_openai_models() -> list[tuple[str, str]]:
    """Get tested working OpenAI models.

    Returns:
        List of (model_name, description) tuples
    """
    return [
        ("gpt-4o-mini", "Cheapest - Fast and cost-effective"),
        ("gpt-4o", "Capable - Best GPT-4 model"),
        ("gpt-5.1", "Best - Latest GPT-5 for complex reasoning"),
    ]


def select_openai_model() -> str:
    """Prompt user to select an OpenAI model.

    Returns:
        Selected model name
    """
    models = get_openai_models()

    print("\n" + "=" * 60)
    print("Select OpenAI Model:")
    print("=" * 60)
    for idx, (model_name, description) in enumerate(models, 1):
        print(f"  [{idx}] {model_name}")
        print(f"      {description}")
    print("=" * 60)

    while True:
        try:
            selection = input(f"\nSelect a model (1-{len(models)}): ").strip()
            idx = int(selection) - 1
            if 0 <= idx < len(models):
                model = models[idx][0]
                print(f"\n✓ Selected: {model}\n")
                return model
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\n\n⚠️  No model selected. Exiting.\n")
            sys.exit(0)


def get_anthropic_models() -> list[tuple[str, str]]:
    """Get tested working Anthropic Claude models.

    Returns:
        List of (model_name, description) tuples
    """
    return [
        ("claude-haiku-4-5", "Cheapest - Fastest with near-frontier intelligence"),
        ("claude-sonnet-4-5", "Best - Smartest for complex agents & coding"),
        ("claude-opus-4-5", "Premium - Maximum intelligence & performance"),
    ]


def select_anthropic_model() -> str:
    """Prompt user to select an Anthropic Claude model.

    Returns:
        Selected model name
    """
    models = get_anthropic_models()

    print("\n" + "=" * 60)
    print("Select Claude Model:")
    print("=" * 60)
    for idx, (model_name, description) in enumerate(models, 1):
        print(f"  [{idx}] {model_name}")
        print(f"      {description}")
    print("=" * 60)

    while True:
        try:
            selection = input(f"\nSelect a model (1-{len(models)}): ").strip()
            idx = int(selection) - 1
            if 0 <= idx < len(models):
                model = models[idx][0]
                print(f"\n✓ Selected: {model}\n")
                return model
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\n\n⚠️  No model selected. Exiting.\n")
            sys.exit(0)


def ensure_ollama_model(model: str | None = None) -> str:
    """Ensure an Ollama model is available, pulling if necessary.

    Args:
        model: Model name to check/pull, or None to prompt user to select

    Returns:
        The model name that is available

    Raises:
        RuntimeError: If Ollama is not available or model pull fails
    """
    if not check_ollama_available():
        raise RuntimeError(
            "Ollama server is not running. Please start it with 'ollama serve' or install from https://ollama.ai"
        )

    available_models = list_ollama_models()

    # If no model specified, prompt user to select
    if model is None:
        if not available_models:
            # No models available, prompt user
            print("\n" + "=" * 60)
            print("⚠️  No Ollama models found!")
            print("=" * 60)
            print("\nOptions:")
            print("  1. Install default model (llama3.2) - ~2GB download")
            print("  2. Install a model manually with 'ollama pull <model>'")
            print("  3. Use llm-anthropic or llm-openai instead")
            print("=" * 60)

            try:
                response = input("\nInstall default model (llama3.2)? [y/N]: ").strip().lower()
                if response in ("y", "yes"):
                    model = "llama3.2"
                    print(f"\n📥 Pulling {model}...")
                    print("(This may take a few minutes...)\n")
                    subprocess.run(["ollama", "pull", model], check=True)
                    print(f"\n✓ Successfully installed {model}\n")
                    return model
                else:
                    print("\n" + "=" * 60)
                    print("To use Ollama:")
                    print("  1. Pull a model: ollama pull llama3.2")
                    print("  2. Run again: cogames play -m <mission> -p llm-ollama")
                    print("\nAlternatively, use cloud LLMs:")
                    print("  • cogames play -m <mission> -p llm-openai")
                    print("  • cogames play -m <mission> -p llm-anthropic")
                    print("=" * 60 + "\n")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\n\n⚠️  Cancelled by user.\n")
                sys.exit(0)

        # Show available models and prompt user to select
        print("\n" + "=" * 60)
        print("Available Ollama Models:")
        print("=" * 60)
        for idx, model_name in enumerate(available_models, 1):
            print(f"  [{idx}] {model_name}")
        print("=" * 60)

        while True:
            try:
                selection = input(f"\nSelect a model (1-{len(available_models)}): ").strip()
                idx = int(selection) - 1
                if 0 <= idx < len(available_models):
                    model = available_models[idx]
                    print(f"\n✓ Selected: {model}\n")
                    return model
                else:
                    print(f"Please enter a number between 1 and {len(available_models)}")
            except ValueError:
                print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\n\n⚠️  No model selected. Exiting.\n")
                sys.exit(0)

    # Model was explicitly specified, check if it's available
    if any(model in m for m in available_models):
        return model

    # Try to pull the specified model
    print(f"\nModel '{model}' not found. Pulling from Ollama...")
    try:
        subprocess.run(
            ["ollama", "pull", model],
            check=True,
            capture_output=False,  # Show progress
        )
        print(f"\n✓ Successfully pulled model: {model}\n")
        return model
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to pull Ollama model '{model}': {e}") from e


class LLMAgentPolicy(AgentPolicy):
    """Per-agent LLM policy that queries GPT or Claude for action selection."""

    # Class-level tracking for all instances
    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        provider: Literal["openai", "anthropic", "ollama"] = "openai",
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        mg_cfg = None,
        agent_id: int = 0,
    ):
        """Initialize LLM agent policy.

        Args:
            policy_env_info: Policy environment interface
            provider: LLM provider ("openai", "anthropic", or "ollama")
            model: Model name (defaults: gpt-4o-mini, claude-3-5-sonnet-20240620, or llama3.2 for ollama)
            temperature: Sampling temperature for LLM
            debug_mode: If True, print human-readable observation debug info (default: True)
            context_window_size: Number of steps before resending basic info (default: 20)
            summary_interval: Number of steps between history summaries (default: 5)
            mg_cfg: Optional MettaGridConfig for extracting game-specific info (chest vibes, etc.)
            agent_id: Agent ID for this policy instance (used for debug output filtering)
        """
        super().__init__(policy_env_info)
        self.provider = provider
        self.temperature = temperature
        self.debug_mode = debug_mode
        self.agent_id = agent_id
        self.last_action: str | None = None
        self.summary_interval = int(summary_interval) if isinstance(summary_interval, str) else summary_interval

        # Track conversation history for debugging
        self.conversation_history: list[dict] = []

        # Stateful conversation messages (for multi-turn conversations with LLM)
        # Format: [{"role": "user"/"assistant", "content": "..."}, ...]
        self._messages: list[dict[str, str]] = []

        # History summaries - one summary per summary_interval (up to 100)
        # Each summary captures what the agent thought and did in that interval
        self._history_summaries: list[str] = []
        self._max_history_summaries = 100

        # Track actions within current summary interval for summarization
        self._current_window_actions: list[dict[str, str]] = []

        # Step counter for summary intervals (separate from context window)
        self._summary_step_count = 0

        # Track global position (agent starts at 0,0 in global coordinates)
        self._global_x = 0
        self._global_y = 0
        # Track positions visited in current summary interval
        self._current_window_positions: list[tuple[int, int]] = [(0, 0)]
        # Track all positions ever visited (for breadcrumb)
        self._all_visited_positions: set[tuple[int, int]] = {(0, 0)}

        # Track discovered objects with their global positions
        # Format: {object_type: (global_x, global_y)}
        self._discovered_objects: dict[str, tuple[int, int]] = {}

        # Track exploration direction and steps in that direction
        self._current_direction: str | None = None
        self._steps_in_direction: int = 0
        self._direction_change_threshold: int = 8  # Change direction after this many steps

        # Track inventory for smarter decisions
        self._last_inventory: dict[str, int] = {}

        # Initialize prompt builder
        from mettagrid.policy.llm_prompt_builder import LLMPromptBuilder

        self.prompt_builder = LLMPromptBuilder(
            policy_env_info=policy_env_info,
            context_window_size=context_window_size,
            mg_cfg=mg_cfg,
            debug_mode=debug_mode,
            agent_id=agent_id,
        )
        if self.debug_mode:
            logger.info(f"Using dynamic prompts with context window size: {context_window_size}")

        # Initialize observation debugger if debug mode is enabled
        if self.debug_mode:
            self.debugger = ObservationDebugger(policy_env_info)
        else:
            self.debugger = None


        # Initialize LLM client
        # Note: API key validation is handled by LLMMultiAgentPolicy before creating agent policies
        if self.provider == "openai":
            from openai import OpenAI

            self.client: OpenAI | None = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.anthropic_client = None
            self.ollama_client = None
            self.model = model if model else select_openai_model()
        elif self.provider == "anthropic":
            from anthropic import Anthropic

            self.client = None
            self.anthropic_client: Anthropic | None = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.ollama_client = None
            self.model = model if model else select_anthropic_model()
        elif self.provider == "ollama":
            from openai import OpenAI

            self.client = None
            self.anthropic_client = None

            # Ensure Ollama is available and model is pulled (or select if not provided)
            self.model = ensure_ollama_model(model)

            self.ollama_client: OpenAI | None = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # Ollama doesn't need a real API key
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _add_to_messages(self, role: str, content: str) -> None:
        """Add a message to conversation history and prune if needed.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self._messages.append({"role": role, "content": content})

        # Prune to keep only last context_window_size turns (2 messages per turn)
        max_messages = self.prompt_builder.context_window_size * 2
        if len(self._messages) > max_messages:
            # Keep system message (if first) + last N messages
            if self._messages and self._messages[0].get("role") == "system":
                self._messages = [self._messages[0]] + self._messages[-(max_messages - 1):]
            else:
                self._messages = self._messages[-max_messages:]

    def _get_messages_for_api(self, user_prompt: str) -> list[dict[str, str]]:
        """Get messages list for API call, including history + new user prompt.

        Args:
            user_prompt: Current user prompt to add

        Returns:
            List of messages for API call
        """
        # Add current user prompt to history
        self._add_to_messages("user", user_prompt)
        # Return a copy of messages for the API call
        return list(self._messages)

    def _should_show(self, component: str) -> bool:
        """Check if a debug component should be shown.

        Only shows debug output for agent 0 to avoid cluttering the console.

        Args:
            component: Component name to check (e.g., "prompt", "llm", "grid")

        Returns:
            True if component should be shown
        """
        # Only show debug for agent 0
        if self.agent_id != 0:
            return False

        # Handle boolean debug_mode
        if isinstance(self.debug_mode, bool):
            return self.debug_mode
        # Handle set debug_mode
        if isinstance(self.debug_mode, set):
            return "all" in self.debug_mode or component in self.debug_mode
        return False

    def _summarize_current_window(self) -> str:
        """Summarize the current context window's actions into a compact summary.

        Returns:
            A compact string summarizing what the agent did in this window.
        """
        if not self._current_window_actions:
            return ""

        # Get start and end positions for this window
        if self._current_window_positions:
            start_pos = self._current_window_positions[0]
            end_pos = self._current_window_positions[-1]
        else:
            start_pos = (0, 0)
            end_pos = (self._global_x, self._global_y)

        # Build summary with position info
        window_num = len(self._history_summaries) + 1

        # Format position as direction from origin
        def pos_to_dir(x: int, y: int) -> str:
            if x == 0 and y == 0:
                return "origin"
            parts = []
            if y < 0:
                parts.append(f"{abs(y)}N")
            elif y > 0:
                parts.append(f"{y}S")
            if x > 0:
                parts.append(f"{x}E")
            elif x < 0:
                parts.append(f"{abs(x)}W")
            return "".join(parts) if parts else "origin"

        # Get unique positions visited this window (excluding duplicates)
        unique_positions = []
        seen = set()
        for pos in self._current_window_positions:
            if pos not in seen:
                unique_positions.append(pos)
                seen.add(pos)

        # Calculate net direction of movement this window
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        net_direction = []
        if dy < 0:
            net_direction.append("North")
        elif dy > 0:
            net_direction.append("South")
        if dx > 0:
            net_direction.append("East")
        elif dx < 0:
            net_direction.append("West")
        net_dir_str = "-".join(net_direction) if net_direction else "stationary"

        summary = f"[Window {window_num}] {pos_to_dir(*start_pos)} → {pos_to_dir(*end_pos)} (heading {net_dir_str})"
        summary += f" | {len(unique_positions)} new spots, {len(self._all_visited_positions)} total"

        return summary

    def _add_action_to_window(self, action: str, reasoning: str = "") -> None:
        """Track an action for the current context window summary.

        Args:
            action: The action taken
            reasoning: The reasoning behind the action (from LLM response)
        """
        self._current_window_actions.append({"action": action, "reasoning": reasoning})

        # Track direction changes for exploration strategy
        direction_map = {
            "move_north": "north",
            "move_south": "south",
            "move_east": "east",
            "move_west": "west",
        }

        if action in direction_map:
            new_direction = direction_map[action]
            if new_direction == self._current_direction:
                self._steps_in_direction += 1
            else:
                self._current_direction = new_direction
                self._steps_in_direction = 1

        # Update global position based on movement action
        # Note: In MettaGrid, North=Y--, South=Y++, East=X++, West=X--
        if action == "move_north":
            self._global_y -= 1
        elif action == "move_south":
            self._global_y += 1
        elif action == "move_east":
            self._global_x += 1
        elif action == "move_west":
            self._global_x -= 1

        # Track this position
        pos = (self._global_x, self._global_y)
        self._current_window_positions.append(pos)
        self._all_visited_positions.add(pos)

    def _finalize_window_summary(self) -> None:
        """Create summary for current window and reset for next window."""
        if self._current_window_actions:
            summary = self._summarize_current_window()
            if summary:
                self._history_summaries.append(summary)
                # Prune to max size
                if len(self._history_summaries) > self._max_history_summaries:
                    self._history_summaries = self._history_summaries[-self._max_history_summaries:]

                # Always print history summary at window boundary
                print(f"\n[HISTORY Agent {self.agent_id}] {summary}\n")

        # Reset for next window
        self._current_window_actions = []
        # Start new window positions from current position
        self._current_window_positions = [(self._global_x, self._global_y)]

    def _get_history_summary_text(self) -> str:
        """Get formatted history summaries to prepend to prompts.

        Returns:
            Formatted string of past window summaries, or empty string if none.
        """
        if not self._history_summaries:
            return ""

        # Format current position as direction from origin
        def pos_to_dir(x: int, y: int) -> str:
            if x == 0 and y == 0:
                return "origin (starting point)"
            parts = []
            if y < 0:
                parts.append(f"{abs(y)} tiles North")
            elif y > 0:
                parts.append(f"{y} tiles South")
            if x > 0:
                parts.append(f"{x} tiles East")
            elif x < 0:
                parts.append(f"{abs(x)} tiles West")
            return " and ".join(parts) + " of origin"

        # Format window summaries
        window_summaries = "\n".join(f"  {summary}" for summary in self._history_summaries)

        # Load template and substitute variables
        from pathlib import Path
        template_path = Path(__file__).parent / "prompts" / "exploration_history.md"
        template = template_path.read_text()

        return (
            template
            .replace("{{CURRENT_POSITION}}", pos_to_dir(self._global_x, self._global_y))
            .replace("{{TOTAL_EXPLORED}}", str(len(self._all_visited_positions)))
            .replace("{{WINDOW_SUMMARIES}}", window_summaries)
        )

    def _extract_discovered_objects(self, obs: AgentObservation) -> None:
        """Extract and track discovered objects from observation.

        Updates self._discovered_objects with global positions of important objects.

        Args:
            obs: Agent observation
        """
        agent_x = self.policy_env_info.obs_width // 2
        agent_y = self.policy_env_info.obs_height // 2

        # Important object types to track
        important_types = {
            "charger", "assembler", "chest",
            "carbon_extractor", "oxygen_extractor",
            "germanium_extractor", "silicon_extractor",
        }

        for token in obs.tokens:
            if token.feature.name == "tag" and token.value < len(self.policy_env_info.tags):
                tag_name = self.policy_env_info.tags[token.value]
                if tag_name in important_types:
                    # Calculate global position
                    rel_x = token.row() - agent_x
                    rel_y = token.col() - agent_y
                    global_x = self._global_x + rel_x
                    global_y = self._global_y + rel_y

                    # Store with global position (keep closest one if multiple)
                    if tag_name not in self._discovered_objects:
                        self._discovered_objects[tag_name] = (global_x, global_y)

    def _extract_inventory_from_obs(self, obs: AgentObservation) -> dict[str, int]:
        """Extract current inventory from observation.

        Args:
            obs: Agent observation

        Returns:
            Dictionary of resource -> amount
        """
        agent_x = self.policy_env_info.obs_width // 2
        agent_y = self.policy_env_info.obs_height // 2

        inventory = {}
        for token in obs.tokens:
            if token.row() == agent_x and token.col() == agent_y:
                if token.feature.name.startswith("inv:"):
                    resource = token.feature.name[4:]
                    inventory[resource] = token.value
                elif token.feature.name == "agent:energy" or token.feature.name == "energy":
                    inventory["energy"] = token.value

        return inventory

    def _get_discovered_objects_text(self) -> str:
        """Get formatted text of discovered objects for the prompt.

        Returns:
            Formatted string listing discovered objects and their locations.
        """
        if not self._discovered_objects:
            return ""

        def pos_to_dir(x: int, y: int) -> str:
            if x == 0 and y == 0:
                return "at origin"
            parts = []
            if y < 0:
                parts.append(f"{abs(y)}N")
            elif y > 0:
                parts.append(f"{y}S")
            if x > 0:
                parts.append(f"{x}E")
            elif x < 0:
                parts.append(f"{abs(x)}W")
            return "".join(parts)

        lines = ["=== DISCOVERED OBJECTS (from exploration) ==="]
        for obj_type, (gx, gy) in sorted(self._discovered_objects.items()):
            lines.append(f"  - {obj_type}: {pos_to_dir(gx, gy)}")

        return "\n".join(lines)

    def _get_visible_extractors(self, obs: AgentObservation) -> list[str]:
        """Get list of extractor types visible in current observation.

        Args:
            obs: Agent observation

        Returns:
            List of visible extractor type names
        """
        visible = []
        extractor_types = {
            "carbon_extractor", "oxygen_extractor",
            "germanium_extractor", "silicon_extractor",
        }

        for token in obs.tokens:
            if token.feature.name == "tag" and token.value < len(self.policy_env_info.tags):
                tag_name = self.policy_env_info.tags[token.value]
                if tag_name in extractor_types and tag_name not in visible:
                    visible.append(tag_name)

        return visible

    def _get_heart_recipe(self) -> dict[str, int]:
        """Get the heart crafting recipe requirements from assembler protocols.

        Returns:
            Dictionary mapping resource names to required amounts.
            Falls back to default values if no protocol found.
        """
        # Try to get recipe from prompt builder's policy env info
        protocols = self.prompt_builder._policy_env_info.assembler_protocols
        if protocols:
            for protocol in protocols:
                if protocol.output_resources.get("heart", 0) == 1:
                    return {
                        "carbon": protocol.input_resources.get("carbon", 0),
                        "oxygen": protocol.input_resources.get("oxygen", 0),
                        "germanium": protocol.input_resources.get("germanium", 0),
                        "silicon": protocol.input_resources.get("silicon", 0),
                    }

        # Fallback to defaults
        return {"carbon": 10, "oxygen": 10, "germanium": 2, "silicon": 30}

    def _get_strategic_hints(
        self, inventory: dict[str, int], obs: AgentObservation | None = None
    ) -> str:
        """Generate strategic hints based on current state.

        Args:
            inventory: Current inventory
            obs: Optional agent observation to check for visible extractors

        Returns:
            Strategic hints text to add to prompt
        """
        hints = []

        # Get recipe requirements (dynamic based on mission)
        recipe = self._get_heart_recipe()
        req_carbon = recipe["carbon"]
        req_oxygen = recipe["oxygen"]
        req_germanium = recipe["germanium"]
        req_silicon = recipe["silicon"]

        # Check for visible extractors that we need (TOP PRIORITY HINT)
        if obs is not None:
            visible_extractors = self._get_visible_extractors(obs)
            needed_extractors = []

            carbon = inventory.get("carbon", 0)
            oxygen = inventory.get("oxygen", 0)
            germanium = inventory.get("germanium", 0)
            silicon = inventory.get("silicon", 0)

            for ext in visible_extractors:
                if ext == "carbon_extractor" and carbon < req_carbon:
                    needed_extractors.append("carbon_extractor")
                elif ext == "oxygen_extractor" and oxygen < req_oxygen:
                    needed_extractors.append("oxygen_extractor")
                elif ext == "germanium_extractor" and germanium < req_germanium:
                    needed_extractors.append("germanium_extractor")
                elif ext == "silicon_extractor" and silicon < req_silicon:
                    needed_extractors.append("silicon_extractor")

            if needed_extractors:
                ext_list = ", ".join(needed_extractors)
                hints.append(
                    f"🎯 VISIBLE EXTRACTOR YOU NEED: {ext_list} - "
                    "PURSUE IT NOW! Navigate around walls if blocked!"
                )

        # Energy warning
        energy = inventory.get("energy", 100)
        if energy < 20:
            hints.append("⚠️ ENERGY CRITICAL (<20): Find charger IMMEDIATELY!")
        elif energy < 40:
            hints.append("⚠️ ENERGY LOW (<40): Head to charger soon!")

        # Direction change suggestion
        if self._steps_in_direction >= self._direction_change_threshold:
            opposite = {
                "north": "south", "south": "north",
                "east": "west", "west": "east"
            }
            suggested = opposite.get(self._current_direction, "different")
            hints.append(
                f"⚠️ You've gone {self._current_direction} for {self._steps_in_direction} steps. "
                f"Consider going {suggested}!"
            )

        # Distance from origin warning
        distance = abs(self._global_x) + abs(self._global_y)
        if distance > 25:
            hints.append(
                f"⚠️ You're {distance} tiles from origin. "
                "Extractors are usually within 20 tiles - try going back!"
            )

        # Resource gathering hints
        carbon = inventory.get("carbon", 0)
        oxygen = inventory.get("oxygen", 0)
        germanium = inventory.get("germanium", 0)
        silicon = inventory.get("silicon", 0)
        heart = inventory.get("heart", 0)

        if heart > 0 and "chest" in self._discovered_objects:
            hints.append(f"💡 You have {heart} heart(s)! Go to chest to deposit for reward!")
        elif carbon >= req_carbon and oxygen >= req_oxygen and germanium >= req_germanium and silicon >= req_silicon:
            hints.append("💡 You have all resources for a heart! Find an assembler and use heart_a vibe!")
        else:
            missing = []
            if carbon < req_carbon:
                missing.append(f"carbon ({carbon}/{req_carbon})")
            if oxygen < req_oxygen:
                missing.append(f"oxygen ({oxygen}/{req_oxygen})")
            if germanium < req_germanium:
                missing.append(f"germanium ({germanium}/{req_germanium})")
            if silicon < req_silicon:
                missing.append(f"silicon ({silicon}/{req_silicon})")
            if missing:
                hints.append(f"📋 Still need: {', '.join(missing)}")

        if not hints:
            return ""

        return "=== STRATEGIC HINTS ===\n" + "\n".join(hints)

    def step(self, obs: AgentObservation) -> Action:
        """Get action from LLM given observation.

        Args:
            obs: Agent observation

        Returns:
            Action to take
        """
        # Extract and track discovered objects from this observation
        self._extract_discovered_objects(obs)

        # Extract current inventory for strategic hints
        inventory = self._extract_inventory_from_obs(obs)
        self._last_inventory = inventory

        # Increment summary step counter
        self._summary_step_count += 1

        # Check if we're at a summary interval boundary (every N steps)
        at_boundary = (self._summary_step_count - 1) % self.summary_interval == 0
        is_summary_boundary = self._summary_step_count > 1 and at_boundary

        # Check if we're about to start a new context window (before incrementing step counter)
        next_step = self.prompt_builder.step_count + 1
        is_window_boundary = next_step > 1 and (next_step - 1) % self.prompt_builder.context_window_size == 0

        # At summary boundary, finalize the current interval's summary
        if is_summary_boundary:
            self._finalize_window_summary()

        # At context window boundary, also clear conversation messages for fresh window
        if is_window_boundary:
            self._messages = []

        user_prompt, includes_basic_info = self.prompt_builder.context_prompt(obs)

        # Prepend history summaries to prompts that include basic info
        if includes_basic_info and self._history_summaries:
            history_text = self._get_history_summary_text()
            user_prompt = history_text + "\n" + user_prompt

        # Add discovered objects and strategic hints to every prompt
        discovered_text = self._get_discovered_objects_text()
        strategic_hints = self._get_strategic_hints(inventory, obs)

        if discovered_text:
            user_prompt = user_prompt + "\n\n" + discovered_text
        if strategic_hints:
            user_prompt = user_prompt + "\n\n" + strategic_hints

        # Query LLM
        try:
            action_name = "noop"  # Default fallback

            if self.provider == "openai":
                assert self.client is not None
                # GPT-5 and o1 models use different parameters and don't support system messages
                is_gpt5_or_o1 = self.model.startswith("gpt-5") or self.model.startswith("o1")

                # Use stateful conversation history
                messages = self._get_messages_for_api(user_prompt)

                completion_params = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": 150 if is_gpt5_or_o1 else None,
                    "max_tokens": None if is_gpt5_or_o1 else 150,
                    "temperature": None if is_gpt5_or_o1 else self.temperature,
                }
                # Remove None values
                completion_params = {k: v for k, v in completion_params.items() if v is not None}

                # Track prompt
                self.conversation_history.append({
                    "step": len(self.conversation_history) + 1,
                    "prompt": user_prompt,
                    "num_messages": len(messages),
                    "response": None,  # Will be filled in below
                })

                response = self.client.chat.completions.create(**completion_params)
                raw_response = response.choices[0].message.content
                if raw_response is None:
                    raw_response = "noop"

                # Always print LLM response with agent ID
                print(f"[LLM Agent {self.agent_id}] {raw_response}")

                action_name = raw_response.strip()

                # Add assistant response to stateful conversation history
                self._add_to_messages("assistant", action_name)

                # Track response for debugging
                self.conversation_history[-1]["response"] = action_name

                # Track usage and cost
                usage = response.usage
                if usage:
                    LLMAgentPolicy.total_calls += 1
                    LLMAgentPolicy.total_input_tokens += usage.prompt_tokens
                    LLMAgentPolicy.total_output_tokens += usage.completion_tokens

                    # Calculate cost based on model
                    call_cost = calculate_llm_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
                    LLMAgentPolicy.total_cost += call_cost

                    if self.debug_mode:
                        logger.debug(
                            f"OpenAI response: '{action_name}' | "
                            f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out | "
                            f"Cost: ${call_cost:.6f} | "
                            f"Total so far: ${LLMAgentPolicy.total_cost:.4f}"
                        )

            elif self.provider == "ollama":
                assert self.ollama_client is not None

                # Use stateful conversation history
                messages = self._get_messages_for_api(user_prompt)

                # Track prompt
                self.conversation_history.append({
                    "step": len(self.conversation_history) + 1,
                    "prompt": user_prompt,
                    "num_messages": len(messages),
                    "response": None,
                })

                response = self.ollama_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=150,
                )

                if self.debug_mode:
                    # Debug: log the raw response
                    logger.debug(f"Ollama response object: {response}")
                    logger.debug(f"Ollama response choices: {response.choices}")

                # Some models (like gpt-oss) put output in 'reasoning' field instead of 'content'
                message = response.choices[0].message
                raw_response = message.content or ""

                # Check reasoning field if content is empty
                if not raw_response and hasattr(message, "reasoning") and message.reasoning:
                    if self.debug_mode:
                        logger.warning(f"Model used reasoning field instead of content: {message.reasoning[:100]}...")
                    # Try to extract action from reasoning (take last line or last word)
                    raw_response = message.reasoning

                if not raw_response:
                    if self.debug_mode:
                        reasoning = getattr(message, "reasoning", None)
                        logger.error(
                            f"Ollama returned empty response! content='{message.content}', reasoning='{reasoning}'"
                        )
                    raw_response = "noop"

                # Always print LLM response with agent ID
                print(f"[LLM Agent {self.agent_id}] {raw_response}")

                action_name = raw_response.strip()

                # Add assistant response to stateful conversation history
                self._add_to_messages("assistant", action_name)

                # Track response for debugging
                self.conversation_history[-1]["response"] = action_name

                # Track usage (Ollama is free/local)
                usage = response.usage
                if usage:
                    LLMAgentPolicy.total_calls += 1
                    LLMAgentPolicy.total_input_tokens += usage.prompt_tokens
                    LLMAgentPolicy.total_output_tokens += usage.completion_tokens
                    # No cost for local Ollama

                    if self.debug_mode:
                        logger.debug(
                            f"Ollama response: '{action_name}' | "
                            f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out | "
                            f"Cost: $0.00 (local)"
                        )

            elif self.provider == "anthropic":
                assert self.anthropic_client is not None

                # Use stateful conversation history
                messages = self._get_messages_for_api(user_prompt)

                # Track prompt
                self.conversation_history.append({
                    "step": len(self.conversation_history) + 1,
                    "prompt": user_prompt,
                    "num_messages": len(messages),
                    "response": None,
                })

                response = self.anthropic_client.messages.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=150,
                )

                # Extract text from response content blocks
                from anthropic.types import TextBlock

                raw_response = "noop"
                for block in response.content:
                    if isinstance(block, TextBlock):
                        raw_response = block.text
                        break

                # Always print LLM response with agent ID
                print(f"[LLM Agent {self.agent_id}] {raw_response}")

                action_name = raw_response.strip()

                # Add assistant response to stateful conversation history
                self._add_to_messages("assistant", action_name)

                # Track response for debugging
                self.conversation_history[-1]["response"] = action_name

                # Track usage and cost
                usage = response.usage
                LLMAgentPolicy.total_calls += 1
                LLMAgentPolicy.total_input_tokens += usage.input_tokens
                LLMAgentPolicy.total_output_tokens += usage.output_tokens

                # Calculate cost based on model
                call_cost = calculate_llm_cost(self.model, usage.input_tokens, usage.output_tokens)
                LLMAgentPolicy.total_cost += call_cost

                if self.debug_mode:
                    logger.debug(
                        f"Anthropic response: '{action_name}' | "
                        f"Tokens: {usage.input_tokens} in, {usage.output_tokens} out | "
                        f"Cost: ${call_cost:.6f} | "
                        f"Total so far: ${LLMAgentPolicy.total_cost:.4f}"
                    )

            # Parse and return action
            parsed_action, reasoning = self._parse_action(action_name)

            # Track action for history summary (with reasoning if available)
            self._add_action_to_window(parsed_action.name, reasoning)

            # Track last action for debug output
            self.last_action = parsed_action.name

            return parsed_action

        except Exception as e:
            logger.error(f"LLM API error: {e}. Falling back to random action.")
            fallback_action = random.choice(self.policy_env_info.actions.actions())
            self._add_action_to_window(fallback_action.name, "API error fallback")
            self.last_action = fallback_action.name
            return fallback_action

    def _parse_action(self, response_text: str) -> tuple[Action, str]:
        """Parse LLM response and return valid Action and reasoning.

        Handles both JSON format {"reasoning": "...", "action": "..."} and plain action names.

        Args:
            response_text: Raw response from LLM

        Returns:
            Tuple of (Action, reasoning_string)
        """
        # Clean up response
        response_text = response_text.strip()
        reasoning = ""

        # Try to parse as JSON first (expected format)
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "action" in parsed:
                action_name = parsed["action"].strip().lower()
                reasoning = parsed.get("reasoning", "")
                for action in self.policy_env_info.actions.actions():
                    if action.name.lower() == action_name:
                        return action, reasoning
                # If action from JSON doesn't match, fall through to other parsing
        except json.JSONDecodeError:
            pass  # Not valid JSON, try other parsing methods

        # Clean up for non-JSON parsing
        action_name = response_text.strip().strip("\"'").lower()

        # Try exact match first (best case - LLM followed instructions)
        for action in self.policy_env_info.actions.actions():
            if action.name.lower() == action_name:
                return action, reasoning

        # If response contains multiple words, try to extract action from end
        # (LLM might have said "I will move_east" instead of just "move_east")
        words = action_name.split()
        if len(words) > 1:
            # Check last word first (most likely to be the actual action)
            last_word = words[-1].strip(".,!?;:")
            for action in self.policy_env_info.actions.actions():
                if action.name.lower() == last_word:
                    return action, reasoning

            # Check each word from end to start
            for word in reversed(words):
                word = word.strip(".,!?;:")
                for action in self.policy_env_info.actions.actions():
                    if action.name.lower() == word:
                        return action, reasoning

        # Last resort: partial match
        # This is dangerous because it might pick up "don't move_north" as move_north
        for action in self.policy_env_info.actions.actions():
            if action.name.lower() in action_name:
                return action, reasoning

        # Fallback to random action if parsing completely fails
        return random.choice(self.policy_env_info.actions.actions()), reasoning

    @classmethod
    def get_cost_summary(cls) -> dict:
        """Get summary of LLM API usage and costs.

        Returns:
            Dictionary with total_calls, total_tokens, total_input_tokens,
            total_output_tokens, and total_cost
        """
        return {
            "total_calls": cls.total_calls,
            "total_tokens": cls.total_input_tokens + cls.total_output_tokens,
            "total_input_tokens": cls.total_input_tokens,
            "total_output_tokens": cls.total_output_tokens,
            "total_cost": cls.total_cost,
        }

    def print_conversation_history(self) -> None:
        """Print all LLM prompts and responses from this episode."""
        if not self.conversation_history:
            print("\n" + "=" * 70)
            print("No conversation history recorded.")
            print("=" * 70 + "\n")
            return

        print("\n" + "=" * 70)
        print(f"LLM CONVERSATION HISTORY ({len(self.conversation_history)} steps)")
        print("=" * 70 + "\n")

        for entry in self.conversation_history:
            step = entry["step"]
            print(f"{'=' * 70}")
            print(f"STEP {step}")
            print(f"{'=' * 70}")

            # Print system message if present (static prompts only)
            if "system" in entry:
                print("\n[SYSTEM MESSAGE]")
                print(entry["system"])
                print()

            # Print user prompt
            print("[USER PROMPT]")
            print(entry["prompt"])
            print()

            # Print LLM response
            print("[LLM RESPONSE]")
            print(entry.get("response", "(no response)"))
            print()

        print("=" * 70 + "\n")


class LLMMultiAgentPolicy(MultiAgentPolicy):
    """LLM-based multi-agent policy for MettaGrid."""

    short_names = ["llm"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        provider: Literal["openai", "anthropic", "ollama"] = "openai",
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        mg_cfg = None,
    ):
        """Initialize LLM multi-agent policy.

        Args:
            policy_env_info: Policy environment interface
            provider: LLM provider ("openai", "anthropic", or "ollama")
            model: Model name (defaults based on provider)
            temperature: Sampling temperature for LLM
            debug_mode: If True, print human-readable observation debug info (default: True)
            context_window_size: Number of steps before resending basic info (default: 20)
            summary_interval: Number of steps between history summaries (default: 5)
            mg_cfg: Optional MettaGridConfig for extracting game-specific info (chest vibes, etc.)
        """
        super().__init__(policy_env_info)
        self.provider: Literal["openai", "anthropic", "ollama"] = provider
        self.temperature = temperature
        # Handle string "true"/"false" from CLI kwargs
        if isinstance(debug_mode, str):
            self.debug_mode = debug_mode.lower() not in ("false", "0", "no", "")
        else:
            self.debug_mode = bool(debug_mode)
        self.context_window_size = context_window_size
        self.summary_interval = int(summary_interval) if isinstance(summary_interval, str) else summary_interval
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
        if provider in ("openai", "anthropic") and not hasattr(LLMMultiAgentPolicy, '_atexit_registered'):
            atexit.register(_print_cost_summary_on_exit)
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
            debug_mode=self.debug_mode,
            context_window_size=self.context_window_size,
            summary_interval=self.summary_interval,
            mg_cfg=self.mg_cfg,
            agent_id=agent_id,
        )



class LLMGPTMultiAgentPolicy(LLMMultiAgentPolicy):
    """OpenAI GPT-based policy for MettaGrid."""

    short_names = ["llm-openai"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        mg_cfg = None,
    ):
        super().__init__(
            policy_env_info,
            provider="openai",
            model=model,
            temperature=temperature,
            debug_mode=debug_mode,
            context_window_size=context_window_size,
            summary_interval=summary_interval,
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
        debug_mode: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        mg_cfg = None,
    ):
        super().__init__(
            policy_env_info,
            provider="anthropic",
            model=model,
            temperature=temperature,
            debug_mode=debug_mode,
            context_window_size=context_window_size,
            summary_interval=summary_interval,
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
        debug_mode: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        mg_cfg = None,
    ):
        super().__init__(
            policy_env_info,
            provider="ollama",
            model=model,
            temperature=temperature,
            debug_mode=debug_mode,
            context_window_size=context_window_size,
            summary_interval=summary_interval,
            mg_cfg=mg_cfg,
        )
