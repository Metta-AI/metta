"""Model configuration for LLM providers."""

import sys

# Model context window sizes (in tokens)
MODEL_CONTEXT_WINDOWS = {
    # OpenAI models
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-5.1": 128_000,
    "gpt-5.2": 128_000,
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
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]

    # Try partial match (e.g., "llama3.2:latest" -> "llama3.2")
    base_model = model.split(":")[0]
    if base_model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[base_model]

    return None


def estimate_required_context(context_window_size: int, summary_interval: int) -> int:
    """Estimate required context tokens for given config.

    Args:
        context_window_size: Number of conversation turns to keep
        summary_interval: Steps between history summaries

    Returns:
        Estimated required tokens (with 50% safety margin)
    """
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
    """Validate that model has sufficient context window for config.

    Args:
        model: Model name
        context_window_size: Number of conversation turns to keep
        summary_interval: Steps between history summaries

    Exits with error if model context is insufficient.
    """
    # Handle string inputs from CLI
    context_window_size = int(context_window_size) if isinstance(context_window_size, str) else context_window_size
    summary_interval = int(summary_interval) if isinstance(summary_interval, str) else summary_interval

    model_context = get_model_context_window(model)
    required_context = estimate_required_context(context_window_size, summary_interval)

    if model_context is None:
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
        compatible = [
            (name, ctx) for name, ctx in MODEL_CONTEXT_WINDOWS.items()
            if ctx >= required_context
        ]
        compatible.sort(key=lambda x: x[1])
        for name, ctx in compatible[:5]:
            print(f"     - {name} ({ctx:,} tokens)")
        print()
        sys.exit(1)

    usage_pct = (required_context / model_context) * 100
    if usage_pct > 50:
        print(
            f"\n\033[1;33mNote:\033[0m Model '{model}' context usage: ~{usage_pct:.0f}%\n"
            f"  ({required_context:,} / {model_context:,} tokens)\n"
        )
