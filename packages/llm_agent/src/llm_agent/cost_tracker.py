"""Token usage tracking for LLM API calls."""


class CostTracker:
    """Tracks LLM API token usage across all policy instances (singleton)."""

    _instance: "CostTracker | None" = None

    def __new__(cls) -> "CostTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from an API call.

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
        """
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def get_summary(self) -> dict:
        """Get summary of token usage.

        Returns:
            Dictionary with total_calls, total_tokens, total_input_tokens,
            total_output_tokens
        """
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

    def reset(self) -> None:
        """Reset all tracking counters."""
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def print_summary(self) -> None:
        """Print token usage summary to console."""
        summary = self.get_summary()
        if summary["total_calls"] > 0:
            print("\n" + "=" * 60)
            print("LLM API TOKEN USAGE")
            print("=" * 60)
            print(f"Total API calls: {summary['total_calls']}")
            print(f"Total tokens: {summary['total_tokens']:,}")
            print(f"  - Input tokens: {summary['total_input_tokens']:,}")
            print(f"  - Output tokens: {summary['total_output_tokens']:,}")
            print("=" * 60 + "\n")
