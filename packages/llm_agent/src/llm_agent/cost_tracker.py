"""Cost tracking for LLM API usage."""

from llm_agent.model_config import calculate_llm_cost


class CostTracker:
    # Singleton instance
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
        self.total_cost = 0.0

    def record_usage(self, model: str, input_tokens: int, output_tokens: int) -> float:
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        call_cost = calculate_llm_cost(model, input_tokens, output_tokens)
        self.total_cost += call_cost

        return call_cost

    def get_summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
        }

    def reset(self) -> None:
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def print_summary(self) -> None:
        """Print cost summary to console."""
        summary = self.get_summary()
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
