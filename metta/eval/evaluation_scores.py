"""Evaluation scores data structure."""

from pydantic import Field

from metta.common.util.typed_config import BaseModelWithForbidExtra


class EvaluationScores(BaseModelWithForbidExtra):
    """Container for evaluation scores from simulations."""

    suite_scores: dict[str, float] = Field(default_factory=dict, description="Average reward for each sim suite")
    simulation_scores: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Average reward for each sim environment (keyed on (suite_name, sim_name))"
    )

    def to_json(self) -> dict[str, dict[str, float] | float]:
        """Convert scores to JSON-friendly format for wandb logging."""
        return {
            "suite_scores": {f"{suite}/score": score for suite, score in self.suite_scores.items()},
            "simulation_scores": {
                f"{suite}/{sim}/score": score for (suite, sim), score in self.simulation_scores.items()
            },
            "reward_avg": sum(self.simulation_scores.values()) / len(self.simulation_scores)
            if self.simulation_scores
            else 0,
        }
