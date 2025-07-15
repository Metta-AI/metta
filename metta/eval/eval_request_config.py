from pydantic import Field

from metta.common.util.typed_config import BaseModelWithForbidExtra


class EvalRewardSummary(BaseModelWithForbidExtra):
    category_scores: dict[str, float] = Field(default_factory=dict, description="Average reward for each category")
    simulation_scores: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Average reward for each simulation (category, short_sim_name)"
    )

    @property
    def avg_category_score(self) -> float:
        return sum(self.category_scores.values()) / len(self.category_scores) if self.category_scores else 0

    @property
    def avg_simulation_score(self) -> float:
        return sum(self.simulation_scores.values()) / len(self.simulation_scores) if self.simulation_scores else 0

    def to_wandb_metrics_format(self) -> dict[str, float]:
        return {
            **{f"{category}/score": score for category, score in self.category_scores.items()},
            **{f"{category}/{sim}": score for (category, sim), score in self.simulation_scores.items()},
        }


class EvalResults(BaseModelWithForbidExtra):
    scores: EvalRewardSummary = Field(..., description="Evaluation scores")
    replay_urls: dict[str, list[str]] = Field(default_factory=dict, description="Replay URLs for each simulation")
