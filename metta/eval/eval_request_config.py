from pydantic import Field

from mettagrid.base_config import Config


class EvalRewardSummary(Config):
    category_scores: dict[str, float] = Field(default_factory=dict, description="Average reward for each category")
    simulation_scores: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Average reward for each simulation (category, short_sim_name)"
    )
    fairness_gap_category_scores: dict[str, float] = Field(
        default_factory=dict, description="Average reward fairness gap (max-min) for each category"
    )
    fairness_gap_simulation_scores: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Average reward fairness gap for each simulation"
    )
    fairness_std_category_scores: dict[str, float] = Field(
        default_factory=dict, description="Average reward fairness standard deviation for each category"
    )
    fairness_std_simulation_scores: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Average reward fairness standard deviation for each simulation"
    )
    fairness_gini_category_scores: dict[str, float] = Field(
        default_factory=dict, description="Average reward fairness Gini coefficient for each category"
    )
    fairness_gini_simulation_scores: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Average reward fairness Gini coefficient for each simulation"
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
            **{f"{category}/fairness_gap": gap for category, gap in self.fairness_gap_category_scores.items()},
            **{
                f"{category}/fairness_gap/{sim}": gap
                for (category, sim), gap in self.fairness_gap_simulation_scores.items()
            },
            **{f"{category}/fairness_std": std for category, std in self.fairness_std_category_scores.items()},
            **{
                f"{category}/fairness_std/{sim}": std
                for (category, sim), std in self.fairness_std_simulation_scores.items()
            },
            **{f"{category}/fairness_gini": gini for category, gini in self.fairness_gini_category_scores.items()},
            **{
                f"{category}/fairness_gini/{sim}": gini
                for (category, sim), gini in self.fairness_gini_simulation_scores.items()
            },
        }


class EvalResults(Config):
    scores: EvalRewardSummary = Field(..., description="Evaluation scores")
    replay_urls: dict[str, list[str]] = Field(default_factory=dict, description="Replay URLs for each simulation")
