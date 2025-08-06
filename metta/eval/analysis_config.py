from pydantic import ConfigDict, Field

from metta.agent.policy_store import PolicySelectorConfig
from metta.common.util.config import Config


class AnalysisConfig(Config):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    __init__ = Config.__init__

    # Policy URI to analyze
    policy_uri: str | None = None
    policy_selector: PolicySelectorConfig = Field(default_factory=PolicySelectorConfig)

    # Metrics to analyze
    # Supports globs, e.g. *.reward
    metrics: list[str] = ["*"]

    # Input database
    eval_db_uri: str

    # Filtering options
    suite: str | None = None

    # Output configuration (add these)
    output_path: str | None = None
    num_output_policies: str | int | None = None

    # Curriculum analysis options
    enable_curriculum_analysis: bool = False
    curriculum_oracle_name: str = "oracle"
    curriculum_regret_metrics: list[str] = ["efficiency_regret", "time_regret"]
    curriculum_adaptation_analysis: bool = True
