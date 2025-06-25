from common.metta.util.config import Config
from metta.agent.policy_store import PolicySelectorConfig


class AnalysisConfig(Config):
    __init__ = Config.__init__
    # Policy URI to analyze
    policy_uri: str | None = None
    policy_selector: PolicySelectorConfig = PolicySelectorConfig()

    # Metrics to analyze
    # Supports globs, e.g. *.reward
    metrics: list[str] = ["*"]

    # Input database
    eval_db_uri: str

    # Filtering options
    suite: str | None = None
