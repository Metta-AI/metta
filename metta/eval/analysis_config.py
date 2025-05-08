from typing import Literal

from metta.util.config import Config


class PolicyStatsConfig(Config):
    metrics: list[str]
    # Input database
    eval_db_uri: str

    # Filtering options
    suite: str | None = None
    policy_uri: str | None = None
